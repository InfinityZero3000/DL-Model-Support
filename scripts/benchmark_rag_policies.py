# pyright: reportMissingImports=false

# File này sẽ thực hiện đánh giá GraphCAG với model Qwen3 sau khi đã fine-tune trên tập dữ liệu Wi+Locness. Kịch bản sẽ chạy hai cấu hình:
# 1. Full-pipeline baseline: tắt cache và sử dụng retrieval đầy đủ cho mọi truy vấn.
# 2. RAPID-GraphCAG: bật cache và sử dụng retrieval có ngân sách (retrieval_policy=rapid).
# Sau đó, nó sẽ so sánh độ trễ, quyết định cache, và các probe kiểm tra tính chọn lọc của cache giữa hai cấu hình.
# Lưu ý: Không chạy benchmark này để so sánh các Paper

"""Benchmark full-pipeline baseline vs RAPID-GraphCAG policies.

This script runs the LexiLingo GraphCAG pipeline in two configurations:
- Full-pipeline baseline: cache off + full retrieval
- RAPID-GraphCAG: cache on + budgeted retrieval (retrieval_policy=rapid)

It reports latency distribution, cache decisions, and controlled-drift probes.

The benchmark accepts either:
- flat JSONL samples with a top-level ``text`` field, or
- instruction/chat JSONL samples with ``messages`` and optional ``task`` metadata.

Usage (from model-development root):
  python scripts/benchmark_rag_policies.py --n 20 --repeats 2

Notes:
- Requires the ai-service dependencies installed (sentence-transformers, langgraph, etc.).
- If Redis is unavailable, cache-on mode will behave like cache-off.
- ``generation_policy=auto`` exercises the runtime LLM chain (Groq -> Gemini -> Ollama).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]  # LexiLingo repo root (contains ai-service/)
MODEL_DEV_ROOT = Path(__file__).resolve().parents[1]  # model-development/ (contains datasets/)
DEFAULT_DATASET = MODEL_DEV_ROOT / "datasets" / "wi+locness" / "json" / "A.dev.json"
DATASET_PRESETS: dict[str, Path] = {
    "wi_locness": DEFAULT_DATASET,
    "graphcag_drift_probes": MODEL_DEV_ROOT / "datasets" / "benchmarks" / "graphcag_drift_probes" / "validation.jsonl",
    "hotpotqa": MODEL_DEV_ROOT / "datasets" / "benchmarks" / "hotpotqa" / "validation.jsonl",
    "2wikimultihopqa": MODEL_DEV_ROOT / "datasets" / "benchmarks" / "2wikimultihopqa" / "validation.jsonl",
}


@dataclass(frozen=True)
class RunResult:
    query: str
    level: str
    latency_ms: int
    cache_hit: bool
    cache_decision: str
    cache_layer: str
    reuse_risk: float
    path: str
    tutor_response: str
    correction_count: int
    fluency_score: float
    grammar_score: float
    overall_score: float
    diagnosis_intent: str
    models_used: list[str]


@dataclass(frozen=True)
class BenchmarkSample:
    text: str
    task: str | None
    expected: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CacheProbeResult:
    probe: str
    category: str
    first_query: str
    second_query: str
    first_level: str
    second_level: str
    second_cache_hit: bool
    second_cache_decision: str
    second_cache_layer: str
    second_reuse_risk: float
    second_latency_ms: int
    response_match: bool


@dataclass(frozen=True)
class CacheProbeSpec:
    probe: str
    category: str
    first_query: str
    second_query: str
    first_level: str
    second_level: str


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))

    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[f])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


def _try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _extract_query_text(obj: dict[str, Any]) -> str:
    text = str(obj.get("text") or "").strip()
    if text:
        return text

    input_text = str(obj.get("input") or "").strip()
    if input_text:
        return input_text

    metadata = obj.get("metadata") or {}
    raw_text = str(metadata.get("raw_text") or "").strip()
    if raw_text:
        return raw_text

    messages = obj.get("messages") or []
    for msg in messages:
        if str(msg.get("role") or "").lower() == "user":
            content = str(msg.get("content") or "").strip()
            if content:
                return content

    return ""


def _iter_dataset_objects(path: Path) -> Iterable[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(parsed, dict):
        if isinstance(parsed.get("data"), list):
            for item in parsed["data"]:
                if isinstance(item, dict):
                    yield item
            return
        yield parsed
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _iter_dataset_samples(path: Path, task_filter: str | None = None) -> Iterable[BenchmarkSample]:
    for obj in _iter_dataset_objects(path):
        task = str(obj.get("task") or "").strip().lower() or None
        if task_filter and task != task_filter.lower():
            continue

        text = _extract_query_text(obj)
        if not text:
            continue

        expected = obj.get("output")
        if expected is None:
            messages = obj.get("messages") or []
            for msg in messages:
                if str(msg.get("role") or "").lower() == "assistant":
                    expected = _try_parse_json(msg.get("content"))
                    break

        raw_metadata = obj.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        yield BenchmarkSample(text=text, task=task, expected=expected, metadata=metadata)


def _normalize_query(text: str, max_chars: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _resolve_dataset_path(dataset: Path | None, dataset_preset: str | None) -> Path:
    if dataset_preset:
        preset = DATASET_PRESETS.get(dataset_preset)
        if preset is None:
            raise SystemExit(f"Unknown dataset preset: {dataset_preset}")
        return preset
    return dataset or DEFAULT_DATASET


def _normalize_response(text: str) -> str:
    return " ".join(text.lower().split())


async def _build_pipeline(*, use_gemini_fallback: bool) -> Any:
    import sys

    # ai-service Settings expects DEBUG to be a parseable boolean. Some local
    # setups use values like "release", which will crash BaseSettings parsing.
    raw_debug = os.getenv("DEBUG", "").strip().lower()
    if raw_debug and raw_debug not in {"0", "1", "true", "false", "yes", "no", "on", "off"}:
        os.environ["DEBUG"] = "false"

    # KuzuDB writes a WAL file; default path lives under ai-service/, which may be
    # read-only in some sandboxed runs. Point it to a writable location.
    os.environ.setdefault("KUZU_DB_PATH", str(Path("/tmp") / "graphcag_kuzu"))

    ai_service_root = REPO_ROOT / "ai-service"
    if str(ai_service_root) not in sys.path:
        sys.path.insert(0, str(ai_service_root))

    get_graph_cag = importlib.import_module("api.services.graph_cag.graph").get_graph_cag

    try:
        setup_gateway = importlib.import_module("api.services.gateway_setup").setup_gateway

        await setup_gateway(
            max_memory_mb=8000,
            enable_auto_unload=False,
            use_gemini_fallback=use_gemini_fallback,
        )
    except Exception:
        pass

    return await get_graph_cag()


async def _invoke_pipeline(
    pipeline: Any,
    *,
    query: str,
    level: str,
    session_id: str,
    cache_policy: str,
    retrieval_policy: str,
    diagnosis_policy: str,
    generation_policy: str,
) -> RunResult:
    start = time.perf_counter()
    out: dict[str, Any] = await pipeline.analyze(
        query,
        session_id=session_id,
        learner_profile={"level": level},
        cache_policy=cache_policy,
        retrieval_policy=retrieval_policy,
        diagnosis_policy=diagnosis_policy,
        generation_policy=generation_policy,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    meta = out.get("metadata") or {}
    scores = out.get("scores") or {}
    corrections = out.get("corrections") or []

    return RunResult(
        query=query,
        level=level,
        latency_ms=int(meta.get("latency_ms") or elapsed_ms),
        cache_hit=bool(meta.get("cache_hit") or False),
        cache_decision=str(meta.get("cache_decision") or "full"),
        cache_layer=str(meta.get("cache_layer") or "none"),
        reuse_risk=float(meta.get("reuse_risk") or 1.0),
        path=str(meta.get("path") or ""),
        tutor_response=str(out.get("tutor_response") or ""),
        correction_count=len(corrections),
        fluency_score=float(scores.get("fluency") or 0.0),
        grammar_score=float(scores.get("grammar") or 0.0),
        overall_score=float(scores.get("overall") or 0.0),
        diagnosis_intent=str(meta.get("diagnosis_intent") or "unknown"),
        models_used=[str(item) for item in (meta.get("models_used") or [])],
    )


async def _run_mode(
    *,
    mode_name: str,
    queries: list[str],
    level: str,
    cache_policy: str,
    retrieval_policy: str,
    diagnosis_policy: str,
    generation_policy: str,
    use_gemini_fallback: bool,
) -> list[RunResult]:
    pipeline = await _build_pipeline(use_gemini_fallback=use_gemini_fallback)

    results: list[RunResult] = []

    # Warm-up once to pay model import/graph compile cost.
    try:
        await pipeline.analyze(
            "Warm-up: I goes to school yesterday.",
            session_id=f"bench_warm_{mode_name}",
            learner_profile={"level": level},
            cache_policy=cache_policy,
            retrieval_policy=retrieval_policy,
            diagnosis_policy=diagnosis_policy,
            generation_policy=generation_policy,
        )
    except Exception:
        # Warm-up failures shouldn't crash the benchmark.
        pass

    for i, q in enumerate(queries):
        session_id = f"bench_{mode_name}_{i}_{random.randint(1, 1_000_000)}"
        results.append(
            await _invoke_pipeline(
                pipeline,
                query=q,
                level=level,
                session_id=session_id,
                cache_policy=cache_policy,
                retrieval_policy=retrieval_policy,
                diagnosis_policy=diagnosis_policy,
                generation_policy=generation_policy,
            )
        )

    return results


def _summarize(mode_name: str, results: list[RunResult]) -> dict[str, Any]:
    latencies = [r.latency_ms for r in results]
    hit_latencies = [r.latency_ms for r in results if r.cache_hit]
    miss_latencies = [r.latency_ms for r in results if not r.cache_hit]
    reuse_count = sum(1 for r in results if r.cache_decision == "reuse")
    patch_count = sum(1 for r in results if r.cache_decision == "patch")

    summary: dict[str, Any] = {
        "mode": mode_name,
        "n": len(results),
        "cache_hit_rate": (sum(1 for r in results if r.cache_hit) / len(results)) if results else 0.0,
        "reuse_rate": (reuse_count / len(results)) if results else 0.0,
        "patch_rate": (patch_count / len(results)) if results else 0.0,
        "lat_ms_mean": statistics.mean(latencies) if latencies else float("nan"),
        "lat_ms_p50": _percentile([float(x) for x in latencies], 50),
        "lat_ms_p95": _percentile([float(x) for x in latencies], 95),
        "hit_ms_mean": statistics.mean(hit_latencies) if hit_latencies else float("nan"),
        "miss_ms_mean": statistics.mean(miss_latencies) if miss_latencies else float("nan"),
    }
    return summary


def _compare_modes(reference: list[RunResult], candidate: list[RunResult]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("Reference and candidate runs must have the same length")

    response_matches = 0
    correction_matches = 0
    score_deltas: list[float] = []

    for ref, cand in zip(reference, candidate):
        if _normalize_response(ref.tutor_response) == _normalize_response(cand.tutor_response):
            response_matches += 1
        if ref.correction_count == cand.correction_count:
            correction_matches += 1
        score_deltas.append(abs(ref.overall_score - cand.overall_score))

    n = len(reference) or 1
    return {
        "response_exact_match_rate": response_matches / n,
        "correction_count_match_rate": correction_matches / n,
        "overall_score_delta_mean": statistics.mean(score_deltas) if score_deltas else 0.0,
        "overall_score_delta_max": max(score_deltas) if score_deltas else 0.0,
    }


def _build_cache_probe_pairs(base_query: str) -> list[CacheProbeSpec]:
    exact = base_query
    case_trim = f"  {base_query.upper()}  "
    internal_ws = "  ".join(base_query.split())
    punctuation = base_query.rstrip(".!?") + "!"
    polite_wrapper = f"Please correct this sentence: {base_query}"
    explain_wrapper = f"Please explain the mistake in this sentence: {base_query}"
    practice_wrapper = f"Create a short practice exercise from this sentence: {base_query}"

    return [
        CacheProbeSpec("exact_repeat", "safe_repeat", exact, exact, "B1", "B1"),
        CacheProbeSpec("case_trim_variant", "benign_surface", exact, case_trim, "B1", "B1"),
        CacheProbeSpec("internal_whitespace_variant", "benign_surface", exact, internal_ws, "B1", "B1"),
        CacheProbeSpec("punctuation_variant", "benign_surface", exact, punctuation, "B1", "B1"),
        CacheProbeSpec("polite_wrapper_variant", "wrapper_surface", exact, polite_wrapper, "B1", "B1"),
        CacheProbeSpec("intent_explain_variant", "intent_shift", exact, explain_wrapper, "B1", "B1"),
        CacheProbeSpec("intent_practice_variant", "intent_shift", exact, practice_wrapper, "B1", "B1"),
        CacheProbeSpec("cross_level_variant", "profile_shift", exact, exact, "B1", "B2"),
    ]


async def _run_cache_selectivity_probes(
    queries: list[str],
    *,
    diagnosis_policy: str,
    generation_policy: str,
    use_gemini_fallback: bool,
) -> list[CacheProbeResult]:
    pipeline = await _build_pipeline(use_gemini_fallback=use_gemini_fallback)
    results: list[CacheProbeResult] = []

    for idx, query in enumerate(queries):
        for spec in _build_cache_probe_pairs(query):
            first = await _invoke_pipeline(
                pipeline,
                query=spec.first_query,
                level=spec.first_level,
                session_id=f"probe_{spec.probe}_{idx}_first_{random.randint(1, 1_000_000)}",
                cache_policy="on",
                retrieval_policy="rapid",
                diagnosis_policy=diagnosis_policy,
                generation_policy=generation_policy,
            )
            second = await _invoke_pipeline(
                pipeline,
                query=spec.second_query,
                level=spec.second_level,
                session_id=f"probe_{spec.probe}_{idx}_second_{random.randint(1, 1_000_000)}",
                cache_policy="on",
                retrieval_policy="rapid",
                diagnosis_policy=diagnosis_policy,
                generation_policy=generation_policy,
            )
            results.append(
                CacheProbeResult(
                    probe=spec.probe,
                    category=spec.category,
                    first_query=spec.first_query,
                    second_query=spec.second_query,
                    first_level=spec.first_level,
                    second_level=spec.second_level,
                    second_cache_hit=second.cache_hit,
                    second_cache_decision=second.cache_decision,
                    second_cache_layer=second.cache_layer,
                    second_reuse_risk=second.reuse_risk,
                    second_latency_ms=second.latency_ms,
                    response_match=_normalize_response(first.tutor_response)
                    == _normalize_response(second.tutor_response),
                )
            )

    return results


def _summarize_probes(results: list[CacheProbeResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CacheProbeResult]] = {}
    for result in results:
        grouped.setdefault(result.probe, []).append(result)

    summaries: list[dict[str, Any]] = []
    for probe, probe_results in grouped.items():
        summaries.append(
            {
                "probe": probe,
                "category": probe_results[0].category if probe_results else "unknown",
                "n": len(probe_results),
                "cache_hit_rate": sum(1 for r in probe_results if r.second_cache_hit) / len(probe_results),
                "reuse_rate": sum(1 for r in probe_results if r.second_cache_decision == "reuse") / len(probe_results),
                "patch_rate": sum(1 for r in probe_results if r.second_cache_decision == "patch") / len(probe_results),
                "l0_hit_rate": sum(1 for r in probe_results if r.second_cache_layer == "L0") / len(probe_results),
                "l1_hit_rate": sum(1 for r in probe_results if r.second_cache_layer == "L1") / len(probe_results),
                "response_match_rate": sum(1 for r in probe_results if r.response_match) / len(probe_results),
                "reuse_risk_mean": statistics.mean(r.second_reuse_risk for r in probe_results),
                "lat_ms_mean": statistics.mean(r.second_latency_ms for r in probe_results),
            }
        )

    return sorted(summaries, key=lambda item: item["probe"])


def _print_probe_summary(summaries: list[dict[str, Any]]) -> None:
    headers = ["Probe", "Category", "N", "2ndHitRate", "Reuse", "Patch", "L0", "L1", "RiskMean", "RespMatch", "2ndMean(ms)"]
    rows = [
        [
            str(item["probe"]),
            str(item["category"]),
            str(item["n"]),
            f"{item['cache_hit_rate']*100:.1f}%",
            f"{item['reuse_rate']*100:.1f}%",
            f"{item['patch_rate']*100:.1f}%",
            f"{item['l0_hit_rate']*100:.1f}%",
            f"{item['l1_hit_rate']*100:.1f}%",
            f"{item['reuse_risk_mean']:.3f}",
            f"{item['response_match_rate']*100:.1f}%",
            f"{item['lat_ms_mean']:.1f}",
        ]
        for item in summaries
    ]

    col_widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))
    ]

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(cols[i].ljust(col_widths[i]) for i in range(len(cols)))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "Mode",
        "N",
        "HitRate",
        "Reuse",
        "Patch",
        "Mean(ms)",
        "P50(ms)",
        "P95(ms)",
        "HitMean(ms)",
        "MissMean(ms)",
    ]

    rows = []
    for s in summaries:
        rows.append(
            [
                s["mode"],
                str(s["n"]),
                f"{s['cache_hit_rate']*100:.1f}%",
                f"{s['reuse_rate']*100:.1f}%",
                f"{s['patch_rate']*100:.1f}%",
                f"{s['lat_ms_mean']:.1f}",
                f"{s['lat_ms_p50']:.1f}",
                f"{s['lat_ms_p95']:.1f}",
                f"{s['hit_ms_mean']:.1f}",
                f"{s['miss_ms_mean']:.1f}",
            ]
        )

    col_widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))
    ]

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(cols[i].ljust(col_widths[i]) for i in range(len(cols)))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to JSONL dataset")
    parser.add_argument(
        "--dataset-preset",
        type=str,
        default=None,
        choices=sorted(DATASET_PRESETS.keys()),
        help="Named benchmark dataset preset under datasets/benchmarks/",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of unique samples")
    parser.add_argument("--repeats", type=int, default=2, help="How many times to repeat each sample")
    parser.add_argument("--level", type=str, default="B1", help="Learner CEFR level")
    parser.add_argument("--task", type=str, default=None, help="Optional task filter for chat-style JSONL")
    parser.add_argument(
        "--diagnosis-policy",
        type=str,
        default="rules",
        choices=["rules", "auto"],
        help="Diagnosis mode for both baseline and RAPID runs",
    )
    parser.add_argument(
        "--generation-policy",
        type=str,
        default="template",
        choices=["template", "auto"],
        help="Generation mode for both baseline and RAPID runs",
    )
    parser.add_argument(
        "--enable-gemini-fallback",
        action="store_true",
        help="Register Gemini in the gateway so auto modes can fall back to cloud APIs",
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=8,
        help="Number of unique samples to use for cache selectivity probes",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to save a JSON report with summaries and probes",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_path = _resolve_dataset_path(args.dataset, args.dataset_preset)

    samples = list(_iter_dataset_samples(dataset_path, task_filter=args.task))
    if not samples:
        if args.task:
            raise SystemExit(f"No texts found in dataset: {dataset_path} for task={args.task}")
        raise SystemExit(f"No texts found in dataset: {dataset_path}")

    sampled = random.sample(samples, k=min(args.n, len(samples)))
    unique_queries = [_normalize_query(sample.text) for sample in sampled]
    task_counts: dict[str, int] = {}
    for sample in sampled:
        task_counts[sample.task or "unknown"] = task_counts.get(sample.task or "unknown", 0) + 1

    queries: list[str] = []
    for q in unique_queries:
        queries.extend([q] * max(1, args.repeats))

    print(f"Dataset: {dataset_path}")
    if args.dataset_preset:
        print(f"Dataset preset: {args.dataset_preset}")
    if args.task:
        print(f"Task filter: {args.task}")
    print(f"Unique samples: {len(unique_queries)} | Repeats: {args.repeats} | Total runs: {len(queries)}")
    print(f"Task mix: {task_counts}")
    print(
        "Policies: diagnosis={} | generation={} | gemini_fallback={}".format(
            args.diagnosis_policy,
            args.generation_policy,
            args.enable_gemini_fallback,
        )
    )

    async def runner() -> None:
        summaries: list[dict[str, Any]] = []

        # Full-pipeline baseline: no cache, full retrieval every time.
        baseline_results = await _run_mode(
            mode_name="full_pipeline_baseline",
            queries=queries,
            level=args.level,
            cache_policy="off",
            retrieval_policy="full",
            diagnosis_policy=args.diagnosis_policy,
            generation_policy=args.generation_policy,
            use_gemini_fallback=args.enable_gemini_fallback,
        )
        summaries.append(_summarize("Full-pipeline baseline (no-cache, full)", baseline_results))

        # RAPID-GraphCAG: cache enabled + budgeted retrieval.
        rapid_results = await _run_mode(
            mode_name="rapid_graphcag",
            queries=queries,
            level=args.level,
            cache_policy="on",
            retrieval_policy="rapid",
            diagnosis_policy=args.diagnosis_policy,
            generation_policy=args.generation_policy,
            use_gemini_fallback=args.enable_gemini_fallback,
        )
        summaries.append(_summarize("RAPID-GraphCAG (cache, rapid)", rapid_results))

        parity = _compare_modes(baseline_results, rapid_results)
        probe_queries = unique_queries[: max(1, min(args.probe_samples, len(unique_queries)))]
        probe_results = await _run_cache_selectivity_probes(
            probe_queries,
            diagnosis_policy=args.diagnosis_policy,
            generation_policy=args.generation_policy,
            use_gemini_fallback=args.enable_gemini_fallback,
        )
        probe_summaries = _summarize_probes(probe_results)

        print("\nResults:")
        _print_summary(summaries)

        print("\nBehavioral Parity:")
        print(
            "ResponseExactMatch={:.1f}% | CorrectionCountMatch={:.1f}% | MeanScoreDelta={:.4f} | MaxScoreDelta={:.4f}".format(
                parity["response_exact_match_rate"] * 100,
                parity["correction_count_match_rate"] * 100,
                parity["overall_score_delta_mean"],
                parity["overall_score_delta_max"],
            )
        )

        print("\nCache Selectivity Probes:")
        _print_probe_summary(probe_summaries)

        if args.report_json:
            report = {
                "dataset": str(dataset_path),
                "dataset_preset": args.dataset_preset,
                "task": args.task,
                "level": args.level,
                "seed": args.seed,
                "diagnosis_policy": args.diagnosis_policy,
                "generation_policy": args.generation_policy,
                "enable_gemini_fallback": args.enable_gemini_fallback,
                "unique_samples": len(unique_queries),
                "repeats": args.repeats,
                "task_mix": task_counts,
                "summaries": summaries,
                "behavioral_parity": parity,
                "probe_summaries": probe_summaries,
                "probe_results": [asdict(result) for result in probe_results],
            }
            args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nJSON report saved to: {args.report_json}")

    asyncio.run(runner())


if __name__ == "__main__":
    main()
