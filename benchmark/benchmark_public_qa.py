# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_rag_policies import DATASET_PRESETS, _build_pipeline, _iter_dataset_samples

_LOG = logging.getLogger("benchmark")
_DEFAULT_GROQ_WAIT_BEFORE_GEMINI_SECONDS = 75.0


def _current_utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _setup_logging(log_path: Path | None) -> None:
    """Configure two-destination logging: DEBUG+ to file, WARNING+ to stderr."""
    _LOG.setLevel(logging.DEBUG)
    ts_fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(ts_fmt)
        _LOG.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    _LOG.addHandler(sh)


def _render_progress(done: int, total: int, results: list) -> None:
    """Overwrite current terminal line with a live progress bar + running metrics."""
    pct = done / max(total, 1)
    bar_filled = int(20 * pct)
    bar = "\u2588" * bar_filled + "\u2591" * (20 - bar_filled)
    if results:
        f1 = statistics.mean(r.token_f1 for r in results)
        em = statistics.mean(r.exact_match for r in results)
        lat = statistics.mean(r.latency_ms for r in results)
        cache_cnt = sum(1 for r in results if r.cache_hit)
        suffix = f"F1={f1*100:4.1f}%  EM={em*100:4.1f}%  lat={lat:5.0f}ms  cache={cache_cnt}/{done}"
    else:
        suffix = ""
    print(f"\r  [{bar}] {done:3}/{total}  {suffix}", end="", flush=True)


@dataclass(frozen=True)
class QaRunResult:
    dataset: str
    mode: str
    task: str
    query: str
    prediction: str
    references: list[str]
    latency_ms: int
    cache_hit: bool
    cache_decision: str
    cache_layer: str
    reuse_risk: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_5: float
    exact_match: float
    token_f1: float
    rouge_l_f1: float
    bleu1: float
    models_used: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    effective_prompt_tokens: float = 0.0
    prompt_cache_token_rate: float = 0.0
    prompt_discount_savings_rate: float = 0.0
    usage_source: str = "none"
    expected_cache_decision: str | None = None
    expected_safe_reuse: bool | None = None
    expected_level_isolation: bool | None = None
    request_level: str | None = None
    cached_level: str | None = None
    drift_label_source: str = "none"
    llm_provider: str = "groq"


@dataclass(frozen=True)
class ModeConfig:
    name: str
    label: str
    cache_policy: str
    retrieval_policy: str
    benchmark_ranker: str
    description: str


MODE_CONFIGS: dict[str, ModeConfig] = {
    "cag_vanilla": ModeConfig(
        name="cag_vanilla",
        label="Vanilla CAG",
        cache_policy="on",
        retrieval_policy="full",
        benchmark_ranker="flat",
        description="Flat reusable-context baseline without graph-indexed modular reuse.",
    ),
    "cag_flat": ModeConfig(
        name="cag_flat",
        label="CAG flat ablation",
        cache_policy="off",
        retrieval_policy="full",
        benchmark_ranker="flat",
        description="Legacy flat ablation without cache; kept only for backwards compatibility.",
    ),
    "graphcag_full": ModeConfig(
        name="graphcag_full",
        label="GraphCAG graph ablation",
        cache_policy="off",
        retrieval_policy="full",
        benchmark_ranker="graph",
        description="Graph-aware ranking on the same candidate set, cache disabled, useful as a graph-only ablation.",
    ),
    "hipporag_proxy": ModeConfig(
        name="hipporag_proxy",
        label="HippoRAG proxy",
        cache_policy="off",
        retrieval_policy="full",
        benchmark_ranker="memory",
        description="Memory-propagation proxy over the candidate graph, not a full HippoRAG implementation.",
    ),
    "graphcag_rapid": ModeConfig(
        name="graphcag_rapid",
        label="GraphCAG",
        cache_policy="on",
        retrieval_policy="rapid",
        benchmark_ranker="graph",
        description="Graph-aware ranking plus RAPID cache, intended as the main GraphCAG mode.",
    ),
}


COMPARISON_PROFILES: dict[str, list[str]] = {
    "public_cag_compare": ["cag_vanilla", "hipporag_proxy", "graphcag_rapid"],
    "public_cag_quality": ["cag_vanilla", "hipporag_proxy", "graphcag_rapid"],
    "public_cag_ablation": ["cag_vanilla", "graphcag_full", "graphcag_rapid"],
    "public_graph_compare": ["cag_flat", "graphcag_full", "hipporag_proxy", "graphcag_rapid"],
    "public_graph_quality": ["cag_flat", "graphcag_full", "hipporag_proxy"],
    "public_graph_efficiency": ["cag_flat", "graphcag_rapid"],
    "state_drift": ["cag_vanilla", "hipporag_proxy", "graphcag_rapid"],
}


DATASET_RATIONALE: dict[str, str] = {
    "graphcag_drift_probes": "Curated GraphCAG drift-probe set with exact-hit, near-hit, and unsafe drift cases labeled for PCC evaluation.",
    "hotpotqa": "Best default for graph-vs-non-graph comparison because bridge/comparison questions reward multi-hop evidence chaining.",
    "2wikimultihopqa": "Good second multi-hop benchmark for entity-link chains and compositional questions.",
}

# State Drift datasets — only multi-hop benchmarks expose meaningful PCC drift
STATE_DRIFT_DATASETS: list[str] = ["graphcag_drift_probes", "hotpotqa", "2wikimultihopqa"]

_CACHED_INPUT_DISCOUNT = 0.5

_DECISION_KEYS = (
    "expected_cache_decision",
    "gold_cache_decision",
    "drift_expected_decision",
    "pcc_expected_decision",
)
_SAFE_REUSE_KEYS = (
    "expected_safe_reuse",
    "gold_safe_reuse",
    "drift_safe_reuse",
    "pcc_safe_reuse",
)
_LEVEL_ISOLATION_KEYS = (
    "expected_level_isolation",
    "gold_level_isolation",
    "drift_level_isolation",
    "pcc_level_isolation",
)
_REQUEST_LEVEL_KEYS = ("request_level", "benchmark_level", "level")
_CACHED_LEVEL_KEYS = ("cached_level", "candidate_level", "source_level")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "safe", "allow", "allowed"}:
        return True
    if text in {"0", "false", "no", "n", "unsafe", "deny", "blocked"}:
        return False
    return None


def _coerce_decision(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"reuse", "l0", "l1", "hit"}:
        return "reuse"
    if text in {"patch", "repair", "partial", "near_hit"}:
        return "patch"
    if text in {"full", "miss", "reconstruct", "l2", "fallback"}:
        return "full"
    return None


def _first_present(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _iter_usage_candidates(output: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for container in (metadata, output):
        if not isinstance(container, dict):
            continue
        for key in (
            "usage",
            "llm_usage",
            "token_usage",
            "provider_usage",
            "response_usage",
            "usage_metadata",
        ):
            value = container.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        response_obj = container.get("response")
        if isinstance(response_obj, dict):
            usage = response_obj.get("usage")
            if isinstance(usage, dict):
                candidates.append(usage)
    return candidates


def _extract_token_usage(output: dict[str, Any], metadata: dict[str, Any], query: str, prediction: str) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    usage_source = "none"

    for usage in _iter_usage_candidates(output, metadata):
        prompt_tokens = max(prompt_tokens, _coerce_int(usage.get("prompt_tokens") or usage.get("prompt_eval_count")))
        completion_tokens = max(
            completion_tokens,
            _coerce_int(usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("eval_count")),
        )
        total_tokens = max(total_tokens, _coerce_int(usage.get("total_tokens")))

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_tokens = max(cached_tokens, _coerce_int(prompt_details.get("cached_tokens")))
        cached_tokens = max(cached_tokens, _coerce_int(usage.get("cached_tokens")))

        if prompt_tokens or completion_tokens or total_tokens or cached_tokens:
            usage_source = "provider"

    if usage_source == "none":
        prompt_tokens = _estimate_tokens(query)
        completion_tokens = _estimate_tokens(prediction)
        total_tokens = prompt_tokens + completion_tokens
        usage_source = "estimated"
    elif total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    effective_prompt_tokens = max(float(prompt_tokens) - (_CACHED_INPUT_DISCOUNT * float(cached_tokens)), 0.0)
    prompt_cache_token_rate = (float(cached_tokens) / float(prompt_tokens)) if prompt_tokens > 0 else 0.0
    prompt_discount_savings_rate = (
        (_CACHED_INPUT_DISCOUNT * float(cached_tokens)) / float(prompt_tokens)
        if prompt_tokens > 0
        else 0.0
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "effective_prompt_tokens": effective_prompt_tokens,
        "prompt_cache_token_rate": prompt_cache_token_rate,
        "prompt_discount_savings_rate": prompt_discount_savings_rate,
        "usage_source": usage_source,
    }


def _extract_drift_labels(metadata: dict[str, Any], level: str) -> dict[str, Any]:
    expected_cache_decision = _coerce_decision(_first_present(metadata, _DECISION_KEYS))
    expected_safe_reuse = _coerce_bool(_first_present(metadata, _SAFE_REUSE_KEYS))
    expected_level_isolation = _coerce_bool(_first_present(metadata, _LEVEL_ISOLATION_KEYS))
    request_level = _first_present(metadata, _REQUEST_LEVEL_KEYS)
    cached_level = _first_present(metadata, _CACHED_LEVEL_KEYS)

    request_level_str = str(request_level).strip() if request_level is not None else str(level).strip()
    cached_level_str = str(cached_level).strip() if cached_level is not None else None

    if expected_cache_decision is None and expected_safe_reuse is not None:
        expected_cache_decision = "reuse" if expected_safe_reuse else "full"
    if expected_safe_reuse is None and expected_cache_decision is not None:
        expected_safe_reuse = expected_cache_decision in {"reuse", "patch"}
    if expected_level_isolation is None and cached_level_str is not None:
        expected_level_isolation = cached_level_str == request_level_str

    has_labels = any(
        value is not None
        for value in (expected_cache_decision, expected_safe_reuse, expected_level_isolation, cached_level_str)
    )

    return {
        "expected_cache_decision": expected_cache_decision,
        "expected_safe_reuse": expected_safe_reuse,
        "expected_level_isolation": expected_level_isolation,
        "request_level": request_level_str,
        "cached_level": cached_level_str,
        "drift_label_source": "metadata" if has_labels else "none",
    }


def _resolve_sample_level(sample: Any, default_level: str) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    level_value = _first_present(metadata, _REQUEST_LEVEL_KEYS)
    if level_value is None:
        return str(default_level).strip()
    level_text = str(level_value).strip()
    return level_text or str(default_level).strip()


def _resolve_mode_configs(modes_arg: str | None, comparison_profile: str) -> list[ModeConfig]:
    mode_names = COMPARISON_PROFILES[comparison_profile]
    if modes_arg:
        mode_names = [item.strip() for item in modes_arg.split(",") if item.strip()]

    resolved: list[ModeConfig] = []
    unknown = [name for name in mode_names if name not in MODE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown mode(s): {', '.join(unknown)}")

    for name in mode_names:
        resolved.append(MODE_CONFIGS[name])
    return resolved


def _resolve_dataset_path(dataset: Path | None, dataset_preset: str | None) -> Path:
    if dataset_preset:
        preset = DATASET_PRESETS.get(dataset_preset)
        if preset is None:
            raise SystemExit(f"Unknown dataset preset: {dataset_preset}")
        return preset
    return dataset or DATASET_PRESETS["hotpotqa"]


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    counts: dict[str, int] = {}
    for token in pred_tokens:
        counts[token] = counts.get(token, 0) + 1

    overlap = 0
    for token in ref_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_metric(prediction: str, references: list[str], metric) -> float:
    if not references:
        return 0.0
    return max(metric(prediction, reference) for reference in references)


def _bleu1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    precision = overlap / len(pred_tokens)
    brevity_penalty = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
    return brevity_penalty * precision


def _extract_references(expected: Any) -> list[str]:
    if isinstance(expected, dict):
        if isinstance(expected.get("answer"), str):
            return [expected["answer"]]
        if isinstance(expected.get("answers"), list):
            return [str(item).strip() for item in expected["answers"] if str(item).strip()]
    if isinstance(expected, list):
        return [str(item).strip() for item in expected if str(item).strip()]
    if isinstance(expected, str) and expected.strip():
        return [expected.strip()]
    return []


def _extract_gold_retrieval_ids(sample: Any) -> set[str]:
    metadata = sample.metadata or {}
    if sample.task == "multihop_qa":
        return {str(item).strip().lower() for item in (metadata.get("supporting_titles") or []) if str(item).strip()}
    if sample.task == "retrieval_qa":
        return {str(item).strip() for item in (metadata.get("relevant_passage_ids") or []) if str(item).strip()}
    return set()


def _recall_at_k(trace: list[dict[str, Any]], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    predicted_ids = {
        str(item.get("item_id") or "").strip().lower()
        for item in trace[:k]
        if str(item.get("item_id") or "").strip()
    }
    if not predicted_ids:
        return 0.0
    return len(predicted_ids & {item.lower() for item in gold_ids}) / max(len(gold_ids), 1)


def _mrr_at_k(trace: list[dict[str, Any]], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    gold_norm = {item.lower() for item in gold_ids}
    for rank, item in enumerate(trace[:k], start=1):
        item_id = str(item.get("item_id") or "").strip().lower()
        if item_id in gold_norm:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Multi-key LLM pool — round-robin rotation with per-key 429 cooldown
# ---------------------------------------------------------------------------

class _KeyPool:
    """Round-robin key pool with per-key cooldown + daily-limit tracking."""

    def __init__(
        self,
        keys: list[str],
        rpd_per_key: int = 14400,
        state_file: Path | None = None,
        namespace: str = "default",
    ) -> None:
        self._keys = [k.strip() for k in keys if k.strip()]
        self._rpd = rpd_per_key
        self._day_counts: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._index: int = 0
        self._state_file = state_file
        self._namespace = namespace
        self._load_state()

    def __len__(self) -> int:
        return len(self._keys)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def pick(self, exclude: set[str] | None = None) -> str | None:
        """Return the next non-cooling, non-exhausted key, or None."""
        now = time.monotonic()
        excluded = exclude or set()
        for _ in range(len(self._keys)):
            key = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % max(len(self._keys), 1)
            if key in excluded:
                continue
            if self._cooldown_until.get(key, 0.0) <= now:
                if self._day_counts.get(key, 0) < self._rpd:
                    return key
        return None

    def on_success(self, key: str) -> None:
        self._day_counts[key] = self._day_counts.get(key, 0) + 1
        self._persist_state()

    def on_fail(self, key: str, cooldown_seconds: float = 65.0) -> None:
        # 65 s = 60 s TPM window + 5 s margin.  If Groq returns a longer
        # Retry-After (TPD exhaustion), the caller should pass that value in.
        self._cooldown_until[key] = time.monotonic() + cooldown_seconds
        cooling = sum(
            1 for k in self._keys
            if self._cooldown_until.get(k, 0.0) > time.monotonic()
        )
        _LOG.info(
            "key_pool  Key ...%s cooling %.0fs (%d/%d keys cooling)",
            key[-6:], cooldown_seconds, cooling, len(self._keys),
        )
        self._persist_state()

    def soonest_available_in(self) -> float:
        now = time.monotonic()
        times = [
            max(0.0, self._cooldown_until.get(k, now) - now)
            for k in self._keys
            if self._day_counts.get(k, 0) < self._rpd
        ]
        return min(times) if times else float("inf")

    def _load_state(self) -> None:
        if not self._state_file or not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            provider_state = ((raw.get("providers") or {}).get(self._namespace) or {})
            if provider_state.get("utc_day") == _current_utc_day():
                self._day_counts = {
                    key: int(value)
                    for key, value in (provider_state.get("day_counts") or {}).items()
                    if key in self._keys
                }
            now_wall = time.time()
            now_mono = time.monotonic()
            for key, wall_time in (provider_state.get("cooldown_until") or {}).items():
                if key not in self._keys:
                    continue
                try:
                    remaining = float(wall_time) - now_wall
                except (TypeError, ValueError):
                    continue
                if remaining > 0:
                    self._cooldown_until[key] = now_mono + remaining
            try:
                self._index = int(provider_state.get("index", 0)) % max(len(self._keys), 1)
            except (TypeError, ValueError):
                self._index = 0
        except Exception as exc:
            _LOG.warning("key_pool  Could not load persisted quota state (%s)", exc)

    def _persist_state(self) -> None:
        if not self._state_file:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {}
            if self._state_file.exists():
                try:
                    payload = json.loads(self._state_file.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
            providers = payload.setdefault("providers", {})
            now_wall = time.time()
            now_mono = time.monotonic()
            providers[self._namespace] = {
                "utc_day": _current_utc_day(),
                "index": self._index,
                "day_counts": {
                    key: self._day_counts.get(key, 0)
                    for key in self._keys
                    if self._day_counts.get(key, 0) > 0
                },
                "cooldown_until": {
                    key: now_wall + max(0.0, until - now_mono)
                    for key, until in self._cooldown_until.items()
                    if key in self._keys and until > now_mono
                },
            }
            self._state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            _LOG.warning("key_pool  Could not persist quota state (%s)", exc)


_GROQ_POOL: _KeyPool | None = None
_GEMINI_POOL: _KeyPool | None = None


def _detected_output_provider(output: dict[str, Any]) -> str | None:
    models_used = [str(item).lower() for item in ((output.get("metadata") or {}).get("models_used") or [])]
    for model in models_used:
        if model.startswith("groq/") or "groq" in model:
            return "groq"
        if model.startswith("gemini") or "gemini" in model:
            return "gemini"
        if model.startswith("ollama"):
            return "ollama"
    return None


def _is_llm_failure(output: dict[str, Any]) -> bool:
    prediction = str(output.get("tutor_response") or "").strip()
    models_used = [str(item) for item in ((output.get("metadata") or {}).get("models_used") or [])]
    return not prediction or (
        len(models_used) > 0 and all("template" in item.lower() for item in models_used)
    )


def _results_for_primary_provider(results: list[QaRunResult], primary_provider: str = "groq") -> list[QaRunResult]:
    primary_results = [result for result in results if result.llm_provider == primary_provider]
    return primary_results if primary_results else results


def _clear_provider_throttle(provider: str) -> None:
    """Clear a provider's backoff timestamp so a rotated key can fire immediately."""
    mod = sys.modules.get("api.services.graph_cag.nodes_v2")
    if mod is not None and hasattr(mod, "_PROVIDER_NEXT_REQUEST_AT"):
        mod._PROVIDER_NEXT_REQUEST_AT.pop(provider, None)  # type: ignore[attr-defined]


def _clear_graphcag_inprocess_caches() -> None:
    """
    Reset GraphCAG in-process caches between modes.

    The ai-service implementation uses module-level in-memory caches as a Redis
    fallback. The GraphCAG pipeline is a singleton, so those caches can leak
    across benchmark modes unless we clear them explicitly.
    """

    mod = sys.modules.get("api.services.graph_cag.nodes_v2")
    if mod is None:
        try:
            import importlib

            mod = importlib.import_module("api.services.graph_cag.nodes_v2")
        except Exception:
            return

    for attr in ("_MEM_RESPONSE_CACHE", "_MEM_GRAPH_BUCKETS", "_MEM_BUCKET_VERSIONS"):
        store = getattr(mod, attr, None)
        if isinstance(store, dict):
            store.clear()


async def _analyze_with_key_rotation(
    pipeline: Any,
    sample: Any,
    *,
    session_id: str,
    level: str,
    cache_policy: str,
    retrieval_policy: str,
    generation_policy: str,
    benchmark_ranker: str,
) -> dict[str, Any]:
    """Run pipeline.analyze() with automatic Groq key rotation on LLM failure."""

    async def _call_pipeline() -> dict[str, Any]:
        benchmark_metadata = dict(sample.metadata or {})
        benchmark_metadata["_benchmark_ranker"] = benchmark_ranker
        return await pipeline.analyze(  # type: ignore[return-value]
            sample.text,
            session_id=session_id,
            learner_profile={"level": level},
            cache_policy=cache_policy,
            retrieval_policy=retrieval_policy,
            diagnosis_policy="rules",
            generation_policy=generation_policy,
            benchmark_task=sample.task,
            benchmark_context=str(sample.metadata.get("context") or ""),
            benchmark_metadata=benchmark_metadata,
        )

    async def _retry_provider_key(
        *,
        provider: str,
        key: str,
        pool: _KeyPool,
    ) -> tuple[dict[str, Any], bool]:
        saved_provider = os.environ.get("GRAPHCAG_BENCHMARK_LLM_PROVIDER", "groq")
        if provider == "groq":
            os.environ["GROQ_API_KEY"] = key
        elif provider == "gemini":
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GRAPHCAG_BENCHMARK_LLM_PROVIDER"] = "gemini"

        _clear_provider_throttle(provider)
        _LOG.info("key_pool  %s rotated -> ...%s, retrying %s", provider.capitalize(), key[-6:], session_id)
        try:
            candidate = await _call_pipeline()
        finally:
            if provider == "gemini":
                os.environ["GRAPHCAG_BENCHMARK_LLM_PROVIDER"] = saved_provider

        if not _is_llm_failure(candidate):
            if _detected_output_provider(candidate) == provider:
                pool.on_success(key)
            candidate.setdefault("_benchmark_provider", provider)
            return candidate, True

        pool.on_fail(key)
        return candidate, False

    # --- Pick initial Groq key (round-robin) ---
    active_key: str | None = None
    if _GROQ_POOL:
        active_key = _GROQ_POOL.pick()
        if active_key:
            os.environ["GROQ_API_KEY"] = active_key

    used_provider = "groq"  # track which provider actually answered

    output: dict[str, Any] = await _call_pipeline()
    llm_failed = _is_llm_failure(output)

    if not llm_failed:
        if active_key and _GROQ_POOL and _detected_output_provider(output) == "groq":
            _GROQ_POOL.on_success(active_key)
        output.setdefault("_benchmark_provider", used_provider)
        return output

    # --- Mark current key and try all immediately-available Groq keys ---
    tried_groq_keys: set[str] = set()
    last_output = output
    if active_key and _GROQ_POOL:
        _GROQ_POOL.on_fail(active_key)
        tried_groq_keys.add(active_key)

    while _GROQ_POOL:
        next_groq_key = _GROQ_POOL.pick(exclude=tried_groq_keys)
        if not next_groq_key:
            break
        tried_groq_keys.add(next_groq_key)
        output, groq_ok = await _retry_provider_key(provider="groq", key=next_groq_key, pool=_GROQ_POOL)
        last_output = output
        if groq_ok:
            output.setdefault("_benchmark_provider", "groq")
            return output

    # --- All Groq keys cooling: wait a bit before touching Gemini ---
    if _GROQ_POOL:
        groq_wait = _GROQ_POOL.soonest_available_in()
        max_groq_wait = _safe_float_env(
            "GRAPHCAG_MAX_GROQ_WAIT_SECONDS",
            _DEFAULT_GROQ_WAIT_BEFORE_GEMINI_SECONDS,
        )
        if 0.0 < groq_wait <= max_groq_wait:
            _LOG.warning(
                "key_pool  All Groq cooling; waiting %.1fs before Gemini fallback  session=%s",
                groq_wait,
                session_id,
            )
            await asyncio.sleep(groq_wait)
            retried_after_wait: set[str] = set()
            while _GROQ_POOL:
                next_groq_key = _GROQ_POOL.pick(exclude=retried_after_wait)
                if not next_groq_key:
                    break
                retried_after_wait.add(next_groq_key)
                output, groq_ok = await _retry_provider_key(provider="groq", key=next_groq_key, pool=_GROQ_POOL)
                last_output = output
                if groq_ok:
                    output.setdefault("_benchmark_provider", "groq")
                    return output

    # --- All Groq keys exhausted: fall back to Gemini ---
    if _GEMINI_POOL:
        tried_gemini_keys: set[str] = set()
        while _GEMINI_POOL:
            gemini_key = _GEMINI_POOL.pick(exclude=tried_gemini_keys)
            if not gemini_key:
                break
            tried_gemini_keys.add(gemini_key)
            _LOG.warning("key_pool  Groq unavailable -> trying Gemini ...%s  session=%s", gemini_key[-6:], session_id)
            output, gemini_ok = await _retry_provider_key(provider="gemini", key=gemini_key, pool=_GEMINI_POOL)
            last_output = output
            if gemini_ok:
                used_provider = "gemini"
                output.setdefault("_benchmark_provider", used_provider)
                return output

    last_output.setdefault("_benchmark_provider", used_provider)
    return last_output


async def _run_mode(
    *,
    dataset_name: str,
    mode_name: str,
    samples: list[Any],
    level: str,
    cache_policy: str,
    retrieval_policy: str,
    generation_policy: str,
    benchmark_ranker: str,
    use_gemini_fallback: bool,
    checkpoint_path: Path | None = None,
) -> list[QaRunResult]:
    pipeline = await _build_pipeline(use_gemini_fallback=use_gemini_fallback)
    results: list[QaRunResult] = []
    start_index = 0

    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            raw_items = ckpt_data.get("results", [])
            results = [
                QaRunResult(
                    **{
                        **item,
                        "reuse_risk": item.get("reuse_risk", 1.0),
                        "llm_provider": item.get("llm_provider", "groq"),
                        "prompt_tokens": item.get("prompt_tokens", 0),
                        "completion_tokens": item.get("completion_tokens", 0),
                        "total_tokens": item.get("total_tokens", 0),
                        "cached_tokens": item.get("cached_tokens", 0),
                        "effective_prompt_tokens": item.get("effective_prompt_tokens", 0.0),
                        "prompt_cache_token_rate": item.get("prompt_cache_token_rate", 0.0),
                        "prompt_discount_savings_rate": item.get("prompt_discount_savings_rate", 0.0),
                        "usage_source": item.get("usage_source", "none"),
                    }
                )
                for item in raw_items
            ]
            start_index = len(results)
            _LOG.info("checkpoint  Resuming %s from sample %d/%d", mode_name, start_index, len(samples))
            if start_index:
                print(f"  resuming from checkpoint: {start_index}/{len(samples)} done", flush=True)
                _render_progress(start_index, len(samples), results)
        except Exception as exc:
            _LOG.warning("checkpoint  Could not load checkpoint (%s), starting fresh", exc)
            results = []
            start_index = 0

    if start_index == 0:
        _clear_graphcag_inprocess_caches()

    for index, sample in enumerate(samples):
        if index < start_index:
            continue
        sample_level = _resolve_sample_level(sample, level)
        output = await _analyze_with_key_rotation(
            pipeline,
            sample,
            session_id=f"paper_{mode_name}_{index}",
            level=sample_level,
            cache_policy=cache_policy,
            retrieval_policy=retrieval_policy,
            generation_policy=generation_policy,
            benchmark_ranker=benchmark_ranker,
        )
        prediction = str(output.get("tutor_response") or "").strip()
        references = _extract_references(sample.expected)
        metadata = output.get("metadata") or {}
        actual_provider = output.get("_benchmark_provider", "groq")
        sample_metadata = dict(sample.metadata or {})
        merged_metadata = {**sample_metadata, **metadata}
        retrieval_trace = metadata.get("retrieval_trace") or []
        gold_retrieval_ids = _extract_gold_retrieval_ids(sample)
        drift_labels = _extract_drift_labels(merged_metadata, sample_level)
        token_usage = _extract_token_usage(output, metadata, sample.text, prediction)

        results.append(
            QaRunResult(
                dataset=dataset_name,
                mode=mode_name,
                task=sample.task or "unknown",
                query=sample.text,
                prediction=prediction,
                references=references,
                latency_ms=int(metadata.get("latency_ms") or 0),
                cache_hit=bool(metadata.get("cache_hit") or False),
                cache_decision=str(metadata.get("cache_decision") or "full"),
                cache_layer=str(metadata.get("cache_layer") or "none"),
                reuse_risk=float(metadata.get("reuse_risk") or 1.0),
                recall_at_1=_recall_at_k(retrieval_trace, gold_retrieval_ids, 1),
                recall_at_3=_recall_at_k(retrieval_trace, gold_retrieval_ids, 3),
                recall_at_5=_recall_at_k(retrieval_trace, gold_retrieval_ids, 5),
                mrr_at_5=_mrr_at_k(retrieval_trace, gold_retrieval_ids, 5),
                exact_match=_best_metric(prediction, references, _exact_match),
                token_f1=_best_metric(prediction, references, _token_f1),
                rouge_l_f1=_best_metric(prediction, references, _rouge_l_f1),
                bleu1=_best_metric(prediction, references, _bleu1),
                models_used=[str(item) for item in (metadata.get("models_used") or [])],
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                total_tokens=token_usage["total_tokens"],
                cached_tokens=token_usage["cached_tokens"],
                effective_prompt_tokens=token_usage["effective_prompt_tokens"],
                prompt_cache_token_rate=token_usage["prompt_cache_token_rate"],
                prompt_discount_savings_rate=token_usage["prompt_discount_savings_rate"],
                usage_source=token_usage["usage_source"],
                expected_cache_decision=drift_labels["expected_cache_decision"],
                expected_safe_reuse=drift_labels["expected_safe_reuse"],
                expected_level_isolation=drift_labels["expected_level_isolation"],
                request_level=drift_labels["request_level"],
                cached_level=drift_labels["cached_level"],
                drift_label_source=drift_labels["drift_label_source"],
                llm_provider=actual_provider,
            )
        )
        _render_progress(len(results), len(samples), results)
        _LOG.debug(
            "sample[%d]  mode=%s  EM=%.2f  F1=%.2f  cache=%s  lat=%dms",
            index, mode_name,
            results[-1].exact_match, results[-1].token_f1,
            results[-1].cache_decision, results[-1].latency_ms,
        )
        if checkpoint_path:
            try:
                checkpoint_path.write_text(
                    json.dumps({"results": [asdict(r) for r in results]}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass  # don't fail benchmark on checkpoint write error

    print()  # end the progress line
    return results


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = min(len(values_sorted) - 1, max(0, int(round((len(values_sorted) - 1) * (p / 100.0)))))
    return float(values_sorted[index])


# ---------------------------------------------------------------------------
# State Drift: PCC Metrics (GraphCAG-specific, paper §4.1)
# ---------------------------------------------------------------------------
# These metrics are only meaningful when cache_policy="on".
# Vanilla CAG reports them as a cache baseline without PCC gating.
# HippoRAG-style retrieval has no state-aware cache and reports N/A.
# ---------------------------------------------------------------------------

_PCC_QUALITY_THRESHOLD = 0.30  # F1 floor for "quality preserved"


def _compute_state_drift_metrics(results: list[QaRunResult], cache_policy: str) -> dict[str, Any]:
    """
    Compute GraphCAG State Drift metrics from a mode's run results.

    Metrics:
            PCC Precision:  P(label says safe reuse | decision ∈ {reuse, patch})
            PCC Recall:     P(decision ∈ {reuse, patch} | label says safe reuse)
            Level Isolation Rate: fraction of reuse/patch decisions that satisfy
                                                        labeled same-level constraints when available
            Drift Detection Accuracy: fraction of decisions matching labeled drift
                                                                expectations, with quality-based fallback
      Incorrect Reuse Rate ★: P(quality degraded | decision ∈ {reuse, patch})
    """
    n = len(results)
    if n == 0 or cache_policy != "on":
        return {
            "available": cache_policy == "on",
            "n": n,
            "pcc_precision": None,
            "pcc_recall": None,
            "level_isolation_rate": None,
            "drift_detection_accuracy": None,
            "incorrect_reuse_rate": None,
            "reuse_count": 0,
            "patch_count": 0,
            "full_count": 0,
            "cache_hit_count": 0,
        }

    reuse_or_patch = [r for r in results if r.cache_decision in {"reuse", "patch"}]
    full_results = [r for r in results if r.cache_decision == "full"]
    cache_hits = [r for r in results if r.cache_hit]
    labeled_results = [r for r in results if r.drift_label_source != "none"]
    labeled_reuse_or_patch = [r for r in reuse_or_patch if r.drift_label_source != "none"]
    labeled_safe = [r for r in labeled_results if r.expected_safe_reuse is True]
    labeled_level_checks = [r for r in labeled_reuse_or_patch if r.expected_level_isolation is not None]

    reuse_count = sum(1 for r in results if r.cache_decision == "reuse")
    patch_count = sum(1 for r in results if r.cache_decision == "patch")
    full_count = len(full_results)

    # PCC Precision: among reuse/patch decisions, safe reuse rate.
    # Prefer labeled drift probes when present; otherwise fall back to answer quality.
    if labeled_reuse_or_patch:
        safe_reuse_count = sum(1 for r in labeled_reuse_or_patch if r.expected_safe_reuse is True)
        pcc_precision = safe_reuse_count / len(labeled_reuse_or_patch)
    elif reuse_or_patch:
        quality_preserved = sum(1 for r in reuse_or_patch if r.token_f1 >= _PCC_QUALITY_THRESHOLD)
        pcc_precision = quality_preserved / len(reuse_or_patch)
    else:
        pcc_precision = None

    # PCC Recall: among labeled safe-reuse probes, how often the system reused.
    # Fall back to cache-hit coverage if labels are absent.
    if labeled_safe:
        pcc_recall = sum(1 for r in labeled_safe if r.cache_decision in {"reuse", "patch"}) / len(labeled_safe)
    elif cache_hits:
        pcc_recall = len(reuse_or_patch) / len(cache_hits)
    else:
        pcc_recall = None

    # Level Isolation Rate: use labeled same-level constraints when available.
    if labeled_level_checks:
        level_isolation_rate = sum(1 for r in labeled_level_checks if r.expected_level_isolation is True) / len(labeled_level_checks)
    else:
        inferred_level_checks = [r for r in reuse_or_patch if r.cached_level and r.request_level]
        if inferred_level_checks:
            level_isolation_rate = (
                sum(1 for r in inferred_level_checks if r.cached_level == r.request_level) / len(inferred_level_checks)
            )
        else:
            level_isolation_rate = 1.0 if reuse_or_patch else None

    # Drift Detection Accuracy: exact decision match when labeled, otherwise quality fallback.
    if labeled_results:
        correct = 0
        total = 0
        for r in labeled_results:
            expected_decision = r.expected_cache_decision
            if expected_decision is None and r.expected_safe_reuse is not None:
                expected_decision = "reuse" if r.expected_safe_reuse else "full"
            if expected_decision is None:
                continue
            total += 1
            if expected_decision == r.cache_decision:
                correct += 1
        drift_detection_accuracy = correct / total if total > 0 else None
    else:
        correct = 0
        total = len(results)
        for r in reuse_or_patch:
            if r.token_f1 >= _PCC_QUALITY_THRESHOLD:
                correct += 1
        correct += full_count  # conservative fallback when no drift labels exist
        drift_detection_accuracy = correct / total if total > 0 else None

    # ★ Incorrect Reuse Rate: labeled unsafe reuse when available, otherwise quality-degraded reuse.
    if labeled_reuse_or_patch:
        incorrect = sum(1 for r in labeled_reuse_or_patch if r.expected_safe_reuse is False)
        incorrect_reuse_rate = incorrect / len(labeled_reuse_or_patch)
    elif reuse_or_patch:
        incorrect = sum(1 for r in reuse_or_patch if r.token_f1 < _PCC_QUALITY_THRESHOLD)
        incorrect_reuse_rate = incorrect / len(reuse_or_patch)
    else:
        incorrect_reuse_rate = None

    return {
        "available": True,
        "n": n,
        "label_grounded_n": len(labeled_results),
        "label_coverage": (len(labeled_results) / n) if n > 0 else 0.0,
        "metric_basis": "labels" if labeled_results else "quality-fallback",
        "pcc_precision": pcc_precision,
        "pcc_recall": pcc_recall,
        "level_isolation_rate": level_isolation_rate,
        "drift_detection_accuracy": drift_detection_accuracy,
        "incorrect_reuse_rate": incorrect_reuse_rate,
        "reuse_count": reuse_count,
        "patch_count": patch_count,
        "full_count": full_count,
        "cache_hit_count": len(cache_hits),
    }


def _summarize(results: list[QaRunResult], primary_provider: str = "groq") -> dict[str, Any]:
    # Split results by provider
    primary_results = _results_for_primary_provider(results, primary_provider)
    fallback_results = [r for r in results if r.llm_provider != primary_provider]
    # Use only primary-provider samples for fair metrics
    scored = primary_results if primary_results else results
    latencies = [result.latency_ms for result in scored]
    prompt_tokens_total = sum(result.prompt_tokens for result in scored)
    completion_tokens_total = sum(result.completion_tokens for result in scored)
    total_tokens_total = sum(result.total_tokens for result in scored)
    cached_tokens_total = sum(result.cached_tokens for result in scored)
    effective_prompt_tokens_total = sum(result.effective_prompt_tokens for result in scored)
    usage_sources = sorted({result.usage_source for result in scored})
    return {
        "n": len(scored),
        "n_total": len(results),
        "n_fallback": len(fallback_results),
        "fallback_rate": len(fallback_results) / max(len(results), 1),
        "fallback_providers": list({r.llm_provider for r in fallback_results}),
        "recall_at_1": statistics.mean(result.recall_at_1 for result in scored) if scored else 0.0,
        "recall_at_3": statistics.mean(result.recall_at_3 for result in scored) if scored else 0.0,
        "recall_at_5": statistics.mean(result.recall_at_5 for result in scored) if scored else 0.0,
        "mrr_at_5": statistics.mean(result.mrr_at_5 for result in scored) if scored else 0.0,
        "exact_match": statistics.mean(result.exact_match for result in scored) if scored else 0.0,
        "token_f1": statistics.mean(result.token_f1 for result in scored) if scored else 0.0,
        "rouge_l_f1": statistics.mean(result.rouge_l_f1 for result in scored) if scored else 0.0,
        "bleu1": statistics.mean(result.bleu1 for result in scored) if scored else 0.0,
        "cache_hit_rate": statistics.mean(1.0 if result.cache_hit else 0.0 for result in scored) if scored else 0.0,
        "l0_rate": statistics.mean(1.0 if result.cache_layer == "L0" else 0.0 for result in scored) if scored else 0.0,
        "l1_rate": statistics.mean(1.0 if result.cache_layer == "L1" else 0.0 for result in scored) if scored else 0.0,
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "cached_tokens_total": cached_tokens_total,
        "effective_prompt_tokens_total": effective_prompt_tokens_total,
        "prompt_cache_token_rate": (cached_tokens_total / prompt_tokens_total) if prompt_tokens_total > 0 else 0.0,
        "prompt_discount_savings_rate": (
            (_CACHED_INPUT_DISCOUNT * cached_tokens_total) / prompt_tokens_total
            if prompt_tokens_total > 0
            else 0.0
        ),
        "usage_sources": usage_sources,
    }


def _build_application_reuse_view(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": summary.get("n", 0),
        "n_total": summary.get("n_total", 0),
        "n_fallback": summary.get("n_fallback", 0),
        "fallback_rate": summary.get("fallback_rate", 0.0),
        "fallback_providers": summary.get("fallback_providers", []),
        "retrieval_quality": {
            "recall_at_1": summary.get("recall_at_1", 0.0),
            "recall_at_3": summary.get("recall_at_3", 0.0),
            "recall_at_5": summary.get("recall_at_5", 0.0),
            "mrr_at_5": summary.get("mrr_at_5", 0.0),
            "exact_match": summary.get("exact_match", 0.0),
            "token_f1": summary.get("token_f1", 0.0),
            "rouge_l_f1": summary.get("rouge_l_f1", 0.0),
            "bleu1": summary.get("bleu1", 0.0),
        },
        "reuse_controller": {
            "cache_hit_rate": summary.get("cache_hit_rate", 0.0),
            "l0_rate": summary.get("l0_rate", 0.0),
            "l1_rate": summary.get("l1_rate", 0.0),
        },
        "latency": {
            "mean_ms": summary.get("latency_ms_mean", 0.0),
            "p50_ms": summary.get("latency_ms_p50", 0.0),
            "p95_ms": summary.get("latency_ms_p95", 0.0),
        },
    }


def _build_provider_prompt_caching_view(summary: dict[str, Any]) -> dict[str, Any]:
    usage_sources = summary.get("usage_sources", [])
    basis = "provider" if "provider" in usage_sources else "estimated"
    return {
        "basis": basis,
        "usage_sources": usage_sources,
        "prompt_tokens_total": summary.get("prompt_tokens_total", 0),
        "completion_tokens_total": summary.get("completion_tokens_total", 0),
        "total_tokens_total": summary.get("total_tokens_total", 0),
        "cached_tokens_total": summary.get("cached_tokens_total", 0),
        "prompt_cache_token_rate": summary.get("prompt_cache_token_rate", 0.0),
        "effective_prompt_tokens_total": summary.get("effective_prompt_tokens_total", 0.0),
        "prompt_discount_savings_rate": summary.get("prompt_discount_savings_rate", 0.0),
    }


def _print_summary(
    summaries: dict[str, dict[str, Any]],
    dataset_preset: str,
    mode_labels: dict[str, str] | None = None,
) -> None:
    print("=== Application-Level Reuse, Quality, and Latency ===")
    print("This table is the GraphCAG controller view: retrieval quality, reuse rates, and latency at the application layer.")
    print()
    # --- Fallback warning ---
    any_fallback = any(s.get("n_fallback", 0) > 0 for s in summaries.values())
    if any_fallback:
        print("⚠  Provider fallback detected — metrics use primary-provider samples only.")
        for mode, s in summaries.items():
            n_fb = s.get("n_fallback", 0)
            if n_fb > 0:
                label = (mode_labels or {}).get(mode, mode)
                fb_providers = ", ".join(s.get("fallback_providers", []))
                print(f"   {label}: {n_fb}/{s['n_total']} samples fell back to {fb_providers} (excluded from metrics)")
        print()

    metric_headers = ["R@1", "R@3", "R@5", "MRR@5", "EM", "F1", "ROUGE-L"]
    headers = ["Mode", "N", *metric_headers, "HitRate", "L0", "L1", "Mean(ms)", "P50(ms)", "P95(ms)"]
    rows = []
    for mode, summary in summaries.items():
        n_display = str(summary["n"])
        if summary.get("n_fallback", 0) > 0:
            n_display += f"/{summary['n_total']}"
        metric_values = [
            f"{summary['recall_at_1']*100:.1f}",
            f"{summary['recall_at_3']*100:.1f}",
            f"{summary['recall_at_5']*100:.1f}",
            f"{summary['mrr_at_5']*100:.1f}",
            f"{summary['exact_match']*100:.1f}",
            f"{summary['token_f1']*100:.1f}",
            f"{summary['rouge_l_f1']*100:.1f}",
        ]
        rows.append([
            (mode_labels or {}).get(mode, mode),
            n_display,
            *metric_values,
            f"{summary['cache_hit_rate']*100:.1f}",
            f"{summary['l0_rate']*100:.1f}",
            f"{summary['l1_rate']*100:.1f}",
            f"{summary['latency_ms_mean']:.1f}",
            f"{summary['latency_ms_p50']:.1f}",
            f"{summary['latency_ms_p95']:.1f}",
        ])

    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def _print_state_drift_summary(
    drift_metrics: dict[str, dict[str, Any]],
    mode_labels: dict[str, str] | None = None,
) -> None:
    """Print the GraphCAG State Drift: PCC Metrics table."""
    print()
    print("=== State Drift: PCC Metrics (GraphCAG-specific) ===")
    print("★ Incorrect Reuse Rate is the key failure metric for cache-based methods under drift; HippoRAG proxy reports N/A.")
    print("Metrics use labeled drift metadata when available and fall back to answer-quality heuristics otherwise.")
    print()

    def _pct(v: float | None) -> str:
        return "N/A" if v is None else f"{v * 100:.1f}"

    headers = ["Mode", "Reuse", "Patch", "Full", "LblN", "Basis", "PCC-P%", "PCC-R%", "LvlIso%", "DriftAcc%", "★ IncReuse%"]
    rows = []
    for mode, m in drift_metrics.items():
        label = (mode_labels or {}).get(mode, mode)
        if not m.get("available"):
            rows.append([label, "-", "-", "-", "-", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            rows.append([
                label,
                str(m["reuse_count"]),
                str(m["patch_count"]),
                str(m["full_count"]),
                str(m.get("label_grounded_n") or 0),
                str(m.get("metric_basis") or "unknown"),
                _pct(m["pcc_precision"]),
                _pct(m["pcc_recall"]),
                _pct(m["level_isolation_rate"]),
                _pct(m["drift_detection_accuracy"]),
                _pct(m["incorrect_reuse_rate"]),
            ])

    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def _print_token_usage_summary(
    summaries: dict[str, dict[str, Any]],
    mode_labels: dict[str, str] | None = None,
) -> None:
    print()
    print("=== Provider-Side Prompt Caching and Token Accounting ===")
    print("This table is the serving-layer view: prompt/completion token usage and cached prompt-token savings independent of GraphCAG reuse correctness.")
    print("When provider usage fields are unavailable, prompt/completion tokens are estimated and cached tokens remain zero.")
    print()

    headers = ["Mode", "Usage", "PromptTok", "CachedTok", "Cached%", "ComplTok", "EffPromptTok", "Save%"]
    rows = []
    for mode, summary in summaries.items():
        rows.append([
            (mode_labels or {}).get(mode, mode),
            ",".join(summary.get("usage_sources", [])) or "none",
            str(int(summary.get("prompt_tokens_total", 0))),
            str(int(summary.get("cached_tokens_total", 0))),
            f"{summary.get('prompt_cache_token_rate', 0.0) * 100:.1f}",
            str(int(summary.get("completion_tokens_total", 0))),
            f"{summary.get('effective_prompt_tokens_total', 0.0):.1f}",
            f"{summary.get('prompt_discount_savings_rate', 0.0) * 100:.1f}",
        ])

    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style public QA benchmark for GraphCAG.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-preset", type=str, default="hotpotqa", choices=["graphcag_drift_probes", "hotpotqa", "2wikimultihopqa"])
    parser.add_argument(
        "--comparison-profile",
        type=str,
        default="public_cag_compare",
        choices=sorted(COMPARISON_PROFILES.keys()),
        help="Named benchmark setup for comparing vanilla CAG, GraphCAG, and the HippoRAG proxy.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Optional comma-separated mode override. Available: cag_vanilla, cag_flat, graphcag_full, hipporag_proxy, graphcag_rapid.",
    )
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--level", type=str, default="B1")
    parser.add_argument("--generation-policy", type=str, default="auto", choices=["template", "auto"])
    parser.add_argument("--llm-provider", type=str, default="groq", choices=["auto", "groq", "gemini", "ollama", "template"])
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--groq-rpm", type=int, default=28)
    parser.add_argument("--groq-key", type=str, default=None)
    parser.add_argument(
        "--groq-keys",
        type=str,
        default=None,
        help="Comma-separated Groq API keys for round-robin rotation on 429",
    )
    parser.add_argument(
        "--gemini-keys",
        type=str,
        default=None,
        help="Comma-separated Gemini API keys (fallback when all Groq keys cooling)",
    )
    parser.add_argument("--enable-gemini-fallback", action="store_true")
    parser.add_argument("--enable-ollama-fallback", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--quota-state-file",
        type=Path,
        default=None,
        help="Path to persisted key-pool quota state shared across dataset runs.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path for the debug log file. Defaults to <report-json path>.log when --report-json is set.",
    )
    args = parser.parse_args()

    # --- Logging setup (before anything else so all events are captured) ---
    log_path: Path | None = args.log_file
    if log_path is None and args.report_json is not None:
        log_path = args.report_json.with_suffix(".log")
    _setup_logging(log_path)
    if log_path:
        print(f"  log → {log_path}", flush=True)

    quota_state_file: Path | None = args.quota_state_file
    if quota_state_file is None and args.report_json is not None:
        quota_state_file = args.report_json.parent.parent / "provider_quota_state.json"

    # --- Base env setup ---
    os.environ["GROQ_MODEL"] = args.groq_model
    os.environ["GRAPHCAG_BENCHMARK_LLM_PROVIDER"] = args.llm_provider
    os.environ["GRAPHCAG_ENABLE_GEMINI_FALLBACK"] = "1" if args.enable_gemini_fallback else "0"
    os.environ["GRAPHCAG_ENABLE_OLLAMA_FALLBACK"] = "1" if args.enable_ollama_fallback else "0"
    # Fast-fail on 429: let _KeyPool rotation handle retries at the pool level.
    # Without this, nodes_v2 would retry 3× internally before returning, wasting
    # time that could be spent trying the next key.
    os.environ["GRAPHCAG_LLM_MAX_RETRIES"] = "1"

    # --- Groq key pool ---
    global _GROQ_POOL, _GEMINI_POOL
    groq_keys_raw: list[str] = []
    if args.groq_keys:
        groq_keys_raw = [k.strip() for k in args.groq_keys.split(",") if k.strip()]
    elif args.groq_key:
        groq_keys_raw = [args.groq_key.strip()]
    elif os.getenv("GROQ_API_KEY"):
        groq_keys_raw = [os.environ["GROQ_API_KEY"]]

    if groq_keys_raw:
        _GROQ_POOL = _KeyPool(
            groq_keys_raw,
            rpd_per_key=14400,
            state_file=quota_state_file,
            namespace="groq",
        )
        os.environ["GROQ_API_KEY"] = groq_keys_raw[0]
        # llama-3.1-8b-instant: TPM=6K → ~8 safe RPM/key; scale linearly with key count
        scaled_rpm = min(args.groq_rpm, 8) * len(groq_keys_raw)
        os.environ["GRAPHCAG_GROQ_RPM"] = str(scaled_rpm)
        msg = f"Groq pool: {len(groq_keys_raw)} key(s)  model={args.groq_model}  RPM={scaled_rpm}"
        print(f"  {msg}", flush=True)
        _LOG.info("key_pool  %s", msg)
    else:
        os.environ["GRAPHCAG_GROQ_RPM"] = str(max(1, args.groq_rpm))

    # --- Gemini key pool (fallback only — RPD=20/key is very limited) ---
    gemini_keys_raw: list[str] = []
    if args.gemini_keys:
        gemini_keys_raw = [k.strip() for k in args.gemini_keys.split(",") if k.strip()]
    elif os.getenv("GEMINI_API_KEY"):
        gemini_keys_raw = [os.environ["GEMINI_API_KEY"]]

    if gemini_keys_raw:
        _GEMINI_POOL = _KeyPool(
            gemini_keys_raw,
            rpd_per_key=20,
            state_file=quota_state_file,
            namespace="gemini",
        )
        os.environ["GEMINI_API_KEY"] = gemini_keys_raw[0]
        msg = f"Gemini pool: {len(gemini_keys_raw)} key(s) — fallback only (RPD=20/key)"
        print(f"  {msg}", flush=True)
        _LOG.info("key_pool  %s", msg)
    if quota_state_file is not None:
        print(f"  quota state → {quota_state_file}", flush=True)
        _LOG.info("key_pool  quota state file=%s", quota_state_file)

    dataset_path = _resolve_dataset_path(args.dataset, args.dataset_preset)
    samples = list(_iter_dataset_samples(dataset_path))[: args.n]
    if not samples:
        raise SystemExit(f"No samples found in dataset: {dataset_path}")
    mode_configs = _resolve_mode_configs(args.modes, args.comparison_profile)

    async def runner() -> None:
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
        ckpt_dir = args.report_json.parent if args.report_json else Path(".")
        mode_results: dict[str, list[QaRunResult]] = {}
        mode_labels = {mode.name: mode.label for mode in mode_configs}

        # ── run banner ────────────────────────────────────────────────────────
        sep = "─" * 70
        print()
        print(sep)
        print(f"  GraphCAG Benchmark")
        print(f"  dataset : {args.dataset_preset}  ({len(samples)} samples)")
        print(f"  profile : {args.comparison_profile}")
        print(f"  modes   : {', '.join(m.label for m in mode_configs)}")
        print(sep)
        _LOG.info(
            "run start  dataset=%s  n=%d  profile=%s  provider=%s  model=%s  rpm=%d  gemini_fallback=%s",
            args.dataset_preset, len(samples), args.comparison_profile,
            args.llm_provider, args.groq_model, args.groq_rpm, args.enable_gemini_fallback,
        )

        # Warm-up once to pay one-time costs (graph compile, KG seed) outside the
        # measured sample loop. Use template generation to avoid LLM spend.
        try:
            pipeline = await _build_pipeline(use_gemini_fallback=args.enable_gemini_fallback)
            warm_sample = samples[0]
            warm_level = _resolve_sample_level(warm_sample, args.level)
            warm_metadata = dict(warm_sample.metadata or {})
            warm_metadata["_benchmark_ranker"] = "flat"
            await pipeline.analyze(  # type: ignore[call-arg]
                warm_sample.text,
                session_id="paper_warmup",
                learner_profile={"level": warm_level},
                cache_policy="off",
                retrieval_policy="full",
                diagnosis_policy="rules",
                generation_policy="template",
                benchmark_task=warm_sample.task,
                benchmark_context=str(warm_sample.metadata.get("context") or ""),
                benchmark_metadata=warm_metadata,
            )
        except Exception as exc:
            _LOG.info("warmup  skipped (%s)", exc)

        for i, mode in enumerate(mode_configs):
            print()
            print(f"  ── [{i + 1}/{len(mode_configs)}]  {mode.label}")
            print(f"       cache={mode.cache_policy}  retrieval={mode.retrieval_policy}  ranker={mode.benchmark_ranker}")
            checkpoint_path = ckpt_dir / f"{args.dataset_preset}_{mode.name}_ckpt.json"
            t0 = time.monotonic()
            mode_results[mode.name] = await _run_mode(
                dataset_name=args.dataset_preset,
                mode_name=mode.name,
                samples=samples,
                level=args.level,
                cache_policy=mode.cache_policy,
                retrieval_policy=mode.retrieval_policy,
                generation_policy=args.generation_policy,
                benchmark_ranker=mode.benchmark_ranker,
                use_gemini_fallback=args.enable_gemini_fallback,
                checkpoint_path=checkpoint_path,
            )
            elapsed = time.monotonic() - t0
            n_done = len(mode_results[mode.name])
            _LOG.info("mode done  %s  n=%d  elapsed=%.1fs", mode.name, n_done, elapsed)

        summaries = {mode.name: _summarize(mode_results[mode.name]) for mode in mode_configs}
        application_views = {mode.name: _build_application_reuse_view(summaries[mode.name]) for mode in mode_configs}
        provider_views = {mode.name: _build_provider_prompt_caching_view(summaries[mode.name]) for mode in mode_configs}
        drift_metrics = {
            mode.name: _compute_state_drift_metrics(
                _results_for_primary_provider(mode_results[mode.name]),
                mode.cache_policy,
            )
            for mode in mode_configs
        }

        # ── results ───────────────────────────────────────────────────────────
        print()
        print("═" * 70)
        print(f"  RESULTS  {args.dataset_preset}  n={len(samples)}")
        print("═" * 70)
        print(f"  {DATASET_RATIONALE.get(args.dataset_preset, 'General public QA benchmark.')}")
        print()
        _print_summary(summaries, args.dataset_preset, mode_labels)
        _print_state_drift_summary(drift_metrics, mode_labels)
        _print_token_usage_summary(summaries, mode_labels)

        # Log the summary tables to the log file too
        _LOG.info("=== QA Summary — %s ===", args.dataset_preset)
        for mode_name, s in summaries.items():
            _LOG.info(
                "  %-24s  F1=%.3f  EM=%.3f  ROUGE-L=%.3f  cache_hit=%.3f  lat_mean=%.0fms  cached_tok=%d",
                mode_name, s["token_f1"], s["exact_match"], s["rouge_l_f1"],
                s["cache_hit_rate"], s["latency_ms_mean"], s["cached_tokens_total"],
            )
        _LOG.info("=== State Drift — %s ===", args.dataset_preset)
        for mode_name, dm in drift_metrics.items():
            if dm.get("available"):
                _LOG.info(
                    "  %-24s  pcc_p=%.3f  pcc_r=%.3f  incReuse=%.3f  basis=%s",
                    mode_name,
                    dm.get("pcc_precision") or 0.0,
                    dm.get("pcc_recall") or 0.0,
                    dm.get("incorrect_reuse_rate") or 0.0,
                    dm.get("metric_basis", "N/A"),
                )
            else:
                _LOG.info("  %-24s  N/A (cache_policy=off)", mode_name)

        if args.report_json:
            application_report_path = args.report_json.with_name(f"{args.report_json.stem}-application.json")
            provider_report_path = args.report_json.with_name(f"{args.report_json.stem}-provider.json")
            report = {
                "dataset": str(dataset_path),
                "dataset_preset": args.dataset_preset,
                "n": len(samples),
                "comparison_profile": args.comparison_profile,
                "dataset_rationale": DATASET_RATIONALE.get(args.dataset_preset, "General public QA benchmark."),
                "protocol": {
                    "context_mode": "oracle_context",
                    "candidate_pool": "same benchmark-provided documents for all modes",
                    "hipporag_proxy": "memory-propagation proxy over the candidate graph; not a full HippoRAG implementation",
                    "state_drift_note": (
                        "PCC metrics (Precision, Recall, Level Isolation, Drift Detection, "
                        "Incorrect Reuse Rate) are central to GraphCAG. Vanilla CAG can still "
                        "be scored as a cache baseline, while HippoRAG proxy has no equivalent "
                        "state-aware cache and reports N/A."
                    ),
                    "state_drift_label_schema": {
                        "expected_cache_decision": ["reuse", "patch", "full"],
                        "expected_safe_reuse": "boolean; whether PCC should allow reuse on this probe",
                        "expected_level_isolation": "boolean; whether source and request level should match",
                        "request_level": "requested learner/task level",
                        "cached_level": "level associated with the reused candidate, when known",
                    },
                    "state_drift_metric_policy": (
                        "When drift labels are present in dataset or pipeline metadata, PCC metrics are "
                        "computed from those labels. Otherwise the harness falls back to answer-quality "
                        "heuristics so the report remains usable as a lower-bound cache study."
                    ),
                },
                "state_drift_datasets": STATE_DRIFT_DATASETS,
                "generation_policy": args.generation_policy,
                "llm_provider": args.llm_provider,
                "groq_model": args.groq_model,
                "groq_rpm": args.groq_rpm,
                "enable_gemini_fallback": args.enable_gemini_fallback,
                "enable_ollama_fallback": args.enable_ollama_fallback,
                "quota_state_file": str(quota_state_file) if quota_state_file is not None else None,
                "modes": [
                    {
                        "name": mode.name,
                        "label": mode.label,
                        "cache_policy": mode.cache_policy,
                        "retrieval_policy": mode.retrieval_policy,
                        "benchmark_ranker": mode.benchmark_ranker,
                        "description": mode.description,
                    }
                    for mode in mode_configs
                ],
                "reported_metrics": ["R@1", "R@3", "R@5", "MRR@5", "EM", "F1", "ROUGE-L", "BLEU-1"],
                "reported_state_drift_metrics": [
                    "PCC Precision", "PCC Recall", "Level Isolation Rate",
                    "Drift Detection Accuracy", "Incorrect Reuse Rate",
                ],
                "reported_token_metrics": [
                    "Prompt Tokens",
                    "Completion Tokens",
                    "Total Tokens",
                    "Cached Prompt Tokens",
                    "Prompt Cache Token Rate",
                    "Effective Billed Prompt Tokens",
                    "Prompt Discount Savings Rate",
                ],
                "state_drift_metric_basis": {
                    "labels": "Preferred. Uses explicit drift annotations from benchmark metadata.",
                    "quality-fallback": "Fallback. Uses token F1 threshold when explicit drift labels are unavailable.",
                },
                "prompt_accounting_policy": {
                    "cached_input_discount": _CACHED_INPUT_DISCOUNT,
                    "provider": "Preferred. Reads prompt/completion/cached token fields when the serving backend exposes them.",
                    "estimated": "Fallback. Estimates prompt and completion tokens from text when provider usage is unavailable; cached tokens remain zero.",
                },
                "report_views": {
                    "application_level_reuse": application_views,
                    "provider_prompt_caching": provider_views,
                },
                "sidecar_reports": {
                    "application_level_reuse": str(application_report_path),
                    "provider_prompt_caching": str(provider_report_path),
                },
                "summaries": summaries,
                "state_drift": drift_metrics,
                "mode_results": {
                    mode.name: [asdict(item) for item in mode_results[mode.name]]
                    for mode in mode_configs
                },
            }
            args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            application_report = {
                "dataset": str(dataset_path),
                "dataset_preset": args.dataset_preset,
                "comparison_profile": args.comparison_profile,
                "view": "application_level_reuse",
                "description": "Application-layer benchmark view for GraphCAG, vanilla CAG, and HippoRAG-style baselines.",
                "reported_metrics": [
                    "R@1",
                    "R@3",
                    "R@5",
                    "MRR@5",
                    "EM",
                    "F1",
                    "ROUGE-L",
                    "BLEU-1",
                    "Cache Hit Rate",
                    "L0 Rate",
                    "L1 Rate",
                    "Latency Mean",
                    "Latency P50",
                    "Latency P95",
                ],
                "summaries": application_views,
                "state_drift": drift_metrics,
            }
            provider_report = {
                "dataset": str(dataset_path),
                "dataset_preset": args.dataset_preset,
                "comparison_profile": args.comparison_profile,
                "view": "provider_prompt_caching",
                "description": "Serving-layer token-accounting view; orthogonal to GraphCAG application-layer reuse correctness.",
                "prompt_accounting_policy": report["prompt_accounting_policy"],
                "reported_metrics": [
                    "Prompt Tokens",
                    "Completion Tokens",
                    "Total Tokens",
                    "Cached Prompt Tokens",
                    "Prompt Cache Token Rate",
                    "Effective Billed Prompt Tokens",
                    "Prompt Discount Savings Rate",
                ],
                "summaries": provider_views,
            }
            application_report_path.write_text(json.dumps(application_report, indent=2), encoding="utf-8")
            provider_report_path.write_text(json.dumps(provider_report, indent=2), encoding="utf-8")
            print()
            print(f"  \u2713  report \u2192 {args.report_json}")
            print(f"  \u2713  app    \u2192 {application_report_path}")
            print(f"  \u2713  token  \u2192 {provider_report_path}")
            if log_path:
                print(f"  \u2713  log    \u2192 {log_path}")
            _LOG.info("report saved: %s", args.report_json)
            # Clean up checkpoint files after successful report write
            for mode in mode_configs:
                ckpt = ckpt_dir / f"{args.dataset_preset}_{mode.name}_ckpt.json"
                if ckpt.exists():
                    ckpt.unlink(missing_ok=True)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
