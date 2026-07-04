from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
from pathlib import Path

from .catalog import DATASETS, MODES, PROFILES, get_dataset, get_modes
from .config import (
    BenchmarkConfig,
    env_bool,
    env_int,
    env_optional_int,
    env_str,
    load_env_file,
)
from .datasets.driftbench import load_driftbench
from .datasets.public_qa import load_public_qa
from .protocols.drift_safety import run_drift_safety_protocol
from .protocols.public_qa import run_public_qa_protocol
from .reporting.console import print_summary
from .reporting.json_report import build_report, write_report


def build_parser() -> argparse.ArgumentParser:
    load_env_file()
    parser = argparse.ArgumentParser(description="Production-backed TRACE-CAG benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    public = subparsers.add_parser("public-qa", help="Run public multi-hop QA benchmark")
    public.add_argument(
        "--dataset",
        choices=[name for name, spec in DATASETS.items() if spec.protocol == "public_qa"],
        default=env_str("TRACECAG_BENCHMARK_DATASET", "hotpotqa"),
    )
    public.add_argument("--n", type=int, default=env_int("TRACECAG_BENCHMARK_N", 20))
    public.add_argument("--seed", type=int, default=env_int("TRACECAG_BENCHMARK_SEED", 42))
    public.add_argument(
        "--profile",
        choices=list(PROFILES),
        default=env_str("TRACECAG_BENCHMARK_PROFILE", "public_cag_compare"),
    )
    public.add_argument("--modes", nargs="+", choices=list(MODES))
    public.add_argument(
        "--cache-repeats",
        type=int,
        default=env_int("TRACECAG_BENCHMARK_CACHE_REPEATS", 2),
    )
    public.add_argument(
        "--generation-policy",
        choices=["auto", "llm", "extractive"],
        default=env_str("TRACECAG_BENCHMARK_GENERATION_POLICY", "auto"),
    )
    public.add_argument(
        "--evidence-mode",
        choices=["candidate_pool", "kg_only"],
        default=env_str("TRACECAG_BENCHMARK_EVIDENCE_MODE", "candidate_pool"),
    )
    _common_args(public)

    drift = subparsers.add_parser("drift-safety", help="Run production PCC/SCAR drift benchmark")
    drift.add_argument(
        "--split",
        choices=["train", "calibration", "test"],
        default=env_str("TRACECAG_BENCHMARK_DRIFT_SPLIT", "test"),
    )
    drift.add_argument(
        "--n-clusters",
        type=int,
        default=env_optional_int("TRACECAG_BENCHMARK_N_CLUSTERS"),
    )
    drift.add_argument(
        "--mode",
        choices=list(MODES),
        default=env_str("TRACECAG_BENCHMARK_DRIFT_MODE", "tracecag_rapid"),
    )
    drift.add_argument(
        "--generation-policy",
        choices=["auto", "llm", "extractive"],
        default=env_str("TRACECAG_BENCHMARK_GENERATION_POLICY", "auto"),
    )
    _common_args(drift)

    all_cmd = subparsers.add_parser("all", help="Run public QA datasets followed by drift safety")
    all_cmd.add_argument("--n", type=int, default=env_int("TRACECAG_BENCHMARK_N", 20))
    all_cmd.add_argument(
        "--n-clusters",
        type=int,
        default=env_optional_int("TRACECAG_BENCHMARK_N_CLUSTERS"),
    )
    all_cmd.add_argument("--seed", type=int, default=env_int("TRACECAG_BENCHMARK_SEED", 42))
    all_cmd.add_argument(
        "--generation-policy",
        choices=["auto", "llm", "extractive"],
        default=env_str("TRACECAG_BENCHMARK_GENERATION_POLICY", "auto"),
    )
    all_cmd.add_argument(
        "--profile",
        choices=list(PROFILES),
        default=env_str("TRACECAG_BENCHMARK_PROFILE", "public_cag_compare"),
    )
    all_cmd.add_argument(
        "--cache-repeats",
        type=int,
        default=env_int("TRACECAG_BENCHMARK_CACHE_REPEATS", 2),
    )
    all_cmd.add_argument(
        "--evidence-mode",
        choices=["candidate_pool", "kg_only"],
        default=env_str("TRACECAG_BENCHMARK_EVIDENCE_MODE", "candidate_pool"),
    )
    all_cmd.add_argument(
        "--drift-split",
        choices=["train", "calibration", "test"],
        default=env_str("TRACECAG_BENCHMARK_DRIFT_SPLIT", "test"),
    )
    all_cmd.add_argument(
        "--drift-mode",
        choices=list(MODES),
        default=env_str("TRACECAG_BENCHMARK_DRIFT_MODE", "tracecag_rapid"),
    )
    _common_args(all_cmd)
    return parser


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default=env_str("TRACECAG_BENCHMARK_LLM_PROVIDER", "groq"),
    )
    parser.add_argument("--model", default=env_str("GROQ_MODEL", "qwen/qwen3-32b"))
    parser.add_argument(
        "--allow-fallback",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRACECAG_BENCHMARK_ALLOW_FALLBACK"),
    )
    parser.add_argument(
        "--allow-degraded-provider",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRACECAG_BENCHMARK_ALLOW_DEGRADED_PROVIDER"),
    )
    parser.add_argument(
        "--allow-degraded-kg",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRACECAG_BENCHMARK_ALLOW_DEGRADED_KG"),
    )
    parser.add_argument("--report-json", type=Path)


async def run_args(args: argparse.Namespace) -> int:
    if args.command == "all":
        return await _run_all(args)
    config = _config_from_args(args)
    config.apply_environment()
    if args.command == "public-qa":
        spec = get_dataset(args.dataset)
        samples = load_public_qa(spec.path, dataset=args.dataset, n=args.n, seed=args.seed)
        result = await run_public_qa_protocol(samples, get_modes(args.profile, args.modes), config)
        report = build_report(result=result, config=config, dataset_path=spec.path, dataset_name=args.dataset)
        path = args.report_json or config.report_dir / f"public-qa-{args.dataset}-{config.model.replace('/', '_')}.json"
    else:
        spec = get_dataset("trace_driftbench")
        path_for_split = spec.path.parent / f"{args.split}.jsonl"
        clusters = load_driftbench(path_for_split, n_clusters=args.n_clusters)
        result = await run_drift_safety_protocol(clusters, MODES[args.mode], config, split=args.split)
        report = build_report(result=result, config=config, dataset_path=path_for_split, dataset_name="trace_driftbench")
        path = args.report_json or config.report_dir / f"drift-safety-{args.split}-{config.model.replace('/', '_')}.json"
    write_report(report, path)
    print_summary(report)
    print(f"Report: {path}")
    return 0 if report["run_validation"]["passed"] else 2


async def _run_all(args: argparse.Namespace) -> int:
    exit_code = 0
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique", "query_clusters"):
        values = vars(args).copy()
        values.update(
            command="public-qa", dataset=dataset, profile=args.profile,
            modes=None, cache_repeats=args.cache_repeats,
            evidence_mode=args.evidence_mode, report_json=None,
        )
        namespace = argparse.Namespace(**values)
        exit_code = max(exit_code, await run_args(namespace))
    values = vars(args).copy()
    values.update(
        command="drift-safety",
        split=args.drift_split,
        mode=args.drift_mode,
        report_json=None,
    )
    namespace = argparse.Namespace(**values)
    return max(exit_code, await run_args(namespace))


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig.from_env(
        provider=args.provider,
        model=args.model,
        seed=getattr(args, "seed", 42),
        cache_repeats=getattr(args, "cache_repeats", 1),
        generation_policy=args.generation_policy,
        evidence_mode=getattr(args, "evidence_mode", "candidate_pool"),
        require_primary_provider=not args.allow_degraded_provider,
        allow_fallback=args.allow_fallback,
        allow_degraded_kg=args.allow_degraded_kg,
    )


def main() -> None:
    with _single_process_lock():
        raise SystemExit(asyncio.run(run_args(build_parser().parse_args())))


@contextmanager
def _single_process_lock():
    lock_path = Path("/tmp/tracecag-benchmark.lock")
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another TRACE-CAG benchmark process is already running") from exc
        yield
