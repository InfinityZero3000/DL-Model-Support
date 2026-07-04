from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DATASETS_DIR


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    protocol: str
    rationale: str


@dataclass(frozen=True)
class ModeSpec:
    name: str
    label: str
    cache_policy: str
    retrieval_policy: str
    ranker: str
    proxy: bool = False


DATASETS = {
    "hotpotqa": DatasetSpec("hotpotqa", DATASETS_DIR / "hotpotqa" / "validation.jsonl", "public_qa", "Bridge and comparison multi-hop QA."),
    "2wikimultihopqa": DatasetSpec("2wikimultihopqa", DATASETS_DIR / "2wikimultihopqa" / "validation.jsonl", "public_qa", "Wikipedia entity-link chains."),
    "musique": DatasetSpec("musique", DATASETS_DIR / "musique" / "validation.jsonl", "public_qa", "Composed two-to-four-hop reasoning."),
    "query_clusters": DatasetSpec("query_clusters", DATASETS_DIR / "query_clusters" / "validation.jsonl", "public_qa", "Paraphrase clusters for L1 cache coverage."),
    "trace_driftbench": DatasetSpec("trace_driftbench", DATASETS_DIR / "trace_driftbench" / "test.jsonl", "drift_safety", "Clustered state-drift safety probes."),
}

MODES = {
    "cag_vanilla": ModeSpec("cag_vanilla", "Vanilla CAG", "on", "full", "flat"),
    "hipporag_proxy": ModeSpec("hipporag_proxy", "HippoRAG proxy", "off", "full", "memory", proxy=True),
    "tracecag_rapid": ModeSpec("tracecag_rapid", "TRACE-CAG", "on", "rapid", "graph"),
    "tracecag_adaptive": ModeSpec("tracecag_adaptive", "TRACE-CAG adaptive", "on", "rapid", "graph"),
    "l2_only": ModeSpec("l2_only", "L2 only", "off", "full", "graph"),
}

PROFILES = {
    "public_cag_compare": ("cag_vanilla", "hipporag_proxy", "tracecag_rapid"),
    "core": ("tracecag_rapid",),
    "adaptive": ("tracecag_rapid", "tracecag_adaptive"),
}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc


def get_modes(profile: str, requested: list[str] | None = None) -> list[ModeSpec]:
    names = requested or list(PROFILES.get(profile, ()))
    if not names:
        raise ValueError(f"Unknown or empty comparison profile: {profile}")
    try:
        return [MODES[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"Unknown mode: {exc.args[0]}") from exc
