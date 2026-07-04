from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextDocument:
    item_id: str
    title: str
    text: str


@dataclass(frozen=True)
class PublicQASample:
    sample_id: str
    dataset: str
    question: str
    answers: tuple[str, ...]
    context_docs: tuple[ContextDocument, ...]
    supporting_titles: tuple[str, ...]
    context: str = ""
    cluster_id: str | None = None
    cluster_variant: int | None = None
    expected_cache_decision: str | None = None
    expected_cache_layer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftRequest:
    request_id: str
    query: str
    expected_output: str
    expected_route: str
    safety_label: str
    drift_type: str
    state: dict[str, Any]


@dataclass(frozen=True)
class DriftCluster:
    cluster_id: str
    domain: str
    base: DriftRequest
    variants: tuple[DriftRequest, ...]


@dataclass
class RunObservation:
    sample_id: str
    mode: str
    answer: str = ""
    gold_answers: tuple[str, ...] = ()
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    cache_hit: bool = False
    cache_decision: str = "full"
    cache_layer: str = "none"
    reuse_risk: float = 1.0
    cache_gate_meta: dict[str, Any] = field(default_factory=dict)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    retrieval_meta: dict[str, Any] = field(default_factory=dict)
    kg_seed_concepts: list[str] = field(default_factory=list)
    kg_expanded_nodes: list[dict[str, Any]] = field(default_factory=list)
    graph_update: dict[str, Any] = field(default_factory=dict)
    models_used: list[str] = field(default_factory=list)
    observed_provider: str = ""
    observed_model: str = ""
    fallback_provider: str = ""
    expected_route: str = ""
    safety_label: str = ""
    drift_type: str = ""
    is_warm: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
