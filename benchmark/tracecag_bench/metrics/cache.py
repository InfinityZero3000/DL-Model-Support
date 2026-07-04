from __future__ import annotations

from collections import Counter

from ..schemas import RunObservation


def cache_metrics(observations: list[RunObservation]) -> dict:
    cold = [item for item in observations if not item.is_warm]
    warm = [item for item in observations if item.is_warm]
    decisions = Counter(item.cache_decision for item in observations)
    return {
        "cache_hit_rate": _rate(observations),
        "cold_hit_rate": _rate(cold),
        "warm_hit_rate": _rate(warm) if warm else None,
        "l0_rate": _layer_rate(observations, "L0"),
        "l1_rate": _layer_rate(observations, "L1"),
        "decision_distribution": dict(decisions),
    }


def _rate(items: list[RunObservation]) -> float:
    return sum(item.cache_hit for item in items) / len(items) if items else 0.0


def _layer_rate(items: list[RunObservation], layer: str) -> float:
    return sum(item.cache_layer == layer for item in items) / len(items) if items else 0.0
