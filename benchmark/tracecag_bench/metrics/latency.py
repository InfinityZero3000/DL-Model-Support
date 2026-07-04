from __future__ import annotations

import statistics

from ..schemas import RunObservation


def latency_metrics(observations: list[RunObservation]) -> dict[str, float | None]:
    values = [item.latency_ms for item in observations]
    cold = [item.latency_ms for item in observations if not item.is_warm]
    warm = [item.latency_ms for item in observations if item.is_warm]
    return {
        "mean_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "cold_mean_ms": statistics.mean(cold) if cold else None,
        "warm_mean_ms": statistics.mean(warm) if warm else None,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]
