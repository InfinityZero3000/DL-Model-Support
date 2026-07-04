from __future__ import annotations

from collections import defaultdict

from ..catalog import ModeSpec
from ..config import BenchmarkConfig
from ..kg.preflight import run_kg_preflight
from ..metrics.latency import latency_metrics
from ..metrics.safety import calibrate_thresholds, safety_metrics
from ..runtime.ai_service import AIServiceRuntime
from ..runtime.reset import reset_runtime_state
from ..schemas import DriftCluster, RunObservation


async def run_drift_safety_protocol(
    clusters: list[DriftCluster],
    mode: ModeSpec,
    config: BenchmarkConfig,
    *,
    runtime: AIServiceRuntime | None = None,
    split: str = "test",
) -> dict:
    runtime = runtime or AIServiceRuntime()
    observations: list[RunObservation] = []
    resets = []
    for cluster in clusters:
        resets.append({"scope": "cluster", "cluster": cluster.cluster_id, "cleared": reset_runtime_state()})
        session_id = f"drift_{cluster.cluster_id}"
        await runtime.run_drift(
            cluster.base, mode, session_id=session_id,
            generation_policy=config.generation_policy,
        )
        for variant in cluster.variants:
            observations.append(await runtime.run_drift(
                variant, mode, session_id=session_id,
                generation_policy=config.generation_policy,
            ))
    by_drift: dict[str, list[RunObservation]] = defaultdict(list)
    for item in observations:
        by_drift[item.drift_type].append(item)
    result = {
        "protocol": "drift_safety",
        "split": split,
        "mode": mode.name,
        "summary": {
            **safety_metrics(observations),
            "latency": latency_metrics(observations),
            "errors": sum(bool(item.error) for item in observations),
        },
        "by_drift": {name: safety_metrics(items) for name, items in sorted(by_drift.items())},
        "observations": [item.as_dict() for item in observations],
        "resets": resets,
        "kg_preflight": run_kg_preflight(),
    }
    if split == "calibration":
        result["calibration"] = {
            "epsilon": 0.01,
            "selected_thresholds": calibrate_thresholds(observations, epsilon=0.01),
        }
    return result
