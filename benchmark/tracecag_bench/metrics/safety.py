from __future__ import annotations

from ..schemas import RunObservation


def safety_metrics(observations: list[RunObservation]) -> dict[str, float]:
    scored = [item for item in observations if item.safety_label in {"safe", "unsafe"}]
    safe = [item for item in scored if item.safety_label == "safe"]
    unsafe = [item for item in scored if item.safety_label == "unsafe"]
    accepted = [item for item in scored if item.cache_decision in {"reuse", "patch"}]
    safe_accepted = [item for item in accepted if item.safety_label == "safe"]
    patchable = [item for item in safe if item.expected_route == "L1_patch"]
    return {
        "safe_reuse_precision": len(safe_accepted) / len(accepted) if accepted else 0.0,
        "admissible_recall": len(safe_accepted) / len(safe) if safe else 0.0,
        "unsafe_acceptance_rate": sum(item.cache_decision in {"reuse", "patch"} for item in unsafe) / len(unsafe) if unsafe else 0.0,
        "fallback_rate": sum(item.cache_decision == "full" for item in scored) / len(scored) if scored else 0.0,
        "patch_recall": sum(item.cache_decision == "patch" for item in patchable) / len(patchable) if patchable else 0.0,
        "route_accuracy": sum(_actual_route(item) == item.expected_route for item in scored) / len(scored) if scored else 0.0,
        "uncertain_count": float(sum(item.safety_label == "uncertain" for item in observations)),
    }


def calibrate_thresholds(
    observations: list[RunObservation],
    *,
    epsilon: float = 0.01,
    grid: tuple[float, ...] = (0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.55),
) -> dict[str, float]:
    safe = [item for item in observations if item.safety_label == "safe"]
    unsafe = [item for item in observations if item.safety_label == "unsafe"]
    best = {"tau_reuse": grid[0], "tau_patch": grid[0], "admissible_recall": 0.0, "unsafe_acceptance_rate": 0.0}
    for tau_reuse in grid:
        for tau_patch in grid:
            if tau_patch < tau_reuse:
                continue
            safe_accept = sum(_risk(item) <= tau_patch for item in safe)
            unsafe_accept = sum(_risk(item) <= tau_patch for item in unsafe)
            recall = safe_accept / len(safe) if safe else 0.0
            unsafe_rate = unsafe_accept / len(unsafe) if unsafe else 0.0
            if unsafe_rate <= epsilon and (recall, tau_patch, tau_reuse) > (
                best["admissible_recall"], best["tau_patch"], best["tau_reuse"]
            ):
                best = {
                    "tau_reuse": tau_reuse,
                    "tau_patch": tau_patch,
                    "admissible_recall": recall,
                    "unsafe_acceptance_rate": unsafe_rate,
                }
    return best


def _actual_route(item: RunObservation) -> str:
    if item.cache_decision == "full":
        return "L2"
    if item.cache_layer == "L0" and item.cache_decision == "reuse":
        return "L0"
    suffix = "patch" if item.cache_decision == "patch" else "reuse"
    return f"{item.cache_layer}_{suffix}" if item.cache_layer else suffix


def _risk(item: RunObservation) -> float:
    return float(item.cache_gate_meta.get("risk", item.reuse_risk))
