from __future__ import annotations

import json
from pathlib import Path

from ..schemas import DriftCluster, DriftRequest


def load_driftbench(path: Path, *, n_clusters: int | None = None) -> list[DriftCluster]:
    clusters: list[DriftCluster] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            base_raw = row["base"]
            cluster = DriftCluster(
                cluster_id=str(row["cluster_id"]),
                domain=str(row.get("domain") or ""),
                base=_request(base_raw, expected_route="L2", safety_label="uncertain", drift_type="base"),
                variants=tuple(
                    _request(
                        {**base_raw, **variant},
                        expected_route=str(variant.get("expected_route") or "L2"),
                        safety_label=str(variant.get("safety_label") or "uncertain"),
                        drift_type=str(variant.get("drift_type") or ""),
                    )
                    for variant in row.get("variants", [])
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid drift cluster: {exc}") from exc
        clusters.append(cluster)
    return clusters[:n_clusters] if n_clusters is not None else clusters


def _request(row: dict, *, expected_route: str, safety_label: str, drift_type: str) -> DriftRequest:
    state_keys = (
        "intent", "level", "concepts", "graph_neighborhood", "profile_epoch",
        "policy_version", "kg_version", "answer_target", "relation_path", "evidence_hash",
    )
    return DriftRequest(
        request_id=str(row["request_id"]),
        query=str(row["query"]),
        expected_output=str(row.get("expected_output") or ""),
        expected_route=expected_route,
        safety_label=safety_label,
        drift_type=drift_type,
        state={key: row.get(key) for key in state_keys},
    )
