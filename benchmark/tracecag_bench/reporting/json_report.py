from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..config import BenchmarkConfig


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(*, result: dict, config: BenchmarkConfig, dataset_path: Path, dataset_name: str) -> dict:
    report = {
        "schema_version": 2,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "path": str(dataset_path),
            "sha256": dataset_sha256(dataset_path),
        },
        "configuration": config.public_dict(),
        **result,
    }
    report["run_validation"] = validate_run(report, config)
    return report


def validate_run(report: dict, config: BenchmarkConfig) -> dict:
    observations = report.get("observations") or []
    violations = []
    if config.require_primary_provider and config.generation_policy != "extractive":
        for item in observations:
            if item.get("cache_hit"):
                continue
            if item.get("observed_provider") != config.provider:
                violations.append(
                    f"{item.get('sample_id')}: expected {config.provider}, observed "
                    f"{item.get('observed_provider') or 'none'}"
                )
    kg = report.get("kg_preflight") or {}
    if config.evidence_mode == "kg_only" and not config.allow_degraded_kg and not kg.get("available"):
        violations.append("KG-only run requires an available Kuzu database")
    return {"passed": not violations, "violations": violations}


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
