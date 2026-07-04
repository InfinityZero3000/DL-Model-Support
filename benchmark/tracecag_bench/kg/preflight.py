from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ..config import REPO_ROOT


def run_kg_preflight(titles: Iterable[str] = ()) -> dict:
    default_path = REPO_ROOT / "data" / "kuzu_db"
    root = Path(os.getenv("KUZU_DB_PATH", str(default_path)))
    exists = root.exists()
    title_list = [item for item in titles if item]
    snapshot = {
        "path": str(root.resolve()) if exists else str(root),
        "available": exists,
        "file_count": sum(1 for item in root.rglob("*") if item.is_file()) if exists and root.is_dir() else int(exists),
        "requested_title_count": len(set(title_list)),
        "status": "ready" if exists else "missing",
    }
    if exists:
        try:
            import kuzu  # noqa: F401
        except ImportError as exc:
            snapshot["status"] = "driver_missing"
            snapshot["driver_error"] = str(exc)
    return snapshot
