from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DEV = REPO_ROOT / "model-development"
DATASETS_DIR = MODEL_DEV / "datasets" / "benchmarks"
DEFAULT_REPORT_DIR = MODEL_DEV / "reports" / "benchmarks"


def load_env_file(path: Path = MODEL_DEV / ".env") -> None:
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    if os.getenv("GROQ_KEYS") and not os.getenv("GROQ_API_KEYS"):
        os.environ["GROQ_API_KEYS"] = os.environ["GROQ_KEYS"]
    if os.getenv("GROQ_API_KEYS") and not os.getenv("GROQ_API_KEY"):
        first_key = os.environ["GROQ_API_KEYS"].split(",", 1)[0].strip()
        if first_key:
            os.environ["GROQ_API_KEY"] = first_key


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class BenchmarkConfig:
    provider: str = "groq"
    model: str = "qwen/qwen3-32b"
    seed: int = 42
    cache_repeats: int = 2
    generation_policy: str = "auto"
    evidence_mode: str = "candidate_pool"
    require_primary_provider: bool = True
    allow_fallback: bool = False
    allow_degraded_kg: bool = False
    report_dir: Path = DEFAULT_REPORT_DIR

    @classmethod
    def from_env(cls, **overrides: object) -> "BenchmarkConfig":
        load_env_file()
        values: dict[str, object] = {
            "provider": env_str("TRACECAG_BENCHMARK_LLM_PROVIDER", "groq"),
            "model": env_str("GROQ_MODEL", "qwen/qwen3-32b"),
            "seed": env_int("TRACECAG_BENCHMARK_SEED", 42),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def apply_environment(self) -> None:
        os.environ["TRACECAG_BENCHMARK_LLM_PROVIDER"] = self.provider
        if self.provider == "groq":
            os.environ["GROQ_MODEL"] = self.model
        os.environ["TRACECAG_ENABLE_GEMINI_FALLBACK"] = "true" if self.allow_fallback else "false"
        os.environ.setdefault("BENCHMARK_REDIS_FAIL_FAST", "true")
        os.environ.setdefault("TRACECAG_ENABLE_JIT_MINIGRAPH", "false")
        os.environ.setdefault("TRACECAG_KG_SKIP_SYNC", "true")
        default_kuzu = REPO_ROOT / "data" / "kuzu_db"
        if default_kuzu.exists():
            os.environ.setdefault("KUZU_DB_PATH", str(default_kuzu))

    def public_dict(self, environ: Mapping[str, str] | None = None) -> dict[str, object]:
        data = asdict(self)
        data["report_dir"] = str(self.report_dir)
        env = environ or os.environ
        data["credential_counts"] = {
            "groq": _count_keys(env.get("GROQ_API_KEYS") or env.get("GROQ_KEYS") or env.get("GROQ_API_KEY", "")),
            "gemini": _count_keys(env.get("GEMINI_KEYS") or env.get("GEMINI_API_KEY", "")),
        }
        return data


def _count_keys(raw: str) -> int:
    return len([item for item in raw.split(",") if item.strip()])
