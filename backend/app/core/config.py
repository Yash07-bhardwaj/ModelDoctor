"""Runtime configuration (overridable with environment variables)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    storage_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODELDOCTOR_STORAGE_DIR", str(BACKEND_DIR / "storage")))
    )
    max_upload_mb: float = float(os.getenv("MODELDOCTOR_MAX_UPLOAD_MB", "50"))
    session_ttl_hours: float = float(os.getenv("MODELDOCTOR_SESSION_TTL_HOURS", "72"))
    fast_training: bool = _env_bool("MODELDOCTOR_FAST_TRAINING", False)
    max_batch_rows: int = int(os.getenv("MODELDOCTOR_MAX_BATCH_ROWS", "10000"))
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv(
            "MODELDOCTOR_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


settings = Settings()
