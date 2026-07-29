"""Environment-based configuration without secret-bearing config files."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def read_env_file(path: Path) -> dict[str, str]:
    """Read a small dotenv subset without executing shell syntax or interpolation."""

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid env assignment at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"invalid env key at {path}:{line_number}")
        if value[:1] in {'"', "'"}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"unterminated quoted value at {path}:{line_number}")
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    value = int(raw) if raw else default
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    value = float(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    backend: str = "hipfire"
    base_url: str = "http://127.0.0.1:11435/v1"
    model: str = "qwen3.5:9b"
    api_key: str | None = None
    timeout_seconds: float = 300.0
    max_steps: int = 6
    memory_path: Path = Path("artifacts/memory.jsonl")

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = None,
        *,
        backend_override: str | None = None,
    ) -> Settings:
        values = read_env_file(env_file) if env_file is not None else {}
        # Exported environment variables take precedence over the dotenv file.
        values.update(os.environ)
        # Merely discovering a cloud credential must never opt a local-private
        # process into network use. Cloud backends require an explicit backend
        # selection and a separate per-command network approval in the CLI.
        backend = backend_override or values.get("RADEON_AGENT_BACKEND") or "hipfire"
        backend = backend.strip().lower()
        deepseek = backend == "deepseek"
        default_base_url = (
            values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if deepseek
            else "http://127.0.0.1:11435/v1"
        )
        default_model = (
            values.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
            if deepseek
            else "qwen3.5:9b"
        )
        return cls(
            backend=backend,
            base_url=values.get("RADEON_AGENT_BASE_URL", default_base_url).rstrip("/"),
            model=values.get("RADEON_AGENT_MODEL", default_model),
            api_key=(
                values.get("RADEON_AGENT_API_KEY")
                or (values.get("DEEPSEEK_API_KEY") if deepseek else None)
                or None
            ),
            timeout_seconds=_positive_float(
                values, "RADEON_AGENT_TIMEOUT_SECONDS", 300.0
            ),
            max_steps=_positive_int(values, "RADEON_AGENT_MAX_STEPS", 6),
            memory_path=Path(
                values.get("RADEON_AGENT_MEMORY_PATH", "artifacts/memory.jsonl")
            ),
        )
