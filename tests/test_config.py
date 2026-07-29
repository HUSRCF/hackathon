from __future__ import annotations

import pytest

from radeon_agent.cli import _require_backend_approval
from radeon_agent.config import Settings, read_env_file


def test_deepseek_env_file_is_loaded_without_shell_execution(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY='test-secret'\n"
        "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
        "DEEPSEEK_MODEL=deepseek-v4-flash\n"
        "UNUSED_LITERAL=$(must-not-run)\n",
        encoding="utf-8",
    )
    for key in (
        "RADEON_AGENT_BACKEND",
        "RADEON_AGENT_BASE_URL",
        "RADEON_AGENT_MODEL",
        "RADEON_AGENT_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    local_settings = Settings.from_env(env_file)
    settings = Settings.from_env(env_file, backend_override="deepseek")

    assert local_settings.backend == "hipfire"
    assert local_settings.base_url == "http://127.0.0.1:11435/v1"
    assert local_settings.api_key is None
    assert settings.backend == "deepseek"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-v4-flash"
    assert settings.api_key == "test-secret"
    assert read_env_file(env_file)["UNUSED_LITERAL"] == "$(must-not-run)"


def test_exported_environment_takes_precedence(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-process")
    monkeypatch.delenv("RADEON_AGENT_BACKEND", raising=False)

    settings = Settings.from_env(env_file, backend_override="deepseek")

    assert settings.model == "from-process"


def test_remote_backend_requires_exact_per_command_network_approval() -> None:
    remote = Settings(
        backend="deepseek",
        base_url="https://api.deepseek.com",
        model="fixture",
        api_key="not-used",
    )

    with pytest.raises(PermissionError, match="approve-network"):
        _require_backend_approval(remote, [])
    _require_backend_approval(remote, ["api.deepseek.com"])
    with pytest.raises(PermissionError, match="loopback"):
        _require_backend_approval(
            Settings(backend="hipfire", base_url="http://lan-host:11435/v1"),
            ["lan-host"],
        )
