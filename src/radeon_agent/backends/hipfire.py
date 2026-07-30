"""HipFire adapter.

HipFire remains an external process. This module intentionally contains no copied
HipFire implementation code and talks only to its OpenAI-compatible HTTP API.
"""

import json
from pathlib import Path

from .base import BackendError
from .openai_compatible import OpenAICompatibleBackend


class HipFireBackend(OpenAICompatibleBackend):
    name = "hipfire"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11435/v1",
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            user_agent="radeon-agent-lab/0.1 (hipfire-adapter)",
        )

    def health(self) -> dict:
        root_url = self.base_url.removesuffix("/v1")
        with self._request(f"{root_url}/health") as response:
            try:
                payload = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise BackendError("HipFire returned invalid health JSON") from exc
        if not isinstance(payload, dict):
            raise BackendError("HipFire returned a non-object health response")
        redacted: dict = {}
        for key, value in payload.items():
            sensitive = any(word in key.lower() for word in ("token", "key", "secret"))
            if sensitive:
                redacted[key] = "<redacted>"
            elif (
                key.lower() in {"model", "path", "model_path"}
                and isinstance(value, str)
                and Path(value).is_absolute()
            ):
                redacted[key] = Path(value).name
            else:
                redacted[key] = value
        return redacted
