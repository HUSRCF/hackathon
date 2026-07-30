"""Small dependency-free client for OpenAI-compatible inference servers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ..models import ChatRequest, ChatResponse, StreamTiming, ToolCall, Usage
from .base import BackendError


class OpenAICompatibleBackend:
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        user_agent: str = "radeon-agent-lab/0.1",
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, path: str, payload: dict[str, Any] | None = None):
        body = None
        method = "GET"
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            method = "POST"
        url = path if path.startswith(("http://", "https://")) else (
            f"{self.base_url}/{path.lstrip('/')}"
        )
        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")[:1000]
                if self.api_key:
                    details = details.replace(self.api_key, "<redacted>")
                retryable = exc.code in {429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    try:
                        delay = float(exc.headers.get("Retry-After", 2**attempt))
                    except (TypeError, ValueError):
                        delay = float(2**attempt)
                    time.sleep(min(max(delay, 0.1), 10.0))
                    continue
                raise BackendError(
                    f"HTTP {exc.code} from inference server: {details}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                raise BackendError(
                    f"Cannot reach inference server at {self.base_url}: {exc}"
                ) from exc
        raise AssertionError("retry loop exhausted unexpectedly")

    @staticmethod
    def _usage(payload: dict[str, Any] | None) -> Usage:
        payload = payload or {}
        return Usage(
            prompt_tokens=payload.get("prompt_tokens"),
            completion_tokens=payload.get("completion_tokens"),
            total_tokens=payload.get("total_tokens"),
        )

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]] | None) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index, raw in enumerate(raw_calls or []):
            function = raw.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise BackendError(
                        f"Model returned invalid JSON for tool {function.get('name')!r}"
                    ) from exc
            if not isinstance(arguments, dict):
                raise BackendError("Tool arguments must decode to a JSON object")
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )
        return tuple(calls)

    def complete(self, request: ChatRequest) -> ChatResponse:
        with self._request("chat/completions", request.to_openai()) as response:
            try:
                payload = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise BackendError("Inference server returned invalid JSON") from exc
        choices = payload.get("choices") or []
        if payload.get("error"):
            raise BackendError(f"Inference server error: {payload['error']}")
        if not choices:
            raise BackendError("Inference server returned no completion choices")
        choice = choices[0]
        message = choice.get("message") or {}
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message.get("tool_calls")),
            usage=self._usage(payload.get("usage")),
            finish_reason=choice.get("finish_reason"),
            model=payload.get("model"),
        )

    def stream_complete(self, request: ChatRequest) -> tuple[ChatResponse, StreamTiming]:
        payload = request.to_openai(stream=True)
        payload.setdefault("stream_options", {"include_usage": True})
        start = time.perf_counter()
        first_token_at: float | None = None
        content_parts: list[str] = []
        finish_reason: str | None = None
        model: str | None = None
        usage = Usage()
        streamed_tool_calls: dict[int, dict[str, Any]] = {}
        with self._request("chat/completions", payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BackendError(f"Invalid SSE chunk: {line[:200]}") from exc
                if chunk.get("error"):
                    raise BackendError(f"Inference stream error: {chunk['error']}")
                model = chunk.get("model") or model
                if chunk.get("usage"):
                    usage = self._usage(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    reasoning_piece = delta.get("reasoning_content")
                    if reasoning_piece and first_token_at is None:
                        first_token_at = time.perf_counter()
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        content_parts.append(piece)
                    for raw_call in delta.get("tool_calls") or []:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        index = int(raw_call.get("index", 0))
                        accumulated = streamed_tool_calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if raw_call.get("id"):
                            accumulated["id"] += str(raw_call["id"])
                        function = raw_call.get("function") or {}
                        if function.get("name"):
                            accumulated["name"] += str(function["name"])
                        if function.get("arguments"):
                            accumulated["arguments"] += str(function["arguments"])
                    finish_reason = choice.get("finish_reason") or finish_reason
        end = time.perf_counter()
        timing = StreamTiming(
            total_seconds=end - start,
            time_to_first_token_seconds=(
                first_token_at - start if first_token_at is not None else None
            ),
        )
        raw_calls = [
            {
                "id": value["id"] or f"call_{index}",
                "function": {
                    "name": value["name"],
                    "arguments": value["arguments"] or "{}",
                },
            }
            for index, value in sorted(streamed_tool_calls.items())
        ]
        return (
            ChatResponse(
                content="".join(content_parts),
                tool_calls=self._parse_tool_calls(raw_calls),
                usage=usage,
                finish_reason=finish_reason,
                model=model,
            ),
            timing,
        )

    def list_models(self) -> tuple[str, ...]:
        with self._request("models") as response:
            try:
                payload = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise BackendError("Inference server returned invalid model-list JSON") from exc
        return tuple(
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        )

    def health(self) -> dict:
        return {"status": "ok", "models": self.list_models()}
