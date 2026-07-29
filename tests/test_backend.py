from __future__ import annotations

import io
import json

import pytest

from radeon_agent.backends import (
    BackendError,
    DeepSeekBackend,
    HipFireBackend,
    OpenAICompatibleBackend,
)
from radeon_agent.models import ChatRequest, Message


def _json_response(payload: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode())


def test_non_streaming_response_and_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OpenAICompatibleBackend("http://127.0.0.1:9999/v1")
    payload = {
        "model": "demo",
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q":"AMD"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
    }
    monkeypatch.setattr(backend, "_request", lambda *_args, **_kwargs: _json_response(payload))

    result = backend.complete(
        ChatRequest(model="demo", messages=(Message(role="user", content="hi"),))
    )

    assert result.tool_calls[0].name == "lookup"
    assert result.tool_calls[0].arguments == {"q": "AMD"}
    assert result.usage.total_tokens == 11


def test_streaming_response_tracks_first_content(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OpenAICompatibleBackend("http://127.0.0.1:9999/v1")
    sse = (
        'data: {"model":"demo","choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"好"},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,'
        '"total_tokens":4}}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    monkeypatch.setattr(backend, "_request", lambda *_args, **_kwargs: io.BytesIO(sse))

    result, timing = backend.stream_complete(
        ChatRequest(model="demo", messages=(Message(role="user", content="hi"),))
    )

    assert result.content == "你好"
    assert result.usage.completion_tokens == 2
    assert timing.time_to_first_token_seconds is not None
    assert timing.total_seconds >= timing.time_to_first_token_seconds


def test_invalid_tool_json_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OpenAICompatibleBackend("http://127.0.0.1:9999/v1")
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bad", "arguments": "{"}}
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(backend, "_request", lambda *_args, **_kwargs: _json_response(payload))

    with pytest.raises(BackendError, match="invalid JSON"):
        backend.complete(
            ChatRequest(model="demo", messages=(Message(role="user", content="hi"),))
        )


def test_stream_embedded_error_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OpenAICompatibleBackend("http://127.0.0.1:9999/v1")
    sse = b'data: {"error":{"message":"queue full"}}\n\ndata: [DONE]\n\n'
    monkeypatch.setattr(backend, "_request", lambda *_args, **_kwargs: io.BytesIO(sse))

    with pytest.raises(BackendError, match="queue full"):
        backend.stream_complete(
            ChatRequest(model="demo", messages=(Message(role="user", content="hi"),))
        )


def test_hipfire_health_redacts_server_token(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HipFireBackend("http://127.0.0.1:11435/v1")
    monkeypatch.setattr(
        backend,
        "_request",
        lambda *_args, **_kwargs: _json_response(
            {"status": "ok", "model": "demo", "token": "do-not-print"}
        ),
    )

    health = backend.health()

    assert health["status"] == "ok"
    assert health["token"] == "<redacted>"


def test_deepseek_forces_non_thinking_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DeepSeekBackend(api_key="test-secret")
    captured: dict = {}

    def fake_request(_path: str, payload: dict):
        captured.update(payload)
        return _json_response(
            {
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr(backend, "_request", fake_request)

    result = backend.complete(
        ChatRequest(
            model="deepseek-v4-flash",
            messages=(Message(role="user", content="hi"),),
            extra={"thinking": {"type": "enabled"}},
        )
    )

    assert result.content == "ok"
    assert captured["thinking"] == {"type": "disabled"}
