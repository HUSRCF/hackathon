"""Typed, allow-listed Agent tools.

The registry deliberately has no generic shell tool. A model can only call tools
that the host application registered explicitly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class SideEffect(IntEnum):
    NONE = 0
    LOCAL_WRITE = 1
    EXTERNAL = 2


class ToolError(RuntimeError):
    pass


class ToolPermissionError(ToolError):
    pass


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    side_effect: SideEffect = SideEffect.NONE

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    value: Any = None
    error: str | None = None

    def as_message_content(self) -> str:
        return json.dumps(
            {"ok": self.ok, "value": self.value, "error": self.error},
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )


def _matches_type(value: Any, json_type: str) -> bool:
    checks = {
        "string": lambda candidate: isinstance(candidate, str),
        "number": lambda candidate: isinstance(candidate, int | float)
        and not isinstance(candidate, bool),
        "integer": lambda candidate: isinstance(candidate, int)
        and not isinstance(candidate, bool),
        "boolean": lambda candidate: isinstance(candidate, bool),
        "array": lambda candidate: isinstance(candidate, list),
        "object": lambda candidate: isinstance(candidate, dict),
        "null": lambda candidate: candidate is None,
    }
    checker = checks.get(json_type)
    return True if checker is None else checker(value)


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    """Validate the small JSON-Schema subset used by built-in tools."""

    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolError(f"missing required arguments: {', '.join(missing)}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ToolError(f"unknown arguments: {', '.join(unknown)}")

    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, Mapping):
            continue
        allowed_type = spec.get("type")
        if isinstance(allowed_type, str) and not _matches_type(value, allowed_type):
            raise ToolError(f"argument {key!r} must be {allowed_type}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"argument {key!r} must be one of {spec['enum']!r}")
        if isinstance(value, str):
            if "minLength" in spec and len(value) < spec["minLength"]:
                raise ToolError(f"argument {key!r} is too short")
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                raise ToolError(f"argument {key!r} is too long")


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[ToolSpec] = (),
        *,
        max_side_effect: SideEffect = SideEffect.NONE,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.max_side_effect = max_side_effect
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if not tool.name or not tool.name.replace("_", "a").isalnum():
            raise ValueError(f"invalid tool name: {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return tuple(tool.to_openai() for tool in self._tools.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")
        try:
            if tool.side_effect > self.max_side_effect:
                raise ToolPermissionError(
                    f"tool {name!r} requires {tool.side_effect.name}, "
                    f"policy allows {self.max_side_effect.name}"
                )
            validate_arguments(tool.parameters, arguments)
            return ToolResult(name=name, ok=True, value=tool.handler(arguments))
        except (ToolError, ValueError, TypeError) as exc:
            return ToolResult(name=name, ok=False, error=str(exc))
        except Exception as exc:  # tool boundary: never crash the Agent loop
            return ToolResult(
                name=name,
                ok=False,
                error=f"tool failed: {type(exc).__name__}: {exc}",
            )
