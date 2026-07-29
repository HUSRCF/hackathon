"""Explicit, fail-closed OpenCode launcher for DeepSeek cloud debugging."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .config import read_env_file

DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_BASE_URL = f"https://{DEEPSEEK_HOST}"
DEEPSEEK_PROVIDER = "deepseek-cloud"
SUPPORTED_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
DEFAULT_MODEL = SUPPORTED_MODELS[0]


def build_inline_config(model: str = DEFAULT_MODEL) -> dict[str, object]:
    """Return the narrow runtime override; never include the credential value."""

    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported DeepSeek model: {model}")

    models = {
        model_name: {
            "name": f"{model_name} (cloud debug only)",
            # Keep debug responses bounded even though the service supports a
            # substantially larger maximum output.
            "limit": {"context": 1_000_000, "output": 8_192},
            # DeepSeek V4 defaults to thinking. Disable it until OpenCode's
            # OpenAI-compatible tool loop is regression-tested for lossless
            # reasoning_content round trips.
            "options": {"thinking": {"type": "disabled"}},
        }
        for model_name in SUPPORTED_MODELS
    }
    selected = f"{DEEPSEEK_PROVIDER}/{model}"
    return {
        "enabled_providers": [DEEPSEEK_PROVIDER],
        "model": selected,
        "small_model": selected,
        "provider": {
            DEEPSEEK_PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DeepSeek cloud (debug only)",
                "options": {
                    "baseURL": DEEPSEEK_BASE_URL,
                    "apiKey": "{env:DEEPSEEK_API_KEY}",
                    "timeout": 300_000,
                },
                "models": models,
            }
        },
    }


def _credential(
    env: Mapping[str, str],
    env_file: Path | None,
) -> str:
    values = read_env_file(env_file) if env_file is not None else {}
    values.update(env)
    credential = values.get("DEEPSEEK_API_KEY", "").strip()
    if not credential:
        source = f" or {env_file}" if env_file is not None else ""
        raise PermissionError(
            f"DEEPSEEK_API_KEY is required in the environment{source}"
        )
    return credential


def prepare_environment(
    *,
    env: Mapping[str, str],
    env_file: Path | None,
    approved_domains: Sequence[str],
    model: str,
) -> dict[str, str]:
    """Build a child environment after exact-domain and credential checks."""

    approved = {domain.lower().rstrip(".") for domain in approved_domains}
    if approved != {DEEPSEEK_HOST}:
        raise PermissionError(
            f"cloud debugging requires exactly --approve-network {DEEPSEEK_HOST}"
        )
    if env.get("OPENCODE_CONFIG_CONTENT"):
        raise PermissionError(
            "OPENCODE_CONFIG_CONTENT must be unset so the safety override cannot collide"
        )

    child_env = dict(env)
    child_env["DEEPSEEK_API_KEY"] = _credential(env, env_file)
    child_env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        build_inline_config(model), separators=(",", ":"), sort_keys=True
    )
    return child_env


def _validate_passthrough(arguments: Sequence[str]) -> list[str]:
    passthrough = list(arguments)
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    blocked = {"--auto", "-m", "--model", "--agent", "--prompt"}
    for argument in passthrough:
        if argument in blocked or any(
            argument.startswith(f"{option}=")
            for option in ("--model", "--agent", "--prompt")
        ):
            raise PermissionError(
                f"OpenCode argument {argument!r} is not allowed by the cloud debug launcher"
            )
    return passthrough


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch OpenCode through DeepSeek only after an explicit exact-domain "
            "network approval. The normal project configuration remains local."
        )
    )
    parser.add_argument(
        "--approve-network",
        action="append",
        default=[],
        metavar="DOMAIN",
        help=f"Required exact approval: {DEEPSEEK_HOST}",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Safely read DEEPSEEK_API_KEY without executing shell syntax.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help="DeepSeek cloud model used by OpenCode.",
    )
    parser.add_argument(
        "opencode_arguments",
        nargs=argparse.REMAINDER,
        help="Optional OpenCode arguments after --.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        passthrough = _validate_passthrough(args.opencode_arguments)
        child_env = prepare_environment(
            env=os.environ,
            env_file=args.env_file,
            approved_domains=args.approve_network,
            model=args.model,
        )
    except (OSError, PermissionError, ValueError) as exc:
        print(f"refusing DeepSeek cloud launch: {exc}", file=sys.stderr)
        return 2

    executable = shutil.which("opencode")
    if executable is None:
        print("refusing DeepSeek cloud launch: opencode is not installed", file=sys.stderr)
        return 127

    os.execvpe(executable, [executable, *passthrough], child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
