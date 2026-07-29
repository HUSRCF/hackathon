"""Command-line entry point for local Agent, hardware evidence and benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .agent import Agent, AgentLimitError
from .backends import (
    BackendError,
    DeepSeekBackend,
    HipFireBackend,
    MockBackend,
    OpenAICompatibleBackend,
)
from .benchmark import BenchmarkConfig, load_suite, run_benchmark, save_benchmark
from .config import Settings
from .cross_verify import compare_results, load_result, report_json
from .hardware import hardware_json, probe_hardware
from .memory import JsonlMemoryStore, memory_tools
from .tools import SideEffect, ToolRegistry


def _backend(settings: Settings, backend_name: str | None = None):
    name = (backend_name or settings.backend).lower()
    if name == "mock":
        return MockBackend()
    if name == "hipfire":
        return HipFireBackend(
            settings.base_url,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
        )
    if name == "deepseek":
        return DeepSeekBackend(
            settings.base_url,
            api_key=settings.api_key or "",
            timeout_seconds=settings.timeout_seconds,
        )
    if name in {"openai", "openai-compatible", "vllm"}:
        return OpenAICompatibleBackend(
            settings.base_url,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
        )
    raise ValueError(f"unsupported backend: {name}")


def _file_sha256(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("hipfire", "deepseek", "openai-compatible", "vllm", "mock"),
        help="Override RADEON_AGENT_BACKEND.",
    )
    parser.add_argument("--base-url", help="Override RADEON_AGENT_BASE_URL.")
    parser.add_argument("--model", help="Override RADEON_AGENT_MODEL.")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Safely load dotenv assignments without executing shell code.",
    )
    parser.add_argument(
        "--approve-network",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="Explicitly approve one exact non-loopback backend domain for this command.",
    )


def _settings_with_overrides(args: argparse.Namespace) -> Settings:
    original = Settings.from_env(args.env_file, backend_override=args.backend)
    return Settings(
        backend=args.backend or original.backend,
        base_url=(args.base_url or original.base_url).rstrip("/"),
        model=args.model or original.model,
        api_key=original.api_key,
        timeout_seconds=original.timeout_seconds,
        max_steps=original.max_steps,
        memory_path=original.memory_path,
    )


def _is_loopback(hostname: str) -> bool:
    if hostname.lower().rstrip(".") == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _require_backend_approval(settings: Settings, approved: list[str]) -> None:
    if settings.backend == "mock":
        return
    parsed = urlparse(settings.base_url)
    if parsed.username or parsed.password:
        raise ValueError("backend URLs must not contain credentials")
    if not parsed.hostname:
        raise ValueError("backend base URL requires a hostname")
    hostname = parsed.hostname.lower().rstrip(".")
    if _is_loopback(hostname):
        return
    if settings.backend == "hipfire":
        raise PermissionError("HipFire may only bind through a loopback base URL")
    if parsed.scheme != "https":
        raise PermissionError("non-loopback inference backends require HTTPS")
    allowed = {domain.lower().rstrip(".") for domain in approved}
    if hostname not in allowed:
        raise PermissionError(
            f"network backend {hostname!r} requires --approve-network {hostname}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radeon-agent",
        description="Local-first Agentic AI framework for AMD Radeon/ROCm.",
    )
    parser.add_argument("--version", action="version", version="radeon-agent-lab 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="Capture ROCm/GPU evidence as JSON.")

    doctor = subparsers.add_parser(
        "doctor", help="Check GPU evidence and inference-server health."
    )
    _add_backend_arguments(doctor)

    models = subparsers.add_parser("models", help="List models from the inference server.")
    _add_backend_arguments(models)

    chat = subparsers.add_parser("chat", help="Run one bounded Agent task.")
    _add_backend_arguments(chat)
    chat.add_argument("prompt", help="Task for the local Agent.")
    chat.add_argument(
        "--enable-memory-writes",
        action="store_true",
        help="Allow the remember tool to append to the local JSONL memory.",
    )
    chat.add_argument("--json", action="store_true", help="Emit result metadata as JSON.")

    benchmark = subparsers.add_parser(
        "benchmark", help="Run a warm, repeated, hash-bound inference benchmark."
    )
    _add_backend_arguments(benchmark)
    benchmark.add_argument("--suite", type=Path, default=Path("benchmarks/suites/smoke.jsonl"))
    benchmark.add_argument("--label", required=True, help="Unique machine/run label.")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--repetitions", type=int, default=5)
    benchmark.add_argument("--warmup-runs", type=int, default=1)
    benchmark.add_argument("--quantization", required=True)
    benchmark.add_argument("--model-revision", required=True)
    benchmark.add_argument("--model-sha256", required=True)
    benchmark.add_argument("--code-revision", required=True)
    benchmark.add_argument("--runtime-revision", required=True)
    benchmark.add_argument(
        "--semantic-config",
        type=Path,
        default=Path("configs/semantic.toml"),
        help="Common workload config; this hash must match across GPUs.",
    )
    benchmark.add_argument(
        "--tuning-config",
        type=Path,
        help="Architecture-specific config; its hash is recorded but may differ across GPUs.",
    )

    compare = subparsers.add_parser("compare", help="Cross-verify two benchmark JSON files.")
    compare.add_argument("primary", type=Path)
    compare.add_argument("verifier", type=Path)
    compare.add_argument("--primary-arch", default="gfx1100")
    compare.add_argument("--verifier-arch", default="gfx1201")
    compare.add_argument("--strict-output", action="store_true")

    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "probe":
        print(hardware_json())
        return 0

    if args.command == "compare":
        report = compare_results(
            load_result(args.primary),
            load_result(args.verifier),
            expected_primary_arch=args.primary_arch,
            expected_verifier_arch=args.verifier_arch,
            strict_output=args.strict_output,
        )
        print(report_json(report))
        return 0 if report.compatible else 2

    settings = _settings_with_overrides(args)
    _require_backend_approval(settings, args.approve_network)
    backend = _backend(settings, args.backend)

    if args.command == "doctor":
        health = backend.health()
        advertised_models = health.get("models") if isinstance(health, dict) else None
        models = (
            tuple(advertised_models)
            if isinstance(advertised_models, list | tuple)
            else backend.list_models()
        )
        print(
            json.dumps(
                {
                    "hardware": probe_hardware().to_dict(),
                    "backend": backend.name,
                    "health": health,
                    "models": models,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "models":
        models = backend.list_models()
        if not models:
            print("The server returned no models.", file=sys.stderr)
            return 2
        for model in models:
            print(model)
        return 0

    if args.command == "chat":
        store = JsonlMemoryStore(settings.memory_path)
        max_effect = SideEffect.LOCAL_WRITE if args.enable_memory_writes else SideEffect.NONE
        registry = ToolRegistry(memory_tools(store), max_side_effect=max_effect)
        result = Agent(
            backend,
            model=settings.model,
            tools=registry,
            max_steps=settings.max_steps,
            timeout_seconds=max(settings.timeout_seconds * settings.max_steps, 300.0),
        ).run(args.prompt)
        if args.json:
            payload: dict[str, Any] = {
                "answer": result.answer,
                "model_calls": result.model_calls,
                "tool_calls": result.tool_calls,
                "elapsed_seconds": result.elapsed_seconds,
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(result.answer)
        return 0

    if args.command == "benchmark":
        suite = load_suite(args.suite)
        hardware = probe_hardware()
        if backend.name == "deepseek":
            print(
                "warning: DeepSeek is cloud-only functional evidence; this result is not "
                "eligible for Radeon/HipFire performance or cross-verification",
                file=sys.stderr,
            )
        if hardware.hsa_override_active:
            raise ValueError(
                "HSA_OVERRIDE_GFX_VERSION is active; remove architecture spoofing "
                "before benchmarking"
            )
        if not hardware.architectures:
            print(
                "warning: no gfx architecture was detected; result cannot pass cross-verification",
                file=sys.stderr,
            )
        config = BenchmarkConfig(
            label=args.label,
            model=settings.model,
            repetitions=args.repetitions,
            warmup_runs=args.warmup_runs,
            quantization=args.quantization,
            model_revision=args.model_revision,
            model_sha256=args.model_sha256,
            code_revision=args.code_revision,
            runtime_revision=args.runtime_revision,
            workload_config_sha256=_file_sha256(args.semantic_config),
            tuning_config_sha256=_file_sha256(args.tuning_config),
        )
        result = run_benchmark(backend, suite, config, hardware=hardware)
        save_benchmark(result, args.output)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"saved: {args.output}", file=sys.stderr)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (BackendError, AgentLimitError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
