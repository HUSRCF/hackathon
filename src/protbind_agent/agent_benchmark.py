"""Radeon/HipFire benchmark for the actual bounded ProtBind Agent workload."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radeon_agent.agent import AgentLimitError
from radeon_agent.backends import HipFireBackend
from radeon_agent.hardware import probe_hardware

from .agent_runtime import create_runtime, require_loopback_hipfire_url
from .artifacts import canonical_json_bytes, sha256_file
from .workflow import PipelineConfig

AGENT_BENCHMARK_SCHEMA_VERSION = "1.0"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_VRAM = re.compile(r"GPU\[(\d+)\].*VRAM Total Used Memory \(B\):\s*(\d+)")


@dataclass(frozen=True, slots=True)
class AgentBenchmarkConfig:
    label: str
    model: str
    model_revision: str
    model_sha256: str
    quantization: str
    hipfire_revision: str
    hipfire_visible_device: int
    hipfire_speculation: str
    hipfire_jinja_mode: str
    code_revision: str
    repetitions: int = 3
    warmup_runs: int = 1
    tool_routing: bool = True

    def __post_init__(self) -> None:
        if self.repetitions < 3:
            raise ValueError("Agent benchmark requires at least three repetitions")
        if self.warmup_runs < 1:
            raise ValueError("Agent benchmark requires at least one warmup")
        if self.hipfire_visible_device < 0:
            raise ValueError("hipfire_visible_device must be >= 0")
        if self.hipfire_speculation not in {"off", "auto", "ngram", "dflash", "mtp"}:
            raise ValueError("unsupported HipFire speculation mode")
        if self.hipfire_jinja_mode not in {
            "default-on",
            "explicit-on",
            "explicit-off",
        }:
            raise ValueError("unsupported HipFire Jinja mode")
        for name in (
            "label",
            "model",
            "model_revision",
            "quantization",
            "hipfire_revision",
            "code_revision",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if _DIGEST.fullmatch(self.model_sha256) is None:
            raise ValueError("model_sha256 must be a lowercase SHA-256 digest")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
    }


def _model_weights_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        digest = sha256_file(resolved)
        file_count = 1
        total_bytes = resolved.stat().st_size
        semantics = "SHA-256 of the exact model file bytes"
    elif resolved.is_dir():
        entries: list[tuple[str, str, int]] = []
        for candidate in sorted(resolved.rglob("*")):
            if candidate.is_symlink():
                raise ValueError("model weights directory cannot contain symlinks")
            if not candidate.is_file():
                continue
            size = candidate.stat().st_size
            entries.append(
                (
                    candidate.relative_to(resolved).as_posix(),
                    sha256_file(candidate),
                    size,
                )
            )
        if not entries:
            raise ValueError("model weights directory has no regular files")
        digest = _sha256(canonical_json_bytes(entries))
        file_count = len(entries)
        total_bytes = sum(entry[2] for entry in entries)
        semantics = (
            "SHA-256 of canonical JSON [relative_name,file_sha256,size] entries"
        )
    else:
        raise FileNotFoundError("model weights path is unavailable")
    if digest != expected_sha256:
        raise ValueError("measured model weights SHA-256 does not match benchmark config")
    return {
        "sha256": digest,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "digest_semantics": semantics,
        "path_disclosed": False,
    }


def _file_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError("receipt input is not a regular file")
    return {
        "sha256": sha256_file(resolved),
        "total_bytes": resolved.stat().st_size,
        "path_disclosed": False,
    }


def _proc_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as exc:
        raise ValueError("cannot inspect the HipFire service environment") from exc
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def _proc_children(pid: int) -> tuple[int, ...]:
    pending = [pid]
    descendants: list[int] = []
    while pending:
        parent = pending.pop()
        try:
            raw = Path(f"/proc/{parent}/task/{parent}/children").read_text(
                encoding="utf-8"
            )
        except OSError:
            continue
        for value in raw.split():
            child = int(value)
            if child not in descendants:
                descendants.append(child)
                pending.append(child)
    return tuple(descendants)


def _loaded_hip_runtime_receipt(pid: int) -> dict[str, Any]:
    try:
        lines = Path(f"/proc/{pid}/maps").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        raise ValueError("cannot inspect the HipFire daemon mappings") from exc
    libraries: dict[str, Path] = {}
    for line in lines:
        path_text = line.rsplit(maxsplit=1)[-1]
        if not path_text.startswith("/") or "libamdhip64.so" not in path_text:
            continue
        candidate = Path(path_text)
        if candidate.is_file():
            libraries[str(candidate.resolve())] = candidate.resolve()
    if not libraries:
        raise ValueError("HipFire daemon has no measurable loaded HIP runtime")
    entries = []
    versions: set[str] = set()
    for resolved in sorted(libraries.values()):
        match = re.search(r"/(?:core-|rocm-)([0-9][^/]*)/", str(resolved))
        if match:
            versions.add(match.group(1))
        entries.append(
            {
                "library_name": resolved.name,
                "sha256": sha256_file(resolved),
                "total_bytes": resolved.stat().st_size,
            }
        )
    return {
        "libraries": entries,
        "version_hints": sorted(versions),
        "path_disclosed": False,
    }


def _service_process_receipt(
    health: dict[str, Any],
    *,
    daemon_path: Path,
    config: AgentBenchmarkConfig,
) -> dict[str, Any]:
    try:
        server_pid = int(health["pid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("HipFire health does not expose a valid local process ID") from exc
    if server_pid <= 1:
        raise ValueError("HipFire health returned an inadmissible process ID")
    environment = _proc_environ(server_pid)
    expected_environment = {
        "HIPFIRE_MODEL": config.model,
        "HIP_VISIBLE_DEVICES": str(config.hipfire_visible_device),
        "HIPFIRE_SPECULATION": config.hipfire_speculation,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"HipFire service setting {key} does not match benchmark config")
    jinja_value = environment.get("HIPFIRE_JINJA_CHAT")
    expected_jinja = {
        "default-on": None,
        "explicit-on": "1",
        "explicit-off": "0",
    }[config.hipfire_jinja_mode]
    if jinja_value != expected_jinja:
        raise ValueError("HipFire Jinja mode does not match benchmark config")

    daemon_resolved = daemon_path.resolve()
    daemon_pid = None
    for candidate_pid in _proc_children(server_pid):
        try:
            executable = Path(os.readlink(f"/proc/{candidate_pid}/exe")).resolve()
        except OSError:
            continue
        if executable == daemon_resolved:
            daemon_pid = candidate_pid
            break
    if daemon_pid is None:
        raise ValueError("HipFire endpoint is not backed by the nominated daemon binary")
    server_executable = Path(os.readlink(f"/proc/{server_pid}/exe")).resolve()
    return {
        "verified": True,
        "server_pid": server_pid,
        "daemon_pid": daemon_pid,
        "server_executable": _file_receipt(server_executable),
        "daemon_executable": _file_receipt(daemon_resolved),
        "runtime_settings": {
            "model": config.model,
            "visible_device": config.hipfire_visible_device,
            "speculation": config.hipfire_speculation,
            "jinja_mode": config.hipfire_jinja_mode,
            "thinking": "disabled-per-request",
        },
        "loaded_hip_runtime": _loaded_hip_runtime_receipt(daemon_pid),
        "process_paths_disclosed": False,
    }


def _successful_subsequence(
    events: list[dict[str, Any]],
    required: list[str],
) -> bool:
    cursor = 0
    for event in events:
        if (
            cursor < len(required)
            and event.get("ok") is True
            and event.get("tool") == required[cursor]
        ):
            cursor += 1
    return cursor == len(required)


def _git_receipt(root: Path, expected_revision: str) -> dict[str, Any]:
    resolved = root.resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=resolved,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("cannot verify benchmark source revision") from exc
    if revision != expected_revision:
        raise ValueError("measured source revision does not match benchmark config")
    dirty_entries = len([line for line in status.splitlines() if line.strip()])
    return {
        "revision": revision,
        "clean": dirty_entries == 0,
        "dirty_entry_count": dirty_entries,
        "porcelain_status_sha256": _sha256(status.encode("utf-8")),
        "path_disclosed": False,
    }


class _VRAMSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.peaks: dict[int, int] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        values = {
            int(device): int(value)
            for device, value in _VRAM.findall(f"{result.stdout}\n{result.stderr}")
        }
        if values:
            self.samples += 1
        for device, value in values.items():
            self.peaks[device] = max(self.peaks.get(device, 0), value)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> _VRAMSampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._sample()


def load_agent_workload(
    path: Path,
    *,
    run_id: str,
    knowledge_query: str,
    preference: str,
) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "1.0"
        or not isinstance(value.get("prompt_template"), str)
        or not isinstance(value.get("required_tools"), list)
    ):
        raise ValueError("invalid ProtBind Agent benchmark workload")
    prompt = value["prompt_template"].format(
        run_id=run_id,
        knowledge_query=knowledge_query,
        preference=preference,
    )
    return value, prompt, _sha256(raw)


def run_agent_benchmark(
    *,
    workspace: Path,
    project_root: Path,
    workload_path: Path,
    run_id: str,
    knowledge_query: str,
    preference: str,
    knowledge_model: Path,
    model_weights: Path,
    hipfire_source_root: Path,
    hipfire_daemon: Path,
    library_config: Path | None,
    pipeline_config: PipelineConfig | None,
    config: AgentBenchmarkConfig,
    base_url: str = "http://127.0.0.1:11435/v1",
) -> dict[str, Any]:
    require_loopback_hipfire_url(base_url)
    if "deepseek" in base_url.lower():
        raise ValueError("DeepSeek is forbidden for Radeon Agent performance evidence")
    workload, prompt, prompt_suite_sha256 = load_agent_workload(
        workload_path,
        run_id=run_id,
        knowledge_query=knowledge_query,
        preference=preference,
    )
    model_weights_receipt = _model_weights_receipt(
        model_weights,
        config.model_sha256,
    )
    code_receipt = _git_receipt(project_root, config.code_revision)
    hipfire_source_receipt = _git_receipt(
        hipfire_source_root,
        config.hipfire_revision,
    )
    hardware = probe_hardware()
    if hardware.hsa_override_active:
        raise ValueError("HSA_OVERRIDE_GFX_VERSION invalidates Radeon evidence")
    if "gfx1100" not in hardware.architectures:
        raise ValueError("Agent benchmark requires an admitted gfx1100 Radeon")
    backend = HipFireBackend(base_url)
    health = backend.health()
    process_receipt = _service_process_receipt(
        health,
        daemon_path=hipfire_daemon,
        config=config,
    )
    advertised = backend.list_models()
    model_stem = config.model.replace(":", "-")
    admitted_model_ids = {
        name for name in advertised if name == config.model or name.startswith(f"{model_stem}.")
    }
    if advertised and not admitted_model_ids:
        raise ValueError(
            "requested model is not represented by the local HipFire model inventory"
        )

    def one_run() -> Any:
        runtime = create_runtime(
            workspace=workspace,
            project_root=project_root,
            model=config.model,
            base_url=base_url,
            library_config=library_config,
            knowledge_model=knowledge_model,
            pipeline_config=pipeline_config,
            confirmation=lambda _preview: True,
            backend=backend,
            max_steps=16,
            stream=True,
            route_tools=config.tool_routing,
        )
        return runtime.run(prompt)

    for _ in range(config.warmup_runs):
        one_run()

    results = []
    with _VRAMSampler() as sampler:
        for repetition in range(config.repetitions):
            started = time.perf_counter()
            try:
                result = one_run()
            except AgentLimitError as exc:
                wall_seconds = time.perf_counter() - started
                trace = [
                    name for model_call in exc.tool_call_trace for name in model_call
                ]
                results.append(
                    {
                        "repetition": repetition,
                        "status": "FAILED_AGENT_LIMIT",
                        "wall_seconds": wall_seconds,
                        "model_calls": len(exc.tool_call_trace),
                        "model_timings": (),
                        "model_usages": (),
                        "first_model_ttft_seconds": None,
                        "final_model_ttft_seconds": None,
                        "completion_tokens": None,
                        "end_to_end_model_tokens_per_second": None,
                        "post_first_model_tokens_per_second": None,
                        "tool_calls": trace,
                        "exposed_tool_names": (),
                        "exposed_tool_schema_bytes": (),
                        "tool_success_rate": 0.0,
                        "required_tool_sequence_passed": False,
                        "artifact_citation_present": False,
                        "answer_sha256": None,
                        "answer_chars": 0,
                        "failure_type": type(exc).__name__,
                    }
                )
                continue
            wall_seconds = time.perf_counter() - started
            events = [
                {"tool": item["name"], "ok": item["ok"]}
                for item in result.tool_results
            ]
            called = [str(event["tool"]) for event in events]
            required = [str(name) for name in workload["required_tools"]]
            required_success = _successful_subsequence(events, required)
            tool_success_rate = (
                sum(bool(event["ok"]) for event in events) / len(events)
                if events
                else 0.0
            )
            model_seconds = sum(
                float(item["total_seconds"]) for item in result.model_timings
            )
            completion_tokens = result.usage["completion_tokens"]
            end_to_end_tokens_per_second = (
                float(completion_tokens) / model_seconds
                if completion_tokens is not None and model_seconds > 0
                else None
            )
            post_first_seconds = 0.0
            post_first_tokens = 0
            post_first_complete = True
            for timing, usage in zip(
                result.model_timings,
                result.model_usages,
                strict=True,
            ):
                ttft = timing["time_to_first_token_seconds"]
                tokens = usage["completion_tokens"]
                if ttft is None or tokens is None:
                    post_first_complete = False
                    break
                streamed_span = max(
                    float(timing["total_seconds"]) - float(ttft), 0.0
                )
                if int(tokens) > 1 and streamed_span < 0.01:
                    post_first_complete = False
                    break
                post_first_seconds += streamed_span
                post_first_tokens += max(int(tokens) - 1, 0)
            post_first_tokens_per_second = (
                post_first_tokens / post_first_seconds
                if post_first_complete and post_first_seconds >= 0.01
                else None
            )
            results.append(
                {
                    "repetition": repetition,
                    "status": "COMPLETED",
                    "wall_seconds": wall_seconds,
                    "model_calls": result.model_calls,
                    "model_timings": result.model_timings,
                    "model_usages": result.model_usages,
                    "first_model_ttft_seconds": (
                        result.model_timings[0]["time_to_first_token_seconds"]
                        if result.model_timings
                        else None
                    ),
                    "final_model_ttft_seconds": (
                        result.model_timings[-1]["time_to_first_token_seconds"]
                        if result.model_timings
                        else None
                    ),
                    "completion_tokens": completion_tokens,
                    "end_to_end_model_tokens_per_second": (
                        end_to_end_tokens_per_second
                    ),
                    "post_first_model_tokens_per_second": (
                        post_first_tokens_per_second
                    ),
                    "tool_calls": called,
                    "exposed_tool_names": result.exposed_tool_names,
                    "exposed_tool_schema_bytes": result.exposed_tool_schema_bytes,
                    "tool_routes": result.tool_routes,
                    "tool_success_rate": tool_success_rate,
                    "tool_timeline": result.tool_timeline,
                    "required_tool_sequence_passed": required_success,
                    "artifact_citation_present": bool(
                        result.validated_artifact_citations
                    ),
                    "validated_artifact_citations": (
                        result.validated_artifact_citations
                    ),
                    "citation_warnings": result.citation_warnings,
                    "answer_sha256": _sha256(result.answer.encode("utf-8")),
                    "answer_chars": len(result.answer),
                }
            )
    required_pass_rate = sum(
        bool(item["required_tool_sequence_passed"]) for item in results
    ) / len(results)
    citation_pass_rate = sum(
        bool(item["artifact_citation_present"]) for item in results
    ) / len(results)
    tool_success_values = [float(item["tool_success_rate"]) for item in results]
    total_values = [float(item["wall_seconds"]) for item in results]
    ttft_values = [
        float(item["first_model_ttft_seconds"])
        for item in results
        if item["first_model_ttft_seconds"] is not None
    ]
    throughput_values = [
        float(item["end_to_end_model_tokens_per_second"])
        for item in results
        if item["end_to_end_model_tokens_per_second"] is not None
    ]
    post_first_throughput_values = [
        float(item["post_first_model_tokens_per_second"])
        for item in results
        if item["post_first_model_tokens_per_second"] is not None
    ]
    first_schema_byte_values = [
        float(item["exposed_tool_schema_bytes"][0])
        for item in results
        if item["exposed_tool_schema_bytes"]
    ]
    first_tool_count_values = [
        float(len(item["exposed_tool_names"][0]))
        for item in results
        if item["exposed_tool_names"]
    ]
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    try:
        import torch

        torch_hip_version = torch.version.hip
    except (ImportError, AttributeError):
        torch_hip_version = None
    evidence_eligible = (
        required_pass_rate == 1.0
        and citation_pass_rate == 1.0
        and min(tool_success_values) == 1.0
        and config.hipfire_visible_device in sampler.peaks
        and code_receipt["clean"] is True
        and hipfire_source_receipt["clean"] is True
        and process_receipt["verified"] is True
    )
    return {
        "schema_version": AGENT_BENCHMARK_SCHEMA_VERSION,
        "kind": "protbind.radeon-agent-tool-benchmark",
        "backend": "hipfire",
        "cloud_result": False,
        "evidence_eligible": evidence_eligible,
        "config": asdict(config),
        "provenance": {
            "model_name": config.model,
            "model_revision": config.model_revision,
            "model_weights_sha256": config.model_sha256,
            "model_weights_receipt": model_weights_receipt,
            "hipfire_revision": config.hipfire_revision,
            "hipfire_source_receipt": hipfire_source_receipt,
            "hipfire_process_receipt": process_receipt,
            "rocm_version": hardware.rocm_version,
            "protbind_python_torch_version": torch_version,
            "protbind_python_torch_hip_version": torch_hip_version,
            "code_revision": config.code_revision,
            "code_receipt": code_receipt,
            "prompt_suite_sha256": prompt_suite_sha256,
            "workload_path_name": workload_path.name,
        },
        "hardware_receipt": hardware.to_dict(),
        "hipfire_health": health,
        "vram": {
            "sampling_scope": "system-wide Radeon used VRAM during measured Agent runs",
            "sample_count": sampler.samples,
            "peak_used_bytes_by_device": {
                str(device): value for device, value in sorted(sampler.peaks.items())
            },
            "selected_device_peak_used_bytes": sampler.peaks.get(
                config.hipfire_visible_device
            ),
        },
        "prompt_request_sha256": _sha256(
            canonical_json_bytes({"prompt": prompt, "model": config.model})
        ),
        "required_tools": workload["required_tools"],
        "samples": results,
        "summary": {
            "required_tool_sequence_pass_rate": required_pass_rate,
            "artifact_citation_pass_rate": citation_pass_rate,
            "tool_success_rate": statistics.fmean(tool_success_values),
            "wall_seconds": _summary(total_values),
            "first_model_ttft_seconds": _summary(ttft_values),
            "end_to_end_model_tokens_per_second": _summary(throughput_values),
            "post_first_model_tokens_per_second": _summary(
                post_first_throughput_values
            ),
            "first_model_exposed_tool_count": _summary(first_tool_count_values),
            "first_model_tool_schema_bytes": _summary(first_schema_byte_values),
        },
        "scientific_semantics": (
            "Measures local Agent inference/tool orchestration only. It is not docking "
            "accuracy, affinity, pose validity, or scientific-kernel performance."
        ),
        "throughput_semantics": (
            "End-to-end model tokens/s divides reported completion tokens by complete "
            "model-call time. Post-first tokens/s is omitted when HipFire emits a "
            "response in one effectively atomic stream span (<0.01 s after TTFT)."
        ),
    }


def save_agent_benchmark(value: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
