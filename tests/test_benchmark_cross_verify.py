from __future__ import annotations

import copy

from radeon_agent.backends import MockBackend
from radeon_agent.benchmark import BenchmarkConfig, load_suite, run_benchmark
from radeon_agent.cross_verify import compare_results
from radeon_agent.hardware import CommandEvidence, HardwareManifest
from radeon_agent.models import ChatRequest, ChatResponse, StreamTiming, Usage


class StableBenchmarkBackend(MockBackend):
    name = "stable-test"

    def stream_complete(self, request: ChatRequest) -> tuple[ChatResponse, StreamTiming]:
        self.requests.append(request)
        content = "AMD好用；结果 391。def merge(): pass；复杂度 O(n)。" + ("可核验。" * 40)
        return (
            ChatResponse(
                content=content,
                usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
                finish_reason="stop",
                model=request.model,
            ),
            StreamTiming(total_seconds=1.0, time_to_first_token_seconds=0.2),
        )


def _hardware(arch: str) -> HardwareManifest:
    return HardwareManifest(
        captured_at="2026-07-20T00:00:00+00:00",
        host_fingerprint="test",
        platform="linux",
        python_version="3.12",
        rocm_version="7.2.1",
        device_architectures=(arch,),
        architectures=(arch,),
        competition_roles=("test",),
        hsa_override_active=False,
        evidence=(CommandEvidence(("rocminfo",), True, 0, "abc", (arch,), None),),
    )


def test_benchmark_binds_suite_and_requests(tmp_path) -> None:
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        '{"id":"one","messages":[{"role":"user","content":"say AMD"}],'
        '"max_tokens":8,"checks":{"contains":["AMD"]}}\n',
        encoding="utf-8",
    )
    suite = load_suite(suite_path)
    result = run_benchmark(
        StableBenchmarkBackend(),
        suite,
        BenchmarkConfig(
            label="w7900",
            model="demo",
            repetitions=3,
            quantization="fp16",
            model_revision="fixture-v1",
            model_sha256="a" * 64,
            code_revision="commit",
            runtime_revision="runtime",
            workload_config_sha256="b" * 64,
        ),
        hardware=_hardware("gfx1100"),
    )

    assert result["suite"]["raw_sha256"]
    assert result["suite"]["raw_md5"]
    assert len(result["samples"]) == 3
    assert len({sample["request_sha256"] for sample in result["samples"]}) == 1
    assert result["summary"]["quality_pass_rate"] == 1.0


def test_cross_verifier_accepts_same_workload_on_two_architectures(tmp_path) -> None:
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        '{"id":"one","messages":[{"role":"user","content":"say AMD"}],'
        '"checks":{"contains":["AMD"]}}\n',
        encoding="utf-8",
    )
    suite = load_suite(suite_path)
    config = BenchmarkConfig(
        label="machine",
        model="demo",
        repetitions=3,
        quantization="fp16",
        model_revision="fixture-v1",
        model_sha256="a" * 64,
        code_revision="commit",
        runtime_revision="runtime",
        workload_config_sha256="b" * 64,
    )
    primary = run_benchmark(
        StableBenchmarkBackend(), suite, config, hardware=_hardware("gfx1100")
    )
    verifier = copy.deepcopy(primary)
    verifier["label"] = "r9700"
    verifier["hardware"] = _hardware("gfx1201").to_dict()

    report = compare_results(primary, verifier)

    assert report.compatible
    assert report.errors == ()
    assert report.metrics["exact_output_match_rate"] == 1.0


def test_cross_verifier_rejects_request_mismatch(tmp_path) -> None:
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        '{"id":"one","messages":[{"role":"user","content":"say AMD"}]}\n',
        encoding="utf-8",
    )
    result = run_benchmark(
        StableBenchmarkBackend(),
        load_suite(suite_path),
        BenchmarkConfig(
            label="machine",
            model="demo",
            repetitions=3,
            quantization="fp16",
            model_revision="fixture-v1",
            model_sha256="a" * 64,
            code_revision="commit",
            runtime_revision="runtime",
            workload_config_sha256="b" * 64,
        ),
        hardware=_hardware("gfx1100"),
    )
    verifier = copy.deepcopy(result)
    verifier["hardware"] = _hardware("gfx1201").to_dict()
    verifier["samples"][0]["request_sha256"] = "different"

    report = compare_results(result, verifier)

    assert not report.compatible
    assert "serialized request hashes differ" in report.errors


def test_cross_verifier_rejects_cloud_deepseek_result(tmp_path) -> None:
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        '{"id":"one","messages":[{"role":"user","content":"say AMD"}]}\n',
        encoding="utf-8",
    )
    primary = run_benchmark(
        StableBenchmarkBackend(),
        load_suite(suite_path),
        BenchmarkConfig(
            label="machine",
            model="demo",
            repetitions=3,
            quantization="fp16",
            model_revision="fixture-v1",
            model_sha256="a" * 64,
            code_revision="commit",
            runtime_revision="runtime",
            workload_config_sha256="b" * 64,
        ),
        hardware=_hardware("gfx1100"),
    )
    verifier = copy.deepcopy(primary)
    primary["backend"] = "deepseek"
    verifier["backend"] = "deepseek"
    verifier["hardware"] = _hardware("gfx1201").to_dict()

    report = compare_results(primary, verifier)

    assert not report.compatible
    assert "cloud DeepSeek results cannot prove local Radeon execution" in report.errors
