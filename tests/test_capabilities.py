from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

import protbind_agent.capabilities as capabilities
from protbind_agent.capabilities import (
    _bubblewrap_network_isolation_preflight,
    _openfold_resource_policy,
    _prediction_fallback_policy,
    doctor_report,
)


@pytest.mark.parametrize(
    ("device_count", "openfold_device", "other_tool_devices", "single_gpu_pause"),
    (
        (0, None, [], False),
        (1, "0", [], True),
        (2, "0", ["1"], False),
        (4, "0", ["1", "2", "3"], False),
    ),
)
def test_openfold_resource_policy_preserves_other_gpu_capacity(
    device_count: int,
    openfold_device: str | None,
    other_tool_devices: list[str],
    single_gpu_pause: bool,
) -> None:
    policy = _openfold_resource_policy(device_count)

    assert policy["openfold_visible_device_default"] == openfold_device
    assert policy["reserved_other_tool_devices"] == other_tool_devices
    assert policy["openfold_devices_per_job"] == 1
    assert policy["max_concurrent_openfold_jobs_default"] == 1
    assert policy["lease_scope"] == (
        "same-user ProtBind workers across host workspaces"
    )
    assert policy["external_services_coordinated"] is False
    assert bool(policy["single_gpu_policy"]) is single_gpu_pause
    assert policy["checkpoint_policy"] == {
        "allowed": ["openfold3-p2-155k"],
        "size_bytes": {"openfold3-p2-155k": 2_287_928_196},
        "has_small_memory_variant": False,
    }


def test_prediction_fallback_policy_never_calls_legacy_esmfold_a_cofolder() -> None:
    policy = _prediction_fallback_policy()

    assert policy["receptor_precedence"][-1] == "legacy_esmfold_v1"
    assert policy["legacy_esmfold_v1_scope"] == "receptor_only_not_ligand_pose"
    assert "openfold3_after_checkpoint_gate" in policy[
        "complex_predictor_precedence"
    ]
    assert "esmfold2_after_three_complex_gate" in policy[
        "complex_predictor_precedence"
    ]
    assert "do not claim cofolding" in policy["no_complex_predictor"]


def test_bubblewrap_preflight_missing_is_not_replaced_by_offline_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(capabilities.shutil, "which", lambda name: None)

    result = _bubblewrap_network_isolation_preflight()

    assert result["status"] == "missing"
    assert result["executable_present"] is False
    assert result["probe_performed"] is False
    assert result["probe_return_code"] is None
    assert result["os_network_isolation_usable"] is False
    assert result["application_offline_env_is_os_isolation"] is False


def test_bubblewrap_preflight_uses_bounded_data_free_namespace_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(capabilities.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(capabilities.subprocess, "run", fake_run)

    result = _bubblewrap_network_isolation_preflight()

    assert result["status"] == "usable"
    assert result["probe_return_code"] == 0
    assert result["os_network_isolation_usable"] is True
    command = observed["command"]
    assert isinstance(command, tuple)
    assert command == (
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--chdir",
        "/",
        "--",
        "/bin/true",
    )
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["timeout"] == 2.0
    assert kwargs["env"] == {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    assert "input" not in kwargs


def test_bubblewrap_preflight_reports_sanitized_unusable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b"bwrap: /home/alice/private/input.sdf TOKEN=topsecret\n",
        )

    monkeypatch.setattr(capabilities.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(capabilities.subprocess, "run", fake_run)

    result = _bubblewrap_network_isolation_preflight()

    assert result["status"] == "present_but_unusable"
    assert result["executable_present"] is True
    assert result["probe_performed"] is True
    assert result["probe_return_code"] == 1
    assert result["os_network_isolation_usable"] is False
    assert "[INTERNAL_PATH]" in result["reason"]
    assert "topsecret" not in result["reason"]
    assert "/home/alice" not in result["reason"]
    assert len(result["reason"]) <= 512


def test_bubblewrap_preflight_timeout_is_fail_closed_and_does_not_echo_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: tuple[str, ...], **kwargs: object) -> None:
        del kwargs
        raise subprocess.TimeoutExpired(
            cmd=(*command, "/home/alice/private-input"),
            timeout=2.0,
        )

    monkeypatch.setattr(capabilities.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(capabilities.subprocess, "run", fake_run)

    result = _bubblewrap_network_isolation_preflight()

    assert result["status"] == "present_but_unusable"
    assert result["probe_return_code"] is None
    assert result["os_network_isolation_usable"] is False
    assert result["reason"] == "bubblewrap namespace probe exceeded its 2-second timeout"
    assert "alice" not in result["reason"]


def test_bubblewrap_preflight_malformed_monkeypatch_result_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capabilities.shutil, "which", lambda name: "/mock/bwrap")
    monkeypatch.setattr(
        capabilities.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode="0"),
    )

    result = _bubblewrap_network_isolation_preflight()

    assert result["status"] == "present_but_unusable"
    assert result["probe_return_code"] is None
    assert result["os_network_isolation_usable"] is False
    assert result["reason"] == "bubblewrap namespace probe returned an invalid status"


def test_doctor_report_exposes_network_isolation_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "mechanism": "bubblewrap-unshare-net",
        "status": "present_but_unusable",
        "os_network_isolation_usable": False,
        "application_offline_env_is_os_isolation": False,
    }

    class _Hardware:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"device_architectures": []}

    monkeypatch.setattr(capabilities, "probe_hardware", _Hardware)
    monkeypatch.setattr(capabilities, "discover_capabilities", lambda: [])
    monkeypatch.setattr(
        capabilities,
        "_bubblewrap_network_isolation_preflight",
        lambda: expected,
    )
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda name: None)

    report = doctor_report()

    assert report["offline_default"] is True
    assert report["runtime_details"]["worker_network_isolation"] == expected
    assert report["runtime_details"]["worker_network_isolation"][
        "os_network_isolation_usable"
    ] is False


def test_discovery_recognizes_bundled_vina_when_not_on_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "vina"
    bundled.write_text("fixture", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: None)
    monkeypatch.setattr(capabilities, "_BUNDLED_EXECUTABLES", {"vina": bundled})

    discovered = {item.name: item for item in capabilities.discover_capabilities()}

    assert discovered["vina"].available is True
    assert discovered["vina"].version == "bundled-executable-found"
    assert discovered["fpocket"].available is False
