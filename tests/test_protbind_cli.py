from __future__ import annotations

from pathlib import Path

import pytest

from protbind_agent.cli import _build_parser, _worker_config
from protbind_agent.manifest import RunState


def _worker_table(name: str) -> str:
    return (
        f"[workers.{name}]\n"
        'engine = "vina-quick"\n'
        'argv = ["python", "workers/quick_vina_worker.py"]\n'
        f"[workers.{name}.provenance]\n"
        'model_revision = "selection-quick-vina-1.0+fixture"\n'
        f'weight_sha256 = "{"a" * 64}"\n'
        f'code_sha256 = "{"b" * 64}"\n'
    )


@pytest.mark.parametrize("name", ("select", "selected", "quick_vina"))
def test_selected_worker_aliases_map_to_selected_stage(tmp_path, name) -> None:
    path = tmp_path / "workers.toml"
    path.write_text(_worker_table(name), encoding="utf-8")

    config = _worker_config(path)

    assert set(config.workers) == {RunState.SELECTED}
    assert config.workers[RunState.SELECTED].engine == "vina-quick"


def test_duplicate_selected_worker_aliases_are_rejected(tmp_path) -> None:
    path = tmp_path / "workers.toml"
    path.write_text(
        _worker_table("select") + "\n" + _worker_table("quick_vina"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple worker sections configure SELECTED"):
        _worker_config(path)


@pytest.mark.parametrize(
    "argv",
    (
        (
            "case",
            "run",
            "--case",
            "case.json",
            "--index",
            "index.sqlite",
            "--vina-environment-lock",
            "aiaa-lock.json",
        ),
        (
            "case",
            "resume",
            "run-id",
            "--vina-environment-lock",
            "aiaa-lock.json",
        ),
    ),
)
def test_case_run_and_resume_accept_upfront_vina_environment_lock(argv) -> None:
    arguments = _build_parser().parse_args(argv)

    assert arguments.vina_environment_lock == Path("aiaa-lock.json")


def test_mcp_cli_is_stdio_only_and_project_bounded() -> None:
    arguments = _build_parser().parse_args(
        (
            "mcp",
            "serve",
            "--workspace",
            "private-workspace",
            "--project-root",
            "project",
        )
    )

    assert arguments.transport == "stdio"
    assert arguments.workspace == Path("private-workspace")
    assert arguments.project_root == Path("project")


def test_mcp_cli_accepts_private_library_config_without_arbitrary_roots() -> None:
    arguments = _build_parser().parse_args(
        (
            "mcp",
            "serve",
            "--library-config",
            ".protbind/library.json",
        )
    )

    assert arguments.library_config == Path(".protbind/library.json")


def test_library_cli_requires_explicit_data_access_confirmation() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(("library", "status"))

    arguments = parser.parse_args(
        ("library", "status", "--confirm-data-access")
    )
    assert arguments.library_command == "status"
    assert arguments.confirm_data_access is True


def test_p2rank_parse_requires_version_provenance() -> None:
    arguments = _build_parser().parse_args(
        (
            "site",
            "p2rank-parse",
            "--receptor",
            "target.cif",
            "--predictions",
            "target_predictions.csv",
            "--bundle",
            "sites.json",
            "--p2rank-version",
            "P2Rank 2.5",
        )
    )

    assert arguments.site_command == "p2rank-parse"
    assert arguments.p2rank_version == "P2Rank 2.5"


def test_case_gate_and_advance_parse_closed_loop_arguments() -> None:
    gate = _build_parser().parse_args(("case", "gate", "run-1"))
    advance = _build_parser().parse_args(
        (
            "case",
            "advance",
            "run-1",
            "--continuation-token",
            "a" * 64,
        )
    )

    assert gate.case_command == "gate"
    assert advance.case_command == "advance"
    assert advance.continuation_token == "a" * 64


def test_case_dossier_and_pose_view_commands_are_read_only_inspection_surfaces() -> None:
    dossier = _build_parser().parse_args(
        ("case", "dossier", "run-1", "--format", "json")
    )
    poses = _build_parser().parse_args(("case", "poses", "run-1"))

    assert dossier.case_command == "dossier"
    assert dossier.format == "json"
    assert poses.case_command == "poses"


def test_3dmol_asset_installer_requires_explicit_source_mode() -> None:
    arguments = _build_parser().parse_args(
        (
            "assets",
            "install-3dmol",
            "--approve-network",
            "cdn.jsdelivr.net",
            "--workspace",
            "private-workspace",
        )
    )

    assert arguments.assets_command == "install-3dmol"
    assert arguments.approve_network == ["cdn.jsdelivr.net"]
    assert arguments.workspace == Path("private-workspace")


def test_public_data_fetch_parses_identifier_only_network_contract() -> None:
    arguments = _build_parser().parse_args(
        (
            "data",
            "fetch",
            "--source",
            "pubchem-cid-sdf-3d",
            "--identifier",
            "2244",
            "--output",
            "inputs/aspirin.sdf",
            "--approve-network",
            "pubchem.ncbi.nlm.nih.gov",
            "--skip-propka",
        )
    )

    assert arguments.data_command == "fetch"
    assert arguments.source == "pubchem-cid-sdf-3d"
    assert arguments.output == Path("inputs/aspirin.sdf")
    assert arguments.approve_network == ["pubchem.ncbi.nlm.nih.gov"]
    assert arguments.skip_propka is True
