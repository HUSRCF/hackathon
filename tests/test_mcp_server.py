from __future__ import annotations

import json

import pytest

from protbind_agent.mcp_server import ProtBindMCPService, create_mcp_server
from protbind_agent.tripharm import build_jsonl_index


def _project_inputs(project_root) -> None:
    (project_root / "query.json").write_text(
        json.dumps(
            {
                "features": [
                    {"type": "Donor", "position": [0, 0, 0], "atom_indices": [0]},
                    {"type": "Acceptor", "position": [3, 0, 0], "atom_indices": [1]},
                    {"type": "Aromatic", "position": [0, 4, 0], "atom_indices": [2]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (project_root / "receptor.pdb").write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      3  C   ALA A   1       2.050   1.400   0.000  1.00 20.00           C  \n"
        "ATOM      4  O   ALA A   1       1.400   2.400   0.000  1.00 20.00           O  \n"
        "END\n",
        encoding="utf-8",
    )
    (project_root / "case.json").write_text(
        json.dumps(
            {
                "case_id": "mcp-case",
                "target": {
                    "name": "target",
                    "structure_file": "receptor.pdb",
                },
                "mode": "ligand_only",
                "ligand": {"pharmacophore_file": "query.json"},
            }
        ),
        encoding="utf-8",
    )
    source = project_root / "library.jsonl"
    source.write_text(
        json.dumps(
            {
                "molecule_id": "mol-a",
                "smiles": "CCO",
                "conformers": [
                    {
                        "id": 0,
                        "features": [
                            {
                                "type": "Donor",
                                "position": [1, 0, 0],
                                "atom_indices": [0],
                            },
                            {
                                "type": "Acceptor",
                                "position": [4, 0, 0],
                                "atom_indices": [1],
                            },
                            {
                                "type": "Aromatic",
                                "position": [1, 4, 0],
                                "atom_indices": [2],
                            },
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    build_jsonl_index(source, project_root / "library.sqlite")


def test_mcp_service_creates_offline_case_and_returns_stage_gate(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _project_inputs(project_root)
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=project_root,
    )

    result = service.case_create(
        case_path="case.json",
        index_path="library.sqlite",
        run_id="mcp-run",
    )

    assert result["run_id"] == "mcp-run"
    assert result["stage_gate"]["gate"]["decision"] == "READY"
    assert result["stage_gate"]["gate"]["stage"] == "INPUT_VALIDATED"


@pytest.mark.parametrize("value", ("../outside.json", "/tmp/outside.json"))
def test_mcp_service_rejects_paths_outside_project_root(tmp_path, value) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="project-relative|escapes"):
        service.case_create(case_path=value, index_path="missing.sqlite")


def test_mcp_public_fetch_rejects_output_before_network(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=project_root,
    )

    with pytest.raises(ValueError, match=r"must use the \.cif suffix"):
        service.fetch_public_data(
            source="rcsb-mmcif",
            identifier="1CRN",
            project_path="inputs/receptor.pdb",
            approved_domain="files.rcsb.org",
        )


def test_mcp_library_reads_require_explicit_data_confirmation(tmp_path) -> None:
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )

    with pytest.raises(PermissionError, match="fresh explicit user confirmation"):
        service.library_status(data_access_confirmed=False)

    status = service.library_status(data_access_confirmed=True)
    assert status["configured"] is False
    assert status["absolute_paths_disclosed"] is False


def test_mcp_server_exposes_only_bounded_domain_tools(tmp_path) -> None:
    pytest.importorskip("mcp")
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )
    server = create_mcp_server(service)

    tool_names = set(server._tool_manager._tools)  # noqa: SLF001

    assert tool_names == {
        "doctor",
        "fetch_public_data",
        "case_create",
        "case_status",
        "case_advance",
        "case_attach_support",
        "case_report",
        "case_dossier",
        "case_pose_view",
        "artifact_metadata",
        "control_history",
        "library_status",
        "library_list",
        "library_show",
        "library_plan_import",
        "library_apply_import",
        "library_verify_uniprot",
        "knowledge_document_inspect",
        "knowledge_import",
        "knowledge_search",
        "library_rag_sync",
        "library_rag_search",
        "knowledge_model_status",
    }
    assert not tool_names & {"bash", "shell", "read_file", "write_file", "fetch", "network"}


def test_mcp_document_inspection_requires_fresh_data_confirmation(tmp_path) -> None:
    document = tmp_path / "paper.md"
    document.write_text("# Result\nEvidence.\n", encoding="utf-8")
    service = ProtBindMCPService(
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
    )

    with pytest.raises(PermissionError, match="fresh explicit user confirmation"):
        service.knowledge_document_inspect(
            project_path="paper.md",
            data_access_confirmed=False,
        )

    result = service.knowledge_document_inspect(
        project_path="paper.md",
        data_access_confirmed=True,
    )
    assert result["chunk_count"] == 1
    assert result["text_returned"] is False
