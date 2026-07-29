from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from protbind_agent.artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from protbind_agent.redock_benchmark import (
    RedockBenchmarkConfig,
    _protbind_code_receipt,
)
from protbind_agent.redock_holdout_batch import (
    RedockHoldoutBatchConfig,
    RedockHoldoutBatchError,
    run_frozen_redock_holdout,
)


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fixture(repo: Path):
    store = ArtifactStore(repo / "holdout-artifacts")
    selected = []
    for index in range(10):
        case_id = f"7A{index:02d}_L{index}"
        receptor = store.put_bytes(
            f"receptor-{case_id}".encode(),
            media_type="chemical/x-pdb",
            producer="fixture",
            producer_version="1",
            source=f"dataset:fixture/{case_id}/protein.pdb",
            license="CC0-1.0",
        )
        ligand = store.put_bytes(
            f"ligand-{case_id}".encode(),
            media_type="chemical/x-mdl-sdfile",
            producer="fixture",
            producer_version="1",
            source=f"dataset:fixture/{case_id}/ligand.sdf",
            license="CC0-1.0",
        )
        source = store.put_bytes(
            canonical_json_bytes({"case_id": case_id}),
            media_type="application/json",
            producer="fixture",
            producer_version="1",
            source=f"dataset:fixture/{case_id}",
            license="CC0-1.0",
        )
        selected.append(
            {
                "complex_id": case_id,
                "ligand_instance_id": f"L{index}",
                "source_complex": source.to_dict(),
                "receptor": receptor.to_dict(),
                "native_ligand": ligand.to_dict(),
                "license": "CC0-1.0",
                "protein_chain_count": 1,
                "protein_residue_count": 100,
                "ligand_count": 1,
                "ligand_heavy_atom_count": 10,
                "is_non_covalent": True,
                "ordinary_nonpolymer_ligand": True,
                "contains_metal": False,
                "requires_cofactor": False,
                "pocket_altloc_ambiguous": False,
                "missing_pocket_heavy_atoms": False,
                "receptor_model_count": 1,
                "contains_nonstandard_protein_residue": False,
                "missing_backbone_atoms": False,
                "ligand_unspecified_stereo": False,
            }
        )
    source_archive = store.put_bytes(
        b"archive",
        media_type="application/zip",
        producer="fixture",
        source="https://example.test/archive",
        license="CC0-1.0",
    )
    candidate_list = store.put_bytes(
        b"candidate-list",
        media_type="text/plain",
        producer="fixture",
        source="https://example.test/list",
        license="CC0-1.0",
    )
    body = {
        "schema_version": "1.1",
        "dataset_name": "fixture",
        "dataset_version": "1",
        "dataset_license": "CC0-1.0",
        "dataset_source": source_archive.to_dict(),
        "candidate_list": candidate_list.to_dict(),
        "eligibility_policy": {
            "version": "fixture-1",
            "source_sha256": "a" * 64,
            "pocket_radius_angstrom": 6.0,
            "heterogen_policy": "exclude",
            "stereo_policy": "exclude",
            "selection_reads_docking_results": False,
        },
        "namespace": "fixture-result-blind",
        "requested_count": 10,
        "selection_rule": "fixture hash order",
        "selected": selected,
        "exclusions": [],
    }
    holdout = {**body, "selection_hash": sha256_bytes(canonical_json_bytes(body))}
    holdout_path = repo / "holdout.json"
    holdout_path.write_text(json.dumps(holdout, indent=2, sort_keys=True) + "\n")

    tools_directory = repo / "tools"
    tools_directory.mkdir()
    tools = {
        "vina": _executable(tools_directory / "vina"),
        "mk_prepare_receptor": _executable(
            tools_directory / "mk_prepare_receptor.py"
        ),
        "mk_prepare_ligand": _executable(tools_directory / "mk_prepare_ligand.py"),
        "mk_export": _executable(tools_directory / "mk_export.py"),
    }
    redock = RedockBenchmarkConfig(
        vina=tools["vina"],
        mk_prepare_receptor=tools["mk_prepare_receptor"],
        mk_prepare_ligand=tools["mk_prepare_ligand"],
        mk_export=tools["mk_export"],
    )
    return store, holdout_path, tools, redock


def _fake_runner(tools, calls: list[str]):
    def run(receptor: Path, native: Path, output: Path, *, config):
        output.mkdir(parents=True)
        case_id = config.receptor_source.split("/")[-2]
        calls.append(case_id)
        code = _protbind_code_receipt()
        toolchain = {
            name: {
                "executable": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in tools.items()
        }
        python = Path(sys.executable).resolve()
        toolchain["python"] = {
            "executable": python.name,
            "sha256": sha256_file(python),
        }
        result = {
            "schema_version": "1.2",
            "benchmark": "redock",
            "status": "COMPLETED",
            "scientific_status": "NOT_RECOVERED_TOP5",
            "failure": None,
            "code_sha256": code["manifest_sha256"],
            "config": {
                "seed": config.seed,
                "padding_angstrom": config.padding_angstrom,
                "exhaustiveness": config.exhaustiveness,
                "num_modes": config.num_modes,
                "energy_range": config.energy_range,
                "cpu": config.cpu,
                "vina_scoring": "vina",
                "receptor_source": config.receptor_source,
                "native_ligand_source": config.native_ligand_source,
                "input_license": config.input_license,
                "conservative_receptor_repair": config.conservative_receptor_repair,
                "repair_protected_radius_angstrom": (
                    config.repair_protected_radius_angstrom
                ),
                "restrained_sidechain_optimization": (
                    config.restrained_sidechain_optimization
                ),
                "sidechain_optimization_iteration_limits": list(
                    config.sidechain_optimization_iteration_limits
                ),
                "screening_calibration": None,
            },
            "input_commitments": {
                "receptor": {
                    "sha256": sha256_file(receptor),
                    "size_bytes": receptor.stat().st_size,
                    "media_type": "chemical/x-pdb",
                    "source": config.receptor_source,
                    "license": config.input_license,
                },
                "native_ligand": {
                    "sha256": sha256_file(native),
                    "size_bytes": native.stat().st_size,
                    "media_type": "chemical/x-mdl-sdfile",
                    "source": config.native_ligand_source,
                    "license": config.input_license,
                },
            },
            "toolchain": toolchain,
            "top1_recovered": False,
            "top5_recovered": False,
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result

    return run


def test_fixed_holdout_runs_all_cases_emits_manifest_and_resumes(tmp_path: Path):
    store, holdout, tools, redock = _fixture(tmp_path)
    calls: list[str] = []
    output = tmp_path / "formal-run"
    config = RedockHoldoutBatchConfig(redock=redock, max_parallel_cases=1)
    first = run_frozen_redock_holdout(
        tmp_path,
        holdout,
        store,
        output,
        config=config,
        runner=_fake_runner(tools, calls),
    )

    assert len(calls) == 10
    assert first["terminal_count"] == 10
    assert first["completed_count"] == 10
    assert first["failed_count"] == 0
    manifest = json.loads((output / "regression-manifest.json").read_text())
    frozen = json.loads(holdout.read_text())
    assert manifest["evaluation_design"] == "FROZEN_HOLDOUT"
    assert [case["case_id"] for case in manifest["cases"]] == [
        case["complex_id"] for case in frozen["selected"]
    ]

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("valid terminal cases must be resumed")

    second = run_frozen_redock_holdout(
        tmp_path,
        holdout,
        store,
        output,
        config=config,
        runner=must_not_run,
    )
    assert all(case["resumed"] is True for case in second["cases"])


def test_fixed_holdout_rejects_tampered_resume_and_plan_change(tmp_path: Path):
    store, holdout, tools, redock = _fixture(tmp_path)
    output = tmp_path / "formal-run"
    config = RedockHoldoutBatchConfig(redock=redock, max_parallel_cases=1)
    run_frozen_redock_holdout(
        tmp_path,
        holdout,
        store,
        output,
        config=config,
        runner=_fake_runner(tools, []),
    )
    first = output / "cases" / "7a00_l0" / "result.json"
    value = json.loads(first.read_text())
    value["code_sha256"] = "0" * 64
    first.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(RedockHoldoutBatchError, match="code binding"):
        run_frozen_redock_holdout(
            tmp_path,
            holdout,
            store,
            output,
            config=config,
            runner=_fake_runner(tools, []),
        )

    another_output = tmp_path / "plan-change"
    run_frozen_redock_holdout(
        tmp_path,
        holdout,
        store,
        another_output,
        config=config,
        runner=_fake_runner(tools, []),
    )
    changed = RedockHoldoutBatchConfig(
        redock=RedockBenchmarkConfig(
            seed=redock.seed + 1,
            vina=redock.vina,
            mk_prepare_receptor=redock.mk_prepare_receptor,
            mk_prepare_ligand=redock.mk_prepare_ligand,
            mk_export=redock.mk_export,
        ),
        max_parallel_cases=1,
    )
    with pytest.raises(RedockHoldoutBatchError, match="run-plan.json differs"):
        run_frozen_redock_holdout(
            tmp_path,
            holdout,
            store,
            another_output,
            config=changed,
            runner=_fake_runner(tools, []),
        )


def test_repair_holdout_requires_and_freezes_named_protocol_revision(tmp_path: Path):
    store, holdout, tools, redock = _fixture(tmp_path)
    repaired = replace(
        redock,
        conservative_receptor_repair=True,
        repair_protected_radius_angstrom=6.0,
    )
    with pytest.raises(ValueError, match="explicit protocol_revision"):
        RedockHoldoutBatchConfig(redock=repaired, max_parallel_cases=1)

    output = tmp_path / "repair-protocol-v1"
    config = RedockHoldoutBatchConfig(
        redock=repaired,
        max_parallel_cases=1,
        protocol_revision="repair-protocol-v1",
    )
    result = run_frozen_redock_holdout(
        tmp_path,
        holdout,
        store,
        output,
        config=config,
        runner=_fake_runner(tools, []),
    )
    plan = json.loads((output / "run-plan.json").read_text())
    assert plan["protocol_revision"] == "repair-protocol-v1"
    assert plan["redock_config"]["conservative_receptor_repair"] is True
    assert plan["redock_config"]["repair_protected_radius_angstrom"] == 6.0
    assert result["protocol_revision"] == "repair-protocol-v1"

    changed = replace(config, protocol_revision="repair-protocol-v2")
    with pytest.raises(RedockHoldoutBatchError, match="run-plan.json differs"):
        run_frozen_redock_holdout(
            tmp_path,
            holdout,
            store,
            output,
            config=changed,
            runner=_fake_runner(tools, []),
        )
