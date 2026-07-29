from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

import protbind_agent.cli as cli_module
import protbind_agent.redock_benchmark as benchmark_module
from protbind_agent.artifacts import sha256_file
from protbind_agent.cli import _build_parser
from protbind_agent.preparation import (
    ConservativeRepairResult,
    RestrainedSidechainOptimizationResult,
)
from protbind_agent.redock_benchmark import (
    RedockBenchmarkConfig,
    _retryable_rdkit_receptor_failure,
    run_redock_benchmark,
)

Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")


def _atom_line(
    serial: int,
    atom_name: str,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:>4} ALA A   1    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2}  "
    )


def _receptor(path: Path) -> None:
    lines = [
        _atom_line(1, "N", (-2.0, 0.0, 0.0), "N"),
        _atom_line(2, "CA", (-1.0, 0.0, 0.0), "C"),
        _atom_line(3, "C", (0.0, 0.0, 0.0), "C"),
        _atom_line(4, "O", (1.0, 0.0, 0.0), "O"),
        _atom_line(5, "CB", (-1.0, 1.0, 0.0), "C"),
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _native_and_modes(native_path: Path) -> str:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 81
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    molecule.SetProp("_Name", "native")
    writer = Chem.SDWriter(str(native_path))
    writer.write(molecule)
    writer.close()

    handle = io.StringIO()
    writer = Chem.SDWriter(handle)
    for mode, shift in ((1, 0.0), (2, 0.75)):
        pose = Chem.Mol(molecule)
        pose.SetProp("_Name", f"mode-{mode}")
        conformer = pose.GetConformer()
        for index in range(pose.GetNumAtoms()):
            point = conformer.GetAtomPosition(index)
            conformer.SetAtomPosition(index, (point.x, point.y + shift, point.z))
        writer.write(pose)
    writer.close()
    return handle.getvalue()


def _executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_tools(
    directory: Path,
    exported_sdf: str,
    *,
    vina_exit: int = 0,
    receptor_rdkit_failures: int = 0,
):
    directory.mkdir()
    receptor = _executable(
        directory / "mk_prepare_receptor.py",
        f"""import json, pathlib, shutil, sys
args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
assert not any('native-reference' in path.name for path in pathlib.Path('.').iterdir())
counter = pathlib.Path('receptor-attempt-count.txt')
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
if attempt <= {receptor_rdkit_failures}:
    print('[RDKit] ERROR: Explicit valence for atom # 8 O, 4, is greater than permitted')
    print('rdkit.Chem.rdchem.AtomValenceException')
    raise SystemExit(1)
shutil.copyfile(value('--read_pdb'), value('--write_pdb'))
pathlib.Path(value('-p')).write_text('ATOM      1  C   REC A   1       0.0 0.0 0.0' + chr(10))
pathlib.Path(value('-j')).write_text(json.dumps({{'prepared': True}}))
""",
    )
    ligand = _executable(
        directory / "mk_prepare_ligand.py",
        """import pathlib, sys
args = sys.argv[1:]
records = ['REMARK SMILES CCO', 'ROOT', 'ENDROOT', 'TORSDOF 0', '']
pathlib.Path(args[args.index('-o') + 1]).write_text(chr(10).join(records))
""",
    )
    vina = _executable(
        directory / "vina",
        f"""import pathlib, sys
args = sys.argv[1:]
if args == ['--version']:
    print('AutoDock Vina v1.2.7')
    raise SystemExit(0)
out = pathlib.Path(args[args.index('--out') + 1])
out.write_text('REMARK VINA RESULT: -5.000 0.000 0.000\\nREMARK VINA RESULT: -4.500 1.000 2.000\\n')
raise SystemExit({vina_exit})
""",
    )
    exporter = _executable(
        directory / "mk_export.py",
        f"""import pathlib, sys
args = sys.argv[1:]
pathlib.Path(args[args.index('-s') + 1]).write_text({exported_sdf!r})
""",
    )
    return receptor, ligand, vina, exporter


def _config(tools) -> RedockBenchmarkConfig:
    receptor, ligand, vina, exporter = tools
    return RedockBenchmarkConfig(
        seed=20260721,
        padding_angstrom=5.0,
        exhaustiveness=32,
        num_modes=9,
        vina=vina,
        mk_prepare_receptor=receptor,
        mk_prepare_ligand=ligand,
        mk_export=exporter,
        receptor_source="dataset:test/receptor",
        native_ligand_source="dataset:test/native",
        input_license="CC0-1.0",
        calibration_target_id="fixture-target",
        calibration_required_rank="top1",
        calibration_rmsd_threshold_angstrom=2.0,
    )


def test_redock_benchmark_runs_sealed_vertical_slice_with_fake_tools(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "receptor.pdb"
    native = tmp_path / "native.sdf"
    _receptor(receptor)
    modes = _native_and_modes(native)
    tools = _fake_tools(tmp_path / "tools", modes)
    monkeypatch.setattr(
        benchmark_module,
        "_posebusters_mode_checks",
        lambda *_: (True, {"fixture_check": True}),
    )

    output = tmp_path / "result"
    result = run_redock_benchmark(
        receptor, native, output, config=_config(tools)
    )

    assert result["status"] == "COMPLETED"
    assert result["scientific_status"] == "REDOCKING_RECOVERED_TOP1"
    assert result["evidence_grade"] == "REDOCKING_RECOVERED"
    assert result["top1_recovered"] is True
    assert result["top5_recovered"] is True
    assert result["toolchain"]["vina"]["version"] == "1.2.7"
    assert result["toolchain"]["rdkit"]["version"] == "2025.9.3"
    assert len(result["toolchain"]["vina"]["sha256"]) == 64
    assert result["config"]["conservative_receptor_repair"] is False
    assert result["input_commitments"]["receptor"]["sha256"] == sha256_file(
        receptor
    )
    assert result["input_commitments"]["native_ligand"]["sha256"] == sha256_file(
        native
    )
    assert result["code"]["root"] == "src/protbind_agent"
    assert result["code"]["file_count"] > 10
    assert len(result["code_sha256"]) == 64
    assert len(result["run_identity_sha256"]) == 64
    assert result["top1"]["posebusters_valid"] is True
    assert result["top1"]["symmetry_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-6)
    assert result["top5_oracle"]["any_pb_valid_and_rmsd_le_2"] is True
    assert len(result["top5_modes"]) == 2
    assert result["screening_calibration"]["decision"]["status"] == "PASS"
    assert result["screening_calibration"]["target_id"] == "fixture-target"
    assert result["screening_calibration"]["metrics"][
        "best_pb_valid_symmetry_rmsd_angstrom"
    ] == pytest.approx(0.0, abs=1e-6)
    assert result["pose_count"] == 2
    assert len(result["commands"]) == 4
    assert "native-reference" not in repr(result["commands"])
    assert any("not experimental binding" in text for text in result["disclaimers"])
    persisted = json.loads((output / "result.json").read_text())
    assert persisted["status"] == "COMPLETED"
    docking_case = (output / "artifacts" / "docking-visible-case.json").read_text()
    native_hash = result["artifacts"]["native_reference"]["sha256"]
    assert native_hash not in docking_case
    assert "native_pose" not in docking_case
    assert persisted["artifacts"]["native_reference"]["access_scope"] == "VALIDATION_ONLY"
    assert persisted["artifacts"]["receptor_input"]["source"] == "dataset:test/receptor"
    assert persisted["artifacts"]["native_reference"]["source"] == "dataset:test/native"
    assert persisted["artifacts"]["native_reference"]["license"] == "CC0-1.0"
    assert persisted["artifacts"]["prepared_receptor"]["license"] == "CC0-1.0"
    assert persisted["artifacts"]["ligand_pdbqt"]["license"] == "CC0-1.0"
    assert persisted["artifacts"]["vina_poses_sdf"]["license"] == "CC0-1.0"
    assert persisted["artifacts"]["receptor_record_order_receipt"][
        "record_order_changed"
    ] is False
    assert persisted["artifacts"]["native_reference"]["file"].startswith(
        "validation-only/"
    )
    assert not (output / "artifacts" / "native-reference.sdf").exists()
    calibration_path = output / "artifacts" / "known-site-calibration.json"
    assert calibration_path.is_file()
    calibration_text = calibration_path.read_text()
    assert native_hash not in calibration_text
    assert "native_reference" not in calibration_text
    assert "ligand_identity" not in calibration_text


def test_restrained_sidechain_retries_only_rdkit_failure_with_more_iterations(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "receptor.pdb"
    native = tmp_path / "native.sdf"
    _receptor(receptor)
    modes = _native_and_modes(native)
    tools = _fake_tools(
        tmp_path / "tools",
        modes,
        receptor_rdkit_failures=1,
    )

    def fake_repair(store, structure, **_kwargs):
        receipt = store.put_json(
            {"schema_version": "1.0", "fixture": "repair"},
            producer="test.repair-receipt",
            source=structure.artifact_id,
        )
        return ConservativeRepairResult(
            structure=structure,
            receipt=receipt,
            missing_residue_count=0,
            added_heavy_atom_count=1,
            warnings=(),
        )

    def fake_optimize(store, original, repaired, *, iteration_limit, **_kwargs):
        receipt = store.put_json(
            {
                "schema_version": "1.0",
                "fixture": "optimization",
                "iteration_limit": iteration_limit,
            },
            producer="test.optimization-receipt",
            source=repaired.artifact_id,
        )
        return RestrainedSidechainOptimizationResult(
            structure=repaired,
            receipt=receipt,
            iteration_limit=iteration_limit,
            fixed_original_heavy_atom_count=5,
            mobile_added_heavy_atom_count=1,
            original_heavy_atom_max_coordinate_delta_angstrom=0.0,
            minimum_nonbonded_distance_ratio=0.8,
            chirality_center_count=1,
            initial_energy_kj_mol=2.0,
            final_energy_kj_mol=1.0,
        )

    monkeypatch.setattr(benchmark_module, "conservative_heavy_atom_repair", fake_repair)
    monkeypatch.setattr(
        benchmark_module,
        "restrained_sidechain_geometry_optimize",
        fake_optimize,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_posebusters_mode_checks",
        lambda *_: (True, {"fixture_check": True}),
    )
    config = _config(tools)
    config = RedockBenchmarkConfig(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field
            not in {
                "conservative_receptor_repair",
                "restrained_sidechain_optimization",
                "sidechain_optimization_iteration_limits",
            }
        },
        conservative_receptor_repair=True,
        restrained_sidechain_optimization=True,
        sidechain_optimization_iteration_limits=(10, 20),
    )
    result = run_redock_benchmark(
        receptor,
        native,
        tmp_path / "result",
        config=config,
    )

    assert result["status"] == "COMPLETED"
    assert result["sidechain_optimization"]["accepted_iteration_limit"] == 20
    assert [
        item["status"] for item in result["sidechain_optimization"]["attempts"]
    ] == ["MEEKO_REJECTED", "ACCEPTED"]
    assert result["sidechain_optimization"]["attempts"][0][
        "retryable_rdkit_chemistry_failure"
    ] is True
    assert len(result["commands"]) == 5


def test_rdkit_receptor_failure_classifier_is_narrow():
    assert _retryable_rdkit_receptor_failure(
        "[RDKit] ERROR: Explicit valence for atom; AtomValenceException"
    )
    assert not _retryable_rdkit_receptor_failure("permission denied")


def test_restrained_sidechain_config_requires_repair_and_ordered_iterations():
    with pytest.raises(ValueError, match="requires conservative receptor repair"):
        RedockBenchmarkConfig(restrained_sidechain_optimization=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        RedockBenchmarkConfig(
            conservative_receptor_repair=True,
            restrained_sidechain_optimization=True,
            sidechain_optimization_iteration_limits=(1000, 250),
        )


def test_restrained_sidechain_cli_parses_iteration_schedule():
    args = _build_parser().parse_args(
        [
            "benchmark",
            "redock",
            "--output",
            "out",
            "--conservative-receptor-repair",
            "--restrained-sidechain-optimization",
            "--sidechain-optimization-iterations",
            "250",
            "1000",
            "5000",
        ]
    )
    assert args.restrained_sidechain_optimization is True
    assert args.sidechain_optimization_iterations == [250, 1000, 5000]


def test_redock_benchmark_accepts_extensionless_content_addressed_inputs(
    tmp_path, monkeypatch
):
    receptor_with_suffix = tmp_path / "receptor.pdb"
    native_with_suffix = tmp_path / "native.sdf"
    _receptor(receptor_with_suffix)
    modes = _native_and_modes(native_with_suffix)
    receptor = tmp_path / "6de627cc552fd8aa"
    native = tmp_path / "5a39de22be12cff7"
    receptor.write_bytes(receptor_with_suffix.read_bytes())
    native.write_bytes(native_with_suffix.read_bytes())
    tools = _fake_tools(tmp_path / "tools", modes)
    monkeypatch.setattr(
        benchmark_module,
        "_posebusters_mode_checks",
        lambda *_: (True, {"fixture_check": True}),
    )

    result = run_redock_benchmark(
        receptor,
        native,
        tmp_path / "extensionless-result",
        config=_config(tools),
    )

    assert result["status"] == "COMPLETED"
    assert result["artifacts"]["receptor_input"]["sha256"]
    assert result["artifacts"]["native_reference"]["sha256"]


def test_redock_benchmark_persists_explicit_tool_failure(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    native = tmp_path / "native.sdf"
    _receptor(receptor)
    modes = _native_and_modes(native)
    tools = _fake_tools(tmp_path / "tools", modes, vina_exit=7)
    monkeypatch.setattr(
        benchmark_module,
        "_posebusters_mode_checks",
        lambda *_: (True, {"fixture_check": True}),
    )

    output = tmp_path / "failed"
    result = run_redock_benchmark(
        receptor, native, output, config=_config(tools)
    )

    assert result["status"] == "FAILED"
    assert result["failure"]["stage"] == "vina"
    assert result["failure"]["code"] == "TOOL_NONZERO_EXIT"
    assert "code 7" in result["failure"]["message"]
    assert json.loads((output / "result.json").read_text())["status"] == "FAILED"
    assert (output / "logs" / "vina.log").is_file()


def test_cli_parser_preserves_flat_benchmark_and_adds_redock():
    parser = _build_parser()
    flat = parser.parse_args(
        ["benchmark", "--index", "index.bin", "--query", "query.json", "--output", "out.json"]
    )
    redock = parser.parse_args(
        [
            "benchmark",
            "redock",
            "--receptor",
            "protein.pdb",
            "--native-ligand",
            "native.sdf",
            "--output",
            "redock-output",
            "--calibration-target-id",
            "public-target",
            "--calibration-required-rank",
            "top5",
            "--calibration-rmsd-threshold",
            "1.5",
        ]
    )

    assert flat.benchmark_command is None
    assert flat.index == Path("index.bin")
    assert redock.benchmark_command == "redock"
    assert redock.seed == 20260721
    assert redock.padding == 5.0
    assert redock.exhaustiveness == 32
    assert redock.num_modes == 9
    assert redock.calibration_target_id == "public-target"
    assert redock.calibration_required_rank == "top5"
    assert redock.calibration_rmsd_threshold == 1.5


def test_cli_returns_nonzero_when_requested_calibration_gate_fails(
    tmp_path, monkeypatch
):
    args = _build_parser().parse_args(
        [
            "benchmark",
            "redock",
            "--receptor",
            "protein.pdb",
            "--native-ligand",
            "native.sdf",
            "--output",
            str(tmp_path / "result"),
            "--calibration-target-id",
            "public-target",
        ]
    )
    monkeypatch.setattr(
        cli_module,
        "run_redock_benchmark",
        lambda *_args, **_kwargs: {
            "status": "COMPLETED",
            "screening_calibration": {"decision": {"status": "FAIL"}},
        },
    )

    assert cli_module._run(args) == 3


def test_pdb_atom_order_normalizer_preserves_exact_atom_record_multiset(tmp_path):
    source = tmp_path / "out-of-order.pdb"
    destination = tmp_path / "canonical.pdb"
    _receptor(source)
    lines = source.read_bytes().splitlines(keepends=True)
    atom_lines = [line for line in lines if line.startswith(b"ATOM  ")]
    source.write_bytes(b"".join([atom_lines[-1], *atom_lines[:-1], *lines[len(atom_lines) :]]))

    receipt = benchmark_module._canonicalize_pdb_atom_record_order(
        source, destination
    )

    output_atoms = [
        line
        for line in destination.read_bytes().splitlines(keepends=True)
        if line.startswith(b"ATOM  ")
    ]
    output_serials = [int(line[6:11]) for line in output_atoms]
    assert receipt["record_order_changed"] is True
    assert receipt["coordinates_changed"] is False
    assert receipt["atom_record_bytes_changed"] is False
    assert output_serials == sorted(output_serials)
    assert sorted(output_atoms) == sorted(atom_lines)


def test_pdb_atom_order_normalizer_refuses_anisou_sidecars(tmp_path):
    source = tmp_path / "unsafe.pdb"
    destination = tmp_path / "canonical.pdb"
    _receptor(source)
    lines = source.read_bytes().splitlines(keepends=True)
    atom_lines = [line for line in lines if line.startswith(b"ATOM  ")]
    unsafe = [atom_lines[-1], *atom_lines[:-1], b"ANISOU sidecar cannot be reordered safely\n"]
    source.write_bytes(b"".join(unsafe))

    with pytest.raises(benchmark_module.RedockBenchmarkError) as error:
        benchmark_module._canonicalize_pdb_atom_record_order(source, destination)

    assert error.value.code == "UNSAFE_RECORD_REORDER"


def test_posebusters_geometry_validity_excludes_reference_rmsd_gate():
    pandas = pytest.importorskip("pandas")
    valid, checks = benchmark_module._boolean_posebusters_report(
        pandas.DataFrame([{"geometry": True, "rmsd_≤_2a": False}])
    )

    assert valid is True
    assert checks["rmsd_≤_2a"] is False


def test_redock_benchmark_persists_preflight_input_failure(tmp_path):
    output = tmp_path / "failed-preflight"
    result = run_redock_benchmark(
        tmp_path / "missing.pdb",
        tmp_path / "missing.sdf",
        output,
    )

    assert result["status"] == "FAILED"
    assert result["failure"]["stage"] == "preflight"
    assert result["failure"]["code"] == "INPUT_UNAVAILABLE"
    assert json.loads((output / "result.json").read_text())["status"] == "FAILED"
