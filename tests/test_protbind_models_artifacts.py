from __future__ import annotations

import json

import pytest

from protbind_agent.artifacts import ArtifactStore, sha256_bytes
from protbind_agent.manifest import (
    CofoldStatus,
    ManifestStore,
    RunManifest,
    RunState,
    StageRecord,
)
from protbind_agent.models import (
    ArtifactRef,
    EvidenceClaim,
    EvidenceStatus,
    LigandHypothesis,
    PocketHypothesis,
    ResearchCase,
    ResearchMode,
    SiteProvenanceKind,
    TargetSpec,
)
from protbind_agent.privacy import redact_text


def _ref(content: bytes = b"artifact") -> ArtifactRef:
    return ArtifactRef(
        sha256=sha256_bytes(content),
        media_type="application/json",
        size_bytes=len(content),
        producer="test",
    )


def test_research_modes_and_v1_boundaries() -> None:
    target = TargetSpec(name="kinase", sequences=("ACDE",))
    ligand = LigandHypothesis(smiles="CCO", heavy_atom_count=3)
    pocket = PocketHypothesis(residues=("A:42",))

    case = ResearchCase(
        case_id="case-1",
        target=target,
        mode=ResearchMode.BOTH,
        ligand=ligand,
        pocket=pocket,
    )

    assert ResearchCase.from_dict(case.to_dict()) == case
    with pytest.raises(ValueError, match="requires ligand and pocket"):
        ResearchCase(case_id="bad", target=target, mode=ResearchMode.BOTH, ligand=ligand)
    with pytest.raises(ValueError, match="at most 700"):
        TargetSpec(name="too-long", sequences=("A" * 701,))
    with pytest.raises(ValueError, match="at most 100"):
        LigandHypothesis(smiles="C", heavy_atom_count=101)
    with pytest.raises(ValueError, match="unsupported v1 chemistry"):
        LigandHypothesis(smiles="C", is_covalent=True)
    with pytest.raises(ValueError, match="three finite"):
        PocketHypothesis(center=(0.0, float("nan"), 0.0), box_size=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="three finite"):
        PocketHypothesis(center=(0.0, 0.0), box_size=(1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires site derivation evidence"):
        PocketHypothesis(
            center=(0.0, 0.0, 0.0),
            box_size=(10.0, 10.0, 10.0),
            site_provenance_kind=SiteProvenanceKind.COCRYSTAL_LIGAND,
        )
    with pytest.raises(ValueError, match="requires an explicit"):
        PocketHypothesis(residues=("A:42",), site_evidence=_ref())
    with pytest.raises(ValueError, match="user-declared site provenance"):
        PocketHypothesis(
            residues=("A:42",),
            site_provenance_kind=SiteProvenanceKind.USER_RESIDUES,
            site_evidence=_ref(),
        )
    with pytest.raises(ValueError, match="provided together"):
        PocketHypothesis(
            center=(0.0, 0.0, 0.0),
            box_size=(10.0, 10.0, 10.0),
            site_provenance_kind=SiteProvenanceKind.PUBLIC_BENCHMARK_REFERENCE,
            site_evidence=_ref(b"site"),
            known_site_calibration_receipt=_ref(b"calibration"),
        )
    with pytest.raises(ValueError, match="independent site provenance"):
        PocketHypothesis(
            center=(0.0, 0.0, 0.0),
            box_size=(10.0, 10.0, 10.0),
            site_provenance_kind=SiteProvenanceKind.USER_CENTER,
            known_site_calibration_receipt=_ref(b"calibration"),
            known_site_calibration_target_id="target",
        )

    calibrated_pocket = PocketHypothesis(
        center=(0.0, 0.0, 0.0),
        box_size=(10.0, 10.0, 10.0),
        site_provenance_kind=SiteProvenanceKind.PUBLIC_BENCHMARK_REFERENCE,
        site_evidence=_ref(b"site"),
        known_site_calibration_receipt=_ref(b"calibration"),
        known_site_calibration_target_id="target",
    )
    with pytest.raises(ValueError, match="only for mode=both"):
        ResearchCase(
            case_id="calibrated-pocket-only",
            target=target,
            mode=ResearchMode.POCKET_ONLY,
            pocket=calibrated_pocket,
        )
    calibrated_case = ResearchCase(
        case_id="calibrated-both",
        target=target,
        mode=ResearchMode.BOTH,
        ligand=ligand,
        pocket=calibrated_pocket,
    )
    assert ResearchCase.from_dict(calibrated_case.to_dict()) == calibrated_case


def test_evidence_claim_requires_real_artifact_reference() -> None:
    with pytest.raises(ValueError, match="require artifact evidence"):
        EvidenceClaim(
            claim_id="claim-1",
            text="This is supported.",
            status=EvidenceStatus.SUPPORTED,
        )
    claim = EvidenceClaim(
        claim_id="claim-1",
        text="The tool supports this claim.",
        status=EvidenceStatus.SUPPORTED,
        evidence=(_ref(),),
    )
    assert claim.evidence[0].artifact_id.startswith("sha256:")


def test_artifact_store_is_deduplicated_and_does_not_persist_absolute_source(tmp_path) -> None:
    source = tmp_path / "private" / "query.json"
    source.parent.mkdir()
    source.write_text('{"features": []}', encoding="utf-8")
    store = ArtifactStore(tmp_path / "store")

    first = store.import_file(source, media_type="application/json")
    second = store.import_file(source, media_type="application/json")

    assert first == second
    assert first.source == "local-import:query.json"
    assert str(tmp_path) not in json.dumps(first.to_dict())
    assert store.read_bytes(first) == source.read_bytes()
    with pytest.raises(ValueError, match="absolute local path"):
        ArtifactRef(
            sha256="a" * 64,
            media_type="application/json",
            size_bytes=1,
            producer="test",
            source="/private/research/input.json",
        )


def test_redaction_covers_provider_keys_bearer_tokens_and_cross_platform_paths() -> None:
    text = (
        'OPENAI_API_KEY="sk-private" '
        "DEEPSEEK_TOKEN=token-value Authorization: Bearer abc.def.ghi "
        "/home/researcher/private/model.pt C:\\Users\\researcher\\model.pt"
    )

    redacted = redact_text(text)

    assert "sk-private" not in redacted
    assert "token-value" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "/home/researcher" not in redacted
    assert "C:\\Users" not in redacted


def test_manifest_transition_cache_and_resume_round_trip(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    case_artifact = store.put_json({"case_id": "x"}, producer="test")
    output = store.put_json({"ok": True}, producer="test")
    manifest = RunManifest(run_id="run-1", case_id="case-1", case_artifact=case_artifact)
    record = StageRecord.create(
        RunState.INPUT_VALIDATED,
        input_hash="a" * 64,
        config_hash="b" * 64,
        outputs=(output,),
        duration_seconds=0.1,
    )

    manifest.complete_stage(record)
    assert manifest.schema_version == "2.0"
    assert manifest.cached_outputs(
        RunState.INPUT_VALIDATED, input_hash="a" * 64, config_hash="b" * 64
    ) == (output,)
    assert manifest.cached_outputs(
        RunState.INPUT_VALIDATED, input_hash="c" * 64, config_hash="b" * 64
    ) is None
    manifest.degrade(
        stage=RunState.INDEXED,
        code="MISSING",
        message="capability missing",
    )
    assert manifest.state is RunState.DEGRADED
    manifest.prepare_resume()
    assert manifest.state is RunState.INPUT_VALIDATED

    manifests = ManifestStore(tmp_path / "workspace")
    manifests.save(manifest)
    loaded = manifests.load("run-1")
    assert loaded.to_dict() == manifest.to_dict()


def test_manifest_rejects_out_of_order_transition() -> None:
    manifest = RunManifest(run_id="run", case_id="case", case_artifact=_ref())
    with pytest.raises(ValueError, match="invalid transition"):
        manifest.complete_stage(
            StageRecord.create(
                RunState.SCREENED,
                input_hash="a" * 64,
                config_hash="b" * 64,
                outputs=(_ref(b"output"),),
                duration_seconds=0,
            )
        )


def test_manifest_invariants_reject_terminal_or_discontinuous_state() -> None:
    with pytest.raises(ValueError, match="recoverable failure"):
        RunManifest(
            run_id="run",
            case_id="case",
            case_artifact=_ref(),
            state=RunState.DEGRADED,
        )
    with pytest.raises(ValueError, match="contiguous"):
        RunManifest(
            run_id="run",
            case_id="case",
            case_artifact=_ref(),
            state=RunState.SCREENED,
            last_completed_stage=RunState.SCREENED,
            stage_records={},
        )


def test_reobserving_cached_stage_does_not_roll_manifest_back() -> None:
    manifest = RunManifest(run_id="run", case_id="case", case_artifact=_ref())
    first = StageRecord.create(
        RunState.INPUT_VALIDATED,
        input_hash="a" * 64,
        config_hash="b" * 64,
        outputs=(_ref(b"first"),),
        duration_seconds=0,
    )
    second = StageRecord.create(
        RunState.RECEPTOR_READY,
        input_hash="c" * 64,
        config_hash="d" * 64,
        outputs=(_ref(b"second"),),
        duration_seconds=0,
    )
    third = StageRecord.create(
        RunState.INDEXED,
        input_hash="e" * 64,
        config_hash="f" * 64,
        outputs=(_ref(b"third"),),
        duration_seconds=0,
    )
    manifest.complete_stage(first)
    manifest.complete_stage(second)
    manifest.complete_stage(third)

    manifest.complete_stage(first)

    assert manifest.state is RunState.INDEXED
    assert manifest.last_completed_stage is RunState.INDEXED


def test_optional_cofold_is_not_a_main_state_transition() -> None:
    manifest = RunManifest(run_id="run", case_id="case", case_artifact=_ref())
    for index, stage in enumerate(
        (
            RunState.INPUT_VALIDATED,
            RunState.RECEPTOR_READY,
            RunState.INDEXED,
            RunState.SCREENED,
            RunState.SELECTED,
        )
    ):
        manifest.complete_stage(
            StageRecord.create(
                stage,
                input_hash=f"{index + 1:x}" * 64,
                config_hash=f"{index + 6:x}" * 64,
                outputs=(_ref(f"stage-{stage.value}".encode()),),
                duration_seconds=0,
            )
        )
    cofold = StageRecord.create(
        RunState.COFOLDED,
        input_hash="b" * 64,
        config_hash="c" * 64,
        outputs=(_ref(b"cofold"),),
        duration_seconds=0,
    )

    manifest.begin_cofold()
    manifest.complete_cofold(cofold)

    assert manifest.state is RunState.SELECTED
    assert manifest.last_completed_stage is RunState.SELECTED
    assert manifest.next_stage is RunState.DOCKED
    assert manifest.cofold_status is CofoldStatus.COMPLETED
    assert RunState.COFOLDED.value not in manifest.stage_records


def test_schema_one_manifest_loads_read_only(tmp_path) -> None:
    value = {
        "schema_version": "1.0",
        "run_id": "legacy",
        "case_id": "case",
        "case_artifact": _ref().to_dict(),
        "input_artifacts": {},
        "artifacts": {},
        "state": "CREATED",
        "last_completed_stage": "CREATED",
        "stage_records": {},
        "failures": [],
        "provenance": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    manifest = RunManifest.from_dict(value)

    assert manifest.is_read_only is True
    assert manifest.to_dict() == value
    with pytest.raises(ValueError, match="read-only"):
        ManifestStore(tmp_path).save(manifest)
    with pytest.raises(ValueError, match="read-only"):
        manifest.degrade(stage=RunState.INPUT_VALIDATED, code="x", message="x")
