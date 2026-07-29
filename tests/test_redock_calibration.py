from __future__ import annotations

from copy import deepcopy

import pytest

from protbind_agent.artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes
from protbind_agent.models import ArtifactRef
from protbind_agent.redock_calibration import (
    KnownSiteCalibrationConfig,
    build_known_site_calibration_receipt,
    validate_known_site_calibration_artifact,
    validate_known_site_calibration_receipt,
)


def _artifact(
    digit: str,
    *,
    media_type: str,
    producer: str,
) -> dict[str, object]:
    return {
        "sha256": digit * 64,
        "media_type": media_type,
        "producer": producer,
        "producer_version": "test-1",
        "access_scope": "DOCKING_VISIBLE",
    }


def _redock_result() -> dict[str, object]:
    modes = [
        {
            "mode": 1,
            "posebusters_valid": True,
            "symmetry_rmsd_angstrom": 1.4,
        },
        {
            "mode": 2,
            "posebusters_valid": False,
            "symmetry_rmsd_angstrom": 0.2,
        },
        {
            "mode": 3,
            "posebusters_valid": True,
            "symmetry_rmsd_angstrom": 0.8,
        },
    ]
    return {
        "schema_version": "1.2",
        "benchmark": "redock",
        "status": "COMPLETED",
        "run_identity_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "toolchain_sha256": "c" * 64,
        "code_sha256": "d" * 64,
        "artifacts": {
            "prepared_receptor": _artifact(
                "1",
                media_type="chemical/x-pdb",
                producer="meeko.mk_prepare_receptor",
            ),
            "receptor_preparation_receipt": _artifact(
                "2",
                media_type="application/json",
                producer="protbind.redocking.meeko-receptor-receipt",
            ),
            "native_box_receipt": _artifact(
                "3",
                media_type="application/json",
                producer="protbind.redocking.native-box-receipt",
            ),
            "native_reference": {
                "sha256": "4" * 64,
                "access_scope": "VALIDATION_ONLY",
            },
        },
        "top1": modes[0],
        "top5_modes": modes,
    }


def _source_result(result: dict[str, object]) -> ArtifactRef:
    data = canonical_json_bytes(result)
    return ArtifactRef(
        sha256=sha256_bytes(data),
        media_type="application/json",
        size_bytes=len(data),
        producer="protbind.redocking.calibration-source",
        producer_version="test-1",
    )


def _receipt(
    result: dict[str, object], config: KnownSiteCalibrationConfig
) -> dict[str, object]:
    return build_known_site_calibration_receipt(
        result,
        config,
        source_result=_source_result(result),
    )


def _stored_calibration(store: ArtifactStore):
    receptor = store.put_bytes(
        b"ATOM receptor\n",
        media_type="chemical/x-pdb",
        producer="meeko.mk_prepare_receptor",
        producer_version="test-1",
    )
    box = store.put_json(
        {
            "schema_version": "1.0",
            "definition": "redock-known-site",
            "algorithm": "fixture",
            "native_pose_commitment": "9" * 64,
            "center": [1.0, 2.0, 3.0],
            "size": [10.0, 11.0, 12.0],
            "padding_angstrom": 5.0,
            "heavy_atom_count": 3,
            "native_coordinates_exposed_to_docking": False,
            "native_coordinates_used_for_box_derivation": True,
        },
        producer="protbind.redocking.native-box-receipt",
        producer_version="test-1",
    )
    preparation = store.put_json(
        {
            "schema_version": "1.0",
            "source_receptor": receptor.artifact_id,
            "meeko_input_receptor": receptor.artifact_id,
            "record_order_receipt": "sha256:" + "8" * 64,
            "prepared_receptor": receptor.artifact_id,
            "receptor_pdbqt": "sha256:" + "7" * 64,
            "meeko_json": "sha256:" + "6" * 64,
            "box_receipt": box.artifact_id,
            "protein_only_input_required": True,
            "allow_bad_residues": False,
            "possible_cofactors_silently_removed": False,
        },
        producer="protbind.redocking.meeko-receptor-receipt",
        producer_version="test-1",
    )
    result = _redock_result()

    def visible(reference: ArtifactRef) -> dict[str, object]:
        return {**reference.to_dict(), "access_scope": "DOCKING_VISIBLE"}

    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["prepared_receptor"] = visible(receptor)
    artifacts["receptor_preparation_receipt"] = visible(preparation)
    artifacts["native_box_receipt"] = visible(box)
    source = store.put_json(
        result,
        producer="protbind.redocking.calibration-source",
        producer_version="test-1",
    )
    receipt = build_known_site_calibration_receipt(
        result,
        KnownSiteCalibrationConfig(target_id="fixture-target", required_rank="top1"),
        source_result=source,
    )
    calibration = store.put_json(
        receipt,
        producer="protbind.redocking.known-site-calibration",
        producer_version="test-1",
    )
    return receptor, receipt, calibration


def test_top1_and_top5_gates_are_explicit_and_pb_valid() -> None:
    result = _redock_result()
    top1 = _receipt(
        result,
        KnownSiteCalibrationConfig(
            target_id="target-kinase",
            required_rank="top1",
            rmsd_threshold_angstrom=1.0,
        ),
    )
    top5 = _receipt(
        result,
        KnownSiteCalibrationConfig(
            target_id="target-kinase",
            required_rank="top5",
            rmsd_threshold_angstrom=1.0,
        ),
    )

    assert top1["metrics"]["top1_posebusters_valid"] is True
    assert top1["metrics"]["top1_symmetry_rmsd_angstrom"] == 1.4
    assert top1["metrics"]["best_pb_valid_mode"] == 3
    assert top1["metrics"]["best_pb_valid_symmetry_rmsd_angstrom"] == 0.8
    assert top1["metrics"]["pb_valid_mode_count"] == 2
    assert top1["decision"]["status"] == "FAIL"
    assert top5["decision"]["status"] == "PASS"
    assert len(top5["calibration_config_sha256"]) == 64
    assert len(top5["receipt_sha256"]) == 64

    with pytest.raises(ValueError, match="did not pass"):
        validate_known_site_calibration_receipt(top1)
    validate_known_site_calibration_receipt(
        top5,
        expected_target_id="target-kinase",
        expected_prepared_receptor_sha256="1" * 64,
    )


def test_calibration_authorization_cannot_relax_past_two_angstrom() -> None:
    with pytest.raises(ValueError, match=r"\(0, 2\.0\]"):
        KnownSiteCalibrationConfig(
            target_id="target-kinase",
            rmsd_threshold_angstrom=2.0001,
        )


def test_receipt_contains_only_box_and_receptor_authorizations() -> None:
    result = _redock_result()
    native_sha = result["artifacts"]["native_reference"]["sha256"]
    receipt = _receipt(
        result,
        KnownSiteCalibrationConfig(target_id="1iep", required_rank="top5"),
    )

    serialized = repr(receipt)
    assert native_sha not in serialized
    assert "native_reference" not in serialized
    assert "ligand_identity" not in serialized
    assert set(receipt["authorized_inputs"]) == {
        "prepared_receptor",
        "receptor_preparation_receipt",
        "known_site_box_receipt",
        "coordinate_frame",
    }
    assert (
        receipt["reference_boundary"]["permitted_downstream_derivative"]
        == "KNOWN_SITE_BOX_RECEIPT_ONLY"
    )
    assert (
        receipt["reference_boundary"][
            "reference_ligand_visible_to_candidate_pose_generation"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_id", "other-target", "different target"),
        ("prepared_receptor", "f" * 64, "different prepared receptor"),
    ],
)
def test_validator_rejects_target_or_receptor_reuse(
    field: str,
    value: str,
    message: str,
) -> None:
    receipt = _receipt(
        _redock_result(),
        KnownSiteCalibrationConfig(target_id="1iep", required_rank="top5"),
    )
    kwargs = {
        "expected_target_id": "1iep",
        "expected_prepared_receptor_sha256": "1" * 64,
    }
    if field == "target_id":
        kwargs["expected_target_id"] = value
    else:
        kwargs["expected_prepared_receptor_sha256"] = value

    with pytest.raises(ValueError, match=message):
        validate_known_site_calibration_receipt(receipt, **kwargs)


def test_validator_rejects_hash_tampering_and_reference_injection() -> None:
    receipt = _receipt(
        _redock_result(),
        KnownSiteCalibrationConfig(target_id="1iep", required_rank="top5"),
    )
    tampered = deepcopy(receipt)
    tampered["metrics"]["best_pb_valid_symmetry_rmsd_angstrom"] = 9.0
    with pytest.raises(ValueError):
        validate_known_site_calibration_receipt(tampered, require_pass=False)

    config_tampered = deepcopy(receipt)
    config_tampered["calibration_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_known_site_calibration_receipt(config_tampered, require_pass=False)

    injected = deepcopy(receipt)
    injected["native_reference"] = {"sha256": "4" * 64}
    with pytest.raises(ValueError, match="unauthorized fields"):
        validate_known_site_calibration_receipt(injected, require_pass=False)


def test_legacy_completed_redock_requires_identifiable_pb_valid_oracle() -> None:
    result = _redock_result()
    del result["top5_modes"]
    result["top5_oracle"] = {
        "evaluated_modes": 5,
        "best_mode": 1,
        "best_symmetry_rmsd_angstrom": 1.4,
        "any_pb_valid_and_rmsd_le_2": True,
        "first_recovered_mode": 1,
    }
    receipt = _receipt(
        result,
        KnownSiteCalibrationConfig(target_id="legacy-1iep"),
    )
    assert receipt["metrics"]["pb_valid_mode_count"] is None
    assert receipt["metrics"]["pb_valid_mode_count_complete"] is False
    assert receipt["decision"]["status"] == "PASS"

    result["top5_oracle"]["first_recovered_mode"] = 2
    with pytest.raises(ValueError, match="cannot identify"):
        _receipt(
            result,
            KnownSiteCalibrationConfig(target_id="legacy-ambiguous"),
        )


def test_store_validation_cross_checks_source_artifacts_metrics_and_box(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    receptor, receipt, calibration = _stored_calibration(store)

    validate_known_site_calibration_artifact(
        store,
        calibration,
        expected_target_id="fixture-target",
        expected_prepared_receptor_sha256=receptor.sha256,
        expected_box_center=[1.0, 2.0, 3.0],
        expected_box_size=[10.0, 11.0, 12.0],
    )

    tampered = deepcopy(receipt)
    tampered["metrics"]["top1_symmetry_rmsd_angstrom"] = 1.5
    tampered_core = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = sha256_bytes(canonical_json_bytes(tampered_core))
    with pytest.raises(ValueError, match="recomputed source"):
        validate_known_site_calibration_receipt(
            tampered,
            require_pass=False,
            store=store,
        )

    with pytest.raises(ValueError, match="different known-site box"):
        validate_known_site_calibration_artifact(
            store,
            calibration,
            expected_target_id="fixture-target",
            expected_prepared_receptor_sha256=receptor.sha256,
            expected_box_center=[2.0, 2.0, 3.0],
            expected_box_size=[10.0, 11.0, 12.0],
        )

    forged = deepcopy(receipt)
    replacement = store.put_bytes(
        b"different receptor\n",
        media_type="chemical/x-pdb",
        producer="meeko.mk_prepare_receptor",
        producer_version="test-1",
    )
    forged["authorized_inputs"]["prepared_receptor"] = {
        "sha256": replacement.sha256,
        "media_type": replacement.media_type,
        "producer": replacement.producer,
        "producer_version": replacement.producer_version,
    }
    forged_core = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = sha256_bytes(canonical_json_bytes(forged_core))
    with pytest.raises(ValueError, match="differ from its source redock artifacts"):
        validate_known_site_calibration_receipt(
            forged,
            require_pass=False,
            store=store,
        )
