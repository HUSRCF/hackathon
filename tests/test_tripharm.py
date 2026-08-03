from __future__ import annotations

import math
import struct
import subprocess
from pathlib import Path

import pytest

from protbind_agent.fusion import reciprocal_rank_fusion
from protbind_agent.tripharm import (
    FeatureConformer,
    FeaturePoint,
    FeatureType,
    IndexedMolecule,
    TriPharmConfig,
    build_index,
    query_index,
    read_index_metadata,
    rigid_transform,
    select_query_triangles,
)
from protbind_agent.tripharm_hip import query_index_batch_hip, query_index_hip


def _features(offset: float = 0.0, distortion: float = 0.0):
    return (
        FeaturePoint(FeatureType.DONOR, (offset, 0.0, 0.0), (0,)),
        FeaturePoint(FeatureType.ACCEPTOR, (offset + 3.0 + distortion, 0.0, 0.0), (1,)),
        FeaturePoint(FeatureType.AROMATIC, (offset, 4.0, 0.0), (2, 3)),
        FeaturePoint(FeatureType.HYDROPHOBE, (offset, 0.0, 5.0), (4,)),
    )


def _molecule(molecule_id: str, features) -> IndexedMolecule:
    return IndexedMolecule(
        molecule_id=molecule_id,
        original_smiles="CCO",
        standardized_smiles="CCO",
        conformers=(FeatureConformer(0, tuple(features)),),
        source="fixture",
    )


def _apply(transform, point):
    return tuple(
        sum(transform[row][column] * point[column] for column in range(3))
        + transform[row][3]
        for row in range(3)
    )


def test_cpu_index_query_order_and_overlay_are_deterministic(tmp_path) -> None:
    index = tmp_path / "library.sqlite"
    config = TriPharmConfig(bin_width_angstrom=0.5, tolerance_angstrom=1.0)
    stats = build_index(
        [
            _molecule("exact", _features(offset=10.0)),
            _molecule("near", _features(offset=-5.0, distortion=0.4)),
            _molecule(
                "wrong-type",
                (
                    FeaturePoint(FeatureType.POSITIVE, (0.0, 0.0, 0.0)),
                    FeaturePoint(FeatureType.NEGATIVE, (3.0, 0.0, 0.0)),
                    FeaturePoint(FeatureType.HYDROPHOBE, (0.0, 4.0, 0.0)),
                ),
            ),
        ],
        index,
        config=config,
    )

    first = query_index(index, _features(), top_k=10)
    second = query_index(index, _features(), top_k=10)

    assert stats.molecule_count == 3
    assert [hit.molecule_id for hit in first] == ["exact", "near"]
    assert first == second
    assert first[0].query_coverage == 1.0
    assert first[0].median_normalized_distance_error < first[1].median_normalized_distance_error
    assert len(first[0].matches) > 1
    assert len({match.overlay for match in first[0].matches}) == 1
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for coherent_match in first[0].matches:
        for query_feature_index, candidate_index in zip(
            coherent_match.query_feature_indices,
            coherent_match.candidate_feature_indices,
            strict=True,
        ):
            assert (
                forward.setdefault(query_feature_index, candidate_index) == candidate_index
            )
            assert (
                reverse.setdefault(candidate_index, query_feature_index)
                == query_feature_index
            )
    match = first[0].matches[0]
    candidate = _features(offset=10.0)[match.candidate_feature_indices[0]].position
    query = _features()[match.query_feature_indices[0]].position
    assert math.dist(_apply(match.overlay, candidate), query) < 1e-7
    stored_config, stored_stats = read_index_metadata(index)
    assert stored_config == config
    assert stored_stats.triangle_count == stats.triangle_count


def test_rigid_transform_recovers_rotation_and_translation() -> None:
    source = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0))
    target = ((5.0, -2.0, 1.0), (5.0, 0.0, 1.0), (2.0, -2.0, 1.0))
    transform = rigid_transform(source, target)

    for left, right in zip(source, target, strict=True):
        assert math.dist(_apply(transform, left), right) < 1e-7


def test_equal_weight_rrf_preserves_independent_branch_ranks(tmp_path) -> None:
    index = tmp_path / "library.sqlite"
    build_index(
        [_molecule("a", _features(1.0)), _molecule("b", _features(2.0, 0.2))],
        index,
    )
    ligand = query_index(index, _features(), top_k=2)
    pocket = list(reversed(ligand))

    fused = reciprocal_rank_fusion({"ligand": ligand, "pocket": pocket}, rrf_k=60)

    assert [hit.molecule_id for hit in fused] == ["a", "b"]
    assert fused[0].branch_ranks == {"ligand": 1, "pocket": 2}
    assert fused[1].branch_ranks == {"ligand": 2, "pocket": 1}
    assert fused[0].rrf_score == fused[1].rrf_score


def test_conflicting_triangle_correspondences_do_not_inflate_coverage(tmp_path) -> None:
    query = _features()
    config = TriPharmConfig(
        bin_width_angstrom=0.5,
        tolerance_angstrom=1.0,
        max_query_triangles=2,
    )
    selected_triangles = select_query_triangles(query, max_triangles=2)
    covered_query_indices = set(selected_triangles[0].feature_indices) | set(
        selected_triangles[1].feature_indices
    )
    assert len(covered_query_indices) == 4

    # Each selected query triangle has an exact candidate copy, but the copies
    # are far apart.  Their shared query features therefore map to different
    # candidate features and cannot describe one pharmacophore overlay.
    candidate_features = []
    for copy_index, triangle in enumerate(selected_triangles):
        translation = 100.0 * copy_index
        for query_feature_index in triangle.feature_indices:
            point = query[query_feature_index]
            candidate_features.append(
                FeaturePoint(
                    point.feature_type,
                    (
                        point.position[0] + translation,
                        point.position[1],
                        point.position[2],
                    ),
                    point.atom_indices,
                )
            )

    index = tmp_path / "conflicting.sqlite"
    build_index([_molecule("conflicting", candidate_features)], index, config=config)

    hits = query_index(index, query, top_k=1)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.query_coverage == 0.75
    assert len(hit.matches) == 1
    assert len(
        {query_index for match in hit.matches for query_index in match.query_feature_indices}
    ) == 3


def test_hip_prefilter_contract_commits_only_after_exact_cpu_parity(
    tmp_path, monkeypatch
) -> None:
    index = tmp_path / "library.sqlite"
    build_index(
        [
            _molecule("exact", _features(offset=10.0)),
            _molecule("near", _features(offset=-5.0, distortion=0.4)),
        ],
        index,
    )
    executable = tmp_path / "tripharm_hip_query"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    def fake_run(argv, **_kwargs):
        request = Path(argv[2]).read_bytes()
        _magic, _schema, _triangles, molecule_count, query_count, _tolerance = (
            struct.unpack_from("<8sIIIIf", request)
        )
        masks = struct.pack(f"<{molecule_count}Q", *([1] * molecule_count))
        errors = bytes(molecule_count * query_count * 4)
        Path(argv[4]).write_bytes(
            struct.pack(
                    "<8sIII",
                    b"TPHIPO1\0",
                    1,
                    molecule_count,
                    query_count,
                )
            + masks
            + errors
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                '{"architecture":"gfx1100","kernel_seconds":0.001,'
                '"matched_molecules":2}'
            ),
            stderr="",
        )

    monkeypatch.setattr("protbind_agent.tripharm_hip.subprocess.run", fake_run)

    result = query_index_hip(
        index,
        _features(),
        executable=executable,
        top_k=2,
    )

    assert [hit.molecule_id for hit in result.hits] == ["exact", "near"]
    assert result.receipt["ranked_molecule_ids_exact"] is True
    assert result.receipt["committed_backend"] == "hip"


def test_hip_request_cache_is_reused_and_tamper_evident(tmp_path, monkeypatch) -> None:
    index = tmp_path / "library.sqlite"
    build_index([_molecule("exact", _features())], index)
    executable = tmp_path / "tripharm_hip_query"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    def fake_run(argv, **_kwargs):
        request = Path(argv[2]).read_bytes()
        _, _, _, molecule_count, query_count, _ = struct.unpack_from(
            "<8sIIIIf", request
        )
        Path(argv[4]).write_bytes(
            struct.pack("<8sIII", b"TPHIPO1\0", 1, molecule_count, query_count)
            + struct.pack(f"<{molecule_count}Q", *([1] * molecule_count))
            + bytes(molecule_count * query_count * 4)
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"architecture":"gfx1100","kernel_seconds":0.001}',
            stderr="",
        )

    monkeypatch.setattr("protbind_agent.tripharm_hip.subprocess.run", fake_run)
    cache = tmp_path / "cache"
    first = query_index_hip(
        index, _features(), executable=executable, request_cache_dir=cache
    )
    second = query_index_hip(
        index,
        _features(),
        executable=executable,
        request_cache_dir=cache,
        cpu_reference_ids=["exact"],
    )
    assert first.receipt["request_cache_hit"] is False
    assert second.receipt["request_cache_hit"] is True
    assert second.receipt["cpu_reference_scope"] == "external_precomputed"
    request = next(cache.glob("*.tphipq"))
    request.write_bytes(request.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="cache (hash|size) mismatch"):
        query_index_hip(
            index, _features(), executable=executable, request_cache_dir=cache
        )


def test_hip_static_batch_is_exact_reused_and_tamper_evident(
    tmp_path, monkeypatch
) -> None:
    index = tmp_path / "library.sqlite"
    build_index(
        [
            _molecule("exact", _features(offset=10.0)),
            _molecule("near", _features(offset=-5.0, distortion=0.4)),
        ],
        index,
    )
    executable = tmp_path / "tripharm_hip_batch_query"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    def fake_run(argv, **_kwargs):
        static_data = Path(argv[2]).read_bytes()
        magic, schema, _triangle_count, molecule_count = struct.unpack_from(
            "<12sIII", static_data
        )
        assert magic == b"TPHIPIDX1\0\0\0"
        query_data = Path(argv[4]).read_bytes()
        query_magic, query_schema, batch_count, _queries, _tolerance = (
            struct.unpack_from("<12sIIIf", query_data)
        )
        assert query_magic == b"TPHIPBAT1\0\0\0"
        assert query_schema == schema
        Path(argv[6]).write_bytes(
            struct.pack(
                "<12sIII", b"TPHIPBO1\0\0\0\0", schema, molecule_count, batch_count
            )
            + struct.pack(
                f"<{molecule_count * batch_count}I",
                *([1] * (molecule_count * batch_count)),
            )
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                '{"architecture":"gfx1100","kernel_seconds":0.001,'
                '"batch_queries":2}'
            ),
            stderr="",
        )

    monkeypatch.setattr("protbind_agent.tripharm_hip.subprocess.run", fake_run)
    cache = tmp_path / "static-cache"
    first = query_index_batch_hip(
        index,
        (_features(), _features(distortion=0.2)),
        executable=executable,
        static_cache_dir=cache,
        tolerance_angstrom=1.0,
        top_k=2,
    )
    second = query_index_batch_hip(
        index,
        (_features(), _features(distortion=0.2)),
        executable=executable,
        static_cache_dir=cache,
        tolerance_angstrom=1.0,
        top_k=2,
        cpu_reference_ids=(
            tuple(hit.molecule_id for hit in first.hits[0]),
            tuple(hit.molecule_id for hit in first.hits[1]),
        ),
    )
    assert first.receipt["static_cache_hit"] is False
    assert second.receipt["static_cache_hit"] is True
    assert first.receipt["ranked_molecule_ids_exact"] is True
    assert first.receipt["cpu_ranked_scores_sha256"] == first.receipt[
        "hip_ranked_scores_sha256"
    ]
    static_path = next(cache.glob("*.tphipidx"))
    static_path.write_bytes(static_path.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="static cache (hash|size) mismatch"):
        query_index_batch_hip(
            index,
            (_features(),),
            executable=executable,
            static_cache_dir=cache,
            tolerance_angstrom=1.0,
        )
