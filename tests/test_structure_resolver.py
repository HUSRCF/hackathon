from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime

import pytest

from protbind_agent.artifacts import ArtifactStore, sha256_bytes
from protbind_agent.chemistry import ChemistryCapabilityError
from protbind_agent.cli import _build_parser
from protbind_agent.manifest import RunState
from protbind_agent.models import (
    ArtifactRef,
    LigandHypothesis,
    PrivacyPolicy,
    RCSBCoordinatePolicy,
    ResearchCase,
    ResearchMode,
    TargetSpec,
)
from protbind_agent.structure import inspect_structure
from protbind_agent.structure_resolver import (
    HTTPResult,
    ResolutionDecision,
    StructureResolutionError,
    StructureResolver,
    UrllibHTTPTransport,
)
from protbind_agent.tripharm import build_jsonl_index
from protbind_agent.workflow import PipelineStageError, ProtBindWorkflow

gemmi = pytest.importorskip("gemmi")

_THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
}


class FakeTransport:
    def __init__(self, responses: dict[tuple[str, str], HTTPResult]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def request(self, method, url, *, body, headers, max_bytes):  # noqa: ANN001
        self.calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "headers": dict(headers),
                "max_bytes": max_bytes,
            }
        )
        result = self.responses[(method, url)]
        if len(result.data) > max_bytes:
            raise ValueError("fixture exceeds limit")
        return result


def _mmcif(
    chains: tuple[tuple[str, str], ...],
    *,
    metal: bool = False,
    covalent: bool = False,
    nonfinite: bool = False,
    altloc: bool = False,
) -> bytes:
    structure = gemmi.Structure()
    structure.name = "TEST"
    model = gemmi.Model("1")
    coordinate = 0.0
    for chain_id, sequence in chains:
        chain = gemmi.Chain(chain_id)
        for residue_number, code in enumerate(sequence, start=1):
            residue = gemmi.Residue()
            residue.name = _THREE_LETTER[code]
            residue.seqid = gemmi.SeqId(residue_number, " ")
            for atom_name, element in (("N", "N"), ("CA", "C"), ("C", "C")):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(element)
                atom.pos = gemmi.Position(
                    float("nan") if nonfinite and coordinate == 0.0 else coordinate,
                    0.0,
                    0.0,
                )
                if altloc and coordinate == 0.0:
                    atom.altloc = "A"
                coordinate += 1.2
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    if metal:
        chain = gemmi.Chain("M")
        residue = gemmi.Residue()
        residue.name = "ZN"
        residue.seqid = gemmi.SeqId(1, " ")
        atom = gemmi.Atom()
        atom.name = "ZN"
        atom.element = gemmi.Element("Zn")
        atom.pos = gemmi.Position(0.0, 5.0, 0.0)
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    document = structure.make_mmcif_document()
    document.sole_block().set_pair(
        "_pdbx_audit_revision_history.revision_date", "2026-07-01"
    )
    document.sole_block().set_pair("_exptl.method", "'X-RAY DIFFRACTION'")
    document.sole_block().set_pair("_refine.ls_d_res_high", "1.8")
    if covalent:
        document.sole_block().set_pair("_struct_conn.conn_type_id", "covale")
    return document.as_string().encode()


def _pdb(chains: tuple[tuple[str, str], ...]) -> bytes:
    document = gemmi.cif.read_string(_mmcif(chains).decode("utf-8"))
    structure = gemmi.make_structure_from_block(document.sole_block())
    return structure.make_pdb_string().encode("utf-8")


def _valid_screen_inputs(tmp_path, store: ArtifactStore):
    features = [
        {"type": "Donor", "position": [0.0, 0.0, 0.0], "atom_indices": [0]},
        {"type": "Acceptor", "position": [3.0, 0.0, 0.0], "atom_indices": [1]},
        {"type": "Aromatic", "position": [0.0, 4.0, 0.0], "atom_indices": [2]},
    ]
    pharmacophore = store.put_json(
        {"features": features}, producer="fixture-query"
    )
    records = tmp_path / "library.jsonl"
    records.write_text(
        json.dumps(
            {
                "molecule_id": "mol-a",
                "smiles": "CCO",
                "conformers": [{"id": 0, "features": features}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    build_jsonl_index(records, index_path)
    return pharmacophore, index_path


def _resolver(tmp_path, transport) -> StructureResolver:  # noqa: ANN001
    return StructureResolver(
        tmp_path / "workspace",
        transport=transport,
        now=lambda: datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
    )


def test_urllib_transport_disables_ambient_proxy_configuration(monkeypatch) -> None:
    captured: list[object] = []

    class Response:
        headers = {"Content-Length": "2"}

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *args):  # noqa: ANN002, ANN204
            return None

        def read(self, size):  # noqa: ANN001
            assert size == 3
            return b"ok"

    class Opener:
        def open(self, request, timeout):  # noqa: ANN001
            assert request.full_url == "https://files.rcsb.org/download/1abc.cif"
            assert timeout == 60
            return Response()

    def build_opener(*handlers):  # noqa: ANN001
        captured.extend(handlers)
        return Opener()

    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy.invalid:8080")
    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    result = UrllibHTTPTransport().request(
        "GET",
        "https://files.rcsb.org/download/1abc.cif",
        body=None,
        headers={},
        max_bytes=2,
    )

    proxy_handlers = [
        handler for handler in captured if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
    assert result.data == b"ok"


def test_user_structure_then_exact_sequence_cache_never_calls_network(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    structure = store.put_bytes(
        _mmcif((("A", "AC"),)),
        media_type="chemical/x-mmcif",
        producer="fixture-user",
    )
    transport = FakeTransport({})
    resolver = _resolver(tmp_path, transport)

    supplied = resolver.resolve(
        TargetSpec(name="private target", sequences=("AC",), structure=structure),
        PrivacyPolicy(),
    )
    cached = resolver.resolve(
        TargetSpec(name="private target", sequences=("AC",)),
        PrivacyPolicy(),
    )

    assert supplied.decision is ResolutionDecision.USER_SUPPLIED
    assert cached.decision is ResolutionDecision.LOCAL_EXACT_CACHE
    assert cached.structure == structure
    assert transport.calls == []
    receipt = store.read_json(cached.receipt)
    assert receipt["sequence_identity"]["sha256"]
    assert "AC" not in json.dumps(receipt)


def test_explicit_pdb_without_exact_domain_approval_falls_back_offline(tmp_path) -> None:
    transport = FakeTransport({})
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(name="target", sequences=("AC",), pdb_id="1ABC"),
        PrivacyPolicy(),
    )

    assert result.decision is ResolutionDecision.FOLDING_REQUIRED
    assert transport.calls == []
    receipt = resolver.artifacts.read_json(result.receipt)
    assert receipt["required_domains"] == ["files.rcsb.org"]
    assert receipt["network_requests"] == []


def test_explicit_pdb_import_selects_unique_exact_chain_and_records_provenance(
    tmp_path,
) -> None:
    url = "https://files.rcsb.org/download/1abc.cif"
    original = _mmcif((("A", "AC"), ("B", "GG")))
    transport = FakeTransport(
        {
            ("GET", url): HTTPResult(
                original,
                {
                    "ETag": '"revision-7"',
                    "Last-Modified": "Wed, 01 Jul 2026 00:00:00 GMT",
                },
            )
        }
    )
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(name="target", sequences=("AC",), pdb_id="1abc"),
        PrivacyPolicy(
            network_allowed=True,
            approved_domains=("files.rcsb.org",),
        ),
    )

    assert result.decision is ResolutionDecision.RCSB_IMPORTED
    assert result.structure is not None
    assert result.structure.source == url
    assert result.structure.license == "CC0-1.0"
    assert result.raw_source is not None
    assert resolver.artifacts.read_bytes(result.raw_source) == original
    assert result.raw_source.producer == "protbind.rcsb-download"
    assert result.raw_source != result.structure
    assert inspect_structure(resolver.artifacts.resolve(result.structure)).chain_ids == ("A",)
    receipt = resolver.artifacts.read_json(result.receipt)
    assert receipt["pdb_id"] == "1ABC"
    assert receipt["selected_chain_ids"] == ["A"]
    assert receipt["discarded_protein_chain_ids"] == ["B"]
    assert receipt["sequence_match"] == "exact"
    assert receipt["archive_revision"] == "2026-07-01"
    assert receipt["experimental_methods"] == ["X-RAY DIFFRACTION"]
    assert receipt["resolution_angstrom"] == 1.8
    assert receipt["retrieved_at"] == "2026-07-21T09:00:00+00:00"
    assert receipt["retrieval_headers"]["etag"] == '"revision-7"'
    assert receipt["raw_source_artifact"] == result.raw_source.to_dict()
    assert receipt["source_artifact"] == result.raw_source.to_dict()
    assert receipt["selected_receptor_artifact"] == result.structure.to_dict()
    assert receipt["coordinate_file_policy"] == "deposited_asymmetric_unit"
    assert receipt["assembly_id"] is None
    cached = resolver.resolve(
        TargetSpec(name="target", sequences=("AC",)), PrivacyPolicy()
    )
    cached_receipt = resolver.artifacts.read_json(cached.receipt)
    assert cached_receipt["cache_source_metadata"]["pdb_id"] == "1ABC"
    assert cached_receipt["cache_source_metadata"]["retrieved_at"] == (
        "2026-07-21T09:00:00+00:00"
    )
    assert cached.raw_source == result.raw_source
    assert cached_receipt["raw_source_artifact"] == result.raw_source.to_dict()
    assert len(transport.calls) == 1


def test_explicit_chain_order_is_preserved_in_selected_receptor(tmp_path) -> None:
    url = "https://files.rcsb.org/download/2abc.cif"
    transport = FakeTransport(
        {("GET", url): HTTPResult(_mmcif((("A", "AC"), ("B", "GG"))), {})}
    )
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(
            name="two-chain target",
            sequences=("GG", "AC"),
            pdb_id="2ABC",
            rcsb_chain_ids=("B", "A"),
        ),
        PrivacyPolicy(network_allowed=True, approved_domains=("files.rcsb.org",)),
    )

    assert result.structure is not None
    inspection = inspect_structure(resolver.artifacts.resolve(result.structure))
    assert inspection.chain_ids == ("B", "A")
    assert inspection.sequences == ("GG", "AC")


def test_explicit_chain_identity_is_part_of_the_rcsb_cache_scope(tmp_path) -> None:
    url = "https://files.rcsb.org/download/8abc.cif"
    transport = FakeTransport(
        {("GET", url): HTTPResult(_mmcif((("A", "AC"), ("B", "AC"))), {})}
    )
    resolver = _resolver(tmp_path, transport)
    online = PrivacyPolicy(
        network_allowed=True,
        approved_domains=("files.rcsb.org",),
    )
    target_a = TargetSpec(
        name="target", sequences=("AC",), pdb_id="8ABC", rcsb_chain_ids=("A",)
    )
    target_b = TargetSpec(
        name="target", sequences=("AC",), pdb_id="8ABC", rcsb_chain_ids=("B",)
    )

    imported_a = resolver.resolve(target_a, online)
    uncached_b = resolver.resolve(target_b, PrivacyPolicy())
    imported_b = resolver.resolve(target_b, online)
    cached_a = resolver.resolve(target_a, PrivacyPolicy())
    cached_b = resolver.resolve(target_b, PrivacyPolicy())

    assert imported_a.decision is ResolutionDecision.RCSB_IMPORTED
    assert uncached_b.decision is ResolutionDecision.FOLDING_REQUIRED
    assert imported_b.decision is ResolutionDecision.RCSB_IMPORTED
    assert cached_a.decision is ResolutionDecision.LOCAL_EXACT_CACHE
    assert cached_b.decision is ResolutionDecision.LOCAL_EXACT_CACHE
    assert cached_a.structure is not None and cached_b.structure is not None
    assert inspect_structure(resolver.artifacts.resolve(cached_a.structure)).chain_ids == (
        "A",
    )
    assert inspect_structure(resolver.artifacts.resolve(cached_b.structure)).chain_ids == (
        "B",
    )
    assert [call["url"] for call in transport.calls] == [url, url]


def test_user_mmcif_declaring_covalent_connection_is_rejected_before_cache(
    tmp_path,
) -> None:
    resolver = _resolver(tmp_path, FakeTransport({}))
    structure = resolver.artifacts.put_bytes(
        _mmcif((("A", "AC"),), covalent=True),
        media_type="chemical/x-mmcif",
        producer="fixture-user",
    )

    with pytest.raises(StructureResolutionError, match="covalent"):
        resolver.resolve(
            TargetSpec(name="target", sequences=("AC",), structure=structure),
            PrivacyPolicy(),
        )

    assert not resolver.cache_path.exists()


def test_corrupt_matching_exact_cache_fails_closed_without_sequence_upload(
    tmp_path,
) -> None:
    transport = FakeTransport({})
    resolver = _resolver(tmp_path, transport)
    structure = resolver.artifacts.put_bytes(
        _mmcif((("A", "AC"),)),
        media_type="chemical/x-mmcif",
        producer="fixture-user",
    )
    resolver.resolve(
        TargetSpec(name="target", sequences=("AC",), structure=structure),
        PrivacyPolicy(),
    )
    cache = json.loads(resolver.cache_path.read_text(encoding="utf-8"))
    entry = next(iter(cache["entries"].values()))
    entry["artifact"]["sha256"] = "f" * 64
    resolver.cache_path.write_text(json.dumps(cache), encoding="utf-8")

    with pytest.raises(StructureResolutionError, match="cache entry"):
        resolver.resolve(
            TargetSpec(name="target", sequences=("AC",)),
            PrivacyPolicy(
                network_allowed=True,
                approved_domains=("search.rcsb.org", "files.rcsb.org"),
                sequence_upload_allowed=True,
            ),
        )

    assert transport.calls == []


def test_biological_assembly_is_explicit_and_has_distinct_url_receipt_and_cache(
    tmp_path,
) -> None:
    asu_url = "https://files.rcsb.org/download/6abc.cif"
    assembly_url = "https://files.rcsb.org/download/6abc-assembly2.cif"
    transport = FakeTransport(
        {
            ("GET", asu_url): HTTPResult(_mmcif((("A", "AC"),)), {}),
            ("GET", assembly_url): HTTPResult(_mmcif((("A", "AC"),)), {}),
        }
    )
    resolver = _resolver(tmp_path, transport)
    privacy = PrivacyPolicy(
        network_allowed=True,
        approved_domains=("files.rcsb.org",),
    )

    asu = resolver.resolve(
        TargetSpec(name="target", sequences=("AC",), pdb_id="6ABC"),
        privacy,
    )
    assembly_target = TargetSpec(
        name="target",
        sequences=("AC",),
        pdb_id="6ABC",
        rcsb_coordinate_policy=RCSBCoordinatePolicy.BIOLOGICAL_ASSEMBLY,
        rcsb_assembly_id="2",
    )
    assembly = resolver.resolve(assembly_target, privacy)

    assert asu.raw_source is not None and asu.raw_source.source == asu_url
    assert assembly.raw_source is not None
    assert assembly.raw_source.source == assembly_url
    receipt = resolver.artifacts.read_json(assembly.receipt)
    assert receipt["coordinate_file_policy"] == "biological_assembly"
    assert receipt["assembly_id"] == "2"
    assert receipt["network_requests"][-1]["url"] == assembly_url
    assert [call["url"] for call in transport.calls] == [asu_url, assembly_url]

    cached_asu = resolver.resolve(
        TargetSpec(name="target", sequences=("AC",), pdb_id="6ABC"),
        PrivacyPolicy(),
    )
    cached_assembly = resolver.resolve(assembly_target, PrivacyPolicy())
    assert cached_asu.decision is ResolutionDecision.LOCAL_EXACT_CACHE
    assert cached_assembly.decision is ResolutionDecision.LOCAL_EXACT_CACHE
    assert cached_asu.raw_source == asu.raw_source
    assert cached_assembly.raw_source == assembly.raw_source
    assert len(transport.calls) == 2


def test_target_rejects_implicit_or_malformed_biological_assembly_policy() -> None:
    with pytest.raises(ValueError, match="requires an explicit pdb_id"):
        TargetSpec(
            name="target",
            sequences=("AC",),
            rcsb_coordinate_policy=RCSBCoordinatePolicy.BIOLOGICAL_ASSEMBLY,
            rcsb_assembly_id="1",
        )
    with pytest.raises(ValueError, match="positive decimal"):
        TargetSpec(
            name="target",
            sequences=("AC",),
            pdb_id="6ABC",
            rcsb_coordinate_policy=RCSBCoordinatePolicy.BIOLOGICAL_ASSEMBLY,
            rcsb_assembly_id="../1",
        )
    with pytest.raises(ValueError, match="requires rcsb_coordinate_policy"):
        TargetSpec(
            name="target",
            sequences=("AC",),
            pdb_id="6ABC",
            rcsb_assembly_id="1",
        )


def test_cli_exposes_explicit_biological_assembly_id() -> None:
    args = _build_parser().parse_args(
        [
            "case",
            "run",
            "--case",
            "case.json",
            "--index",
            "library.sqlite",
            "--rcsb-pdb-id",
            "6ABC",
            "--rcsb-assembly-id",
            "2",
        ]
    )

    assert args.rcsb_pdb_id == "6ABC"
    assert args.rcsb_assembly_id == "2"


@pytest.mark.parametrize(
    ("coordinate_data", "reason_code"),
    [
        (_mmcif((("A", "AC"), ("B", "AC"))), "AMBIGUOUS_CHAIN_ASSIGNMENT"),
        (_mmcif((("A", "AC"),), metal=True), "METAL_STRUCTURE_REJECTED"),
        (_mmcif((("A", "AC"),), covalent=True), "COVALENT_STRUCTURE_REJECTED"),
        (_mmcif((("A", "AC"),), nonfinite=True), "NON_FINITE_COORDINATES"),
        (_mmcif((("A", "AC"),), altloc=True), "UNRESOLVED_ALTLOC"),
        (_mmcif((("A", "GG"),)), "SEQUENCE_MISMATCH"),
    ],
    ids=(
        "ambiguous-multichain",
        "metal",
        "covalent",
        "nonfinite",
        "altloc",
        "sequence-mismatch",
    ),
)
def test_unsafe_rcsb_candidates_are_rejected_not_silently_accepted(
    tmp_path, coordinate_data, reason_code
) -> None:
    url = "https://files.rcsb.org/download/3abc.cif"
    transport = FakeTransport({("GET", url): HTTPResult(coordinate_data, {})})
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(name="target", sequences=("AC",), pdb_id="3ABC"),
        PrivacyPolicy(network_allowed=True, approved_domains=("files.rcsb.org",)),
    )

    assert result.decision is ResolutionDecision.FOLDING_REQUIRED
    assert result.structure is None
    receipt = resolver.artifacts.read_json(result.receipt)
    assert len(receipt["candidate_attempts"]) == 1
    attempt = receipt["candidate_attempts"][0]
    assert attempt["accepted"] is False
    assert attempt["pdb_id"] == "3ABC"
    assert attempt["reason_code"] == reason_code
    raw_source = attempt["raw_source_artifact"]
    raw_ref = ArtifactRef.from_dict(raw_source)
    assert resolver.artifacts.read_bytes(raw_ref) == coordinate_data


def test_sequence_search_requires_separate_upload_gate(tmp_path) -> None:
    transport = FakeTransport({})
    resolver = _resolver(tmp_path, transport)
    policy = PrivacyPolicy(
        network_allowed=True,
        approved_domains=("search.rcsb.org", "files.rcsb.org"),
    )

    result = resolver.resolve(TargetSpec(name="private", sequences=("AC",)), policy)

    assert result.decision is ResolutionDecision.FOLDING_REQUIRED
    assert transport.calls == []


def test_approved_sequence_search_uploads_only_after_gate(tmp_path) -> None:
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    file_url = "https://files.rcsb.org/download/4abc.cif"
    transport = FakeTransport(
        {
            ("POST", search_url): HTTPResult(
                b'{"result_set":[{"identifier":"4ABC_1"}]}', {}
            ),
            ("GET", file_url): HTTPResult(_mmcif((("A", "AC"),)), {}),
        }
    )
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(name="private", sequences=("AC",)),
        PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
            sequence_upload_allowed=True,
        ),
    )

    assert result.decision is ResolutionDecision.RCSB_IMPORTED
    payload = json.loads(transport.calls[0]["body"])
    assert payload["query"]["service"] == "sequence"
    assert payload["query"]["parameters"]["value"] == "AC"
    receipt = resolver.artifacts.read_json(result.receipt)
    assert receipt["network_requests"][0]["sequence_uploaded"] is True


def test_uniprot_discovery_does_not_upload_target_sequence(tmp_path) -> None:
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    file_url = "https://files.rcsb.org/download/5abc.cif"
    transport = FakeTransport(
        {
            ("POST", search_url): HTTPResult(
                b'{"result_set":[{"identifier":"5ABC_2"}]}', {}
            ),
            ("GET", file_url): HTTPResult(_mmcif((("A", "AC"),)), {}),
        }
    )
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(
            name="private",
            sequences=("AC",),
            uniprot_accession="P12345",
        ),
        PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
        ),
    )

    assert result.decision is ResolutionDecision.RCSB_IMPORTED
    payload = json.loads(transport.calls[0]["body"])
    assert payload["query"]["nodes"][0]["parameters"]["value"] == "P12345"
    assert all(node["service"] == "text" for node in payload["query"]["nodes"])
    receipt = resolver.artifacts.read_json(result.receipt)
    assert receipt["network_requests"][0]["sequence_uploaded"] is False


def test_empty_rcsb_search_response_is_a_normal_no_hit_outcome(tmp_path) -> None:
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    transport = FakeTransport({("POST", search_url): HTTPResult(b"", {})})
    resolver = _resolver(tmp_path, transport)

    result = resolver.resolve(
        TargetSpec(
            name="private",
            sequences=("AC",),
            uniprot_accession="P12345",
        ),
        PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
        ),
    )

    assert result.decision is ResolutionDecision.FOLDING_REQUIRED
    receipt = resolver.artifacts.read_json(result.receipt)
    assert receipt["candidate_attempts"] == []
    assert "discovery failed" not in receipt["reason"]


def test_workflow_binds_cached_receptor_and_resolution_receipt_before_input_stage(
    tmp_path,
) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    structure = store.put_bytes(
        _mmcif((("A", "AC"),)),
        media_type="chemical/x-mmcif",
        producer="fixture-local-fold",
    )
    StructureResolver(workspace, artifacts=store).register(
        structure,
        ("AC",),
        origin="fixture_local_fold",
    )
    pharmacophore, index_path = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="cached-receptor",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
    )
    workflow = ProtBindWorkflow(workspace)

    manifest = workflow.create(case, index_path, run_id="cached-receptor-run")
    resolved_case = workflow.load_case(manifest)
    resolution = store.read_json(manifest.input_artifacts["target_resolution"])
    manifest = workflow.run(manifest, stop_after=RunState.INPUT_VALIDATED)

    assert resolved_case.target.structure == structure
    assert resolution["decision"] == "local_exact_sequence_cache"
    assert manifest.provenance["target_structure_resolution"] == (
        "local_exact_sequence_cache"
    )
    assert manifest.state is RunState.INPUT_VALIDATED
    validation = store.read_json(
        manifest.stage_records[RunState.INPUT_VALIDATED.value].outputs[0]
    )
    assert validation["structure_resolution"]["folding_required"] is False


def test_public_register_rejects_forged_explicit_rcsb_chain_scope(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    structure = store.put_bytes(
        _mmcif((("B", "AC"),)),
        media_type="chemical/x-mmcif",
        producer="fixture-import",
    )

    with pytest.raises(StructureResolutionError, match="chain metadata differs"):
        StructureResolver(workspace, artifacts=store).register(
            structure,
            ("AC",),
            origin="forged-rcsb-import",
            source_metadata={
                "pdb_id": "1ABC",
                "coordinate_file_policy": "deposited_asymmetric_unit",
                "assembly_id": None,
                "requested_chain_ids": ["A"],
                "selected_chain_ids": ["A"],
                "chain_selection_policy": "explicit",
                "selected_receptor_artifact": structure.to_dict(),
            },
        )


def test_workflow_rejects_resolution_receipt_with_wrong_selected_receptor(
    tmp_path,
) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    structure = store.put_bytes(
        _mmcif((("A", "AC"),)),
        media_type="chemical/x-mmcif",
        producer="fixture-local-fold",
    )
    StructureResolver(workspace, artifacts=store).register(
        structure, ("AC",), origin="fixture_local_fold"
    )
    pharmacophore, index_path = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="forged-resolution",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
    )
    workflow = ProtBindWorkflow(workspace)
    manifest = workflow.create(case, index_path, run_id="forged-resolution-run")
    receipt = store.read_json(manifest.input_artifacts["target_resolution"])
    wrong_structure = store.put_bytes(
        _mmcif((("X", "AC"),)),
        media_type="chemical/x-mmcif",
        producer="fixture-other-fold",
    )
    receipt["selected_receptor_artifact"] = wrong_structure.to_dict()
    forged = store.put_json(
        receipt,
        producer="protbind.structure-resolver",
        producer_version=manifest.input_artifacts[
            "target_resolution"
        ].producer_version,
    )
    manifest.input_artifacts["target_resolution"] = forged

    with pytest.raises(PipelineStageError, match="differs from the frozen"):
        workflow._validate_resolution_receipt(manifest, workflow.load_case(manifest))


def test_workflow_binds_rcsb_raw_download_as_hashed_input(tmp_path) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    pharmacophore, index_path = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="rcsb-raw-source",
        target=TargetSpec(name="public", sequences=("AC",), pdb_id="7ABC"),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
        privacy=PrivacyPolicy(
            network_allowed=True,
            approved_domains=("files.rcsb.org",),
        ),
    )
    url = "https://files.rcsb.org/download/7abc.cif"
    original = _mmcif((("A", "AC"), ("B", "GG")))
    transport = FakeTransport({("GET", url): HTTPResult(original, {})})
    workflow = ProtBindWorkflow(workspace)
    workflow.structure_resolver = StructureResolver(
        workspace,
        artifacts=workflow.artifacts,
        transport=transport,
    )
    manifest = workflow.create(case, index_path, run_id="rcsb-raw-source-run")
    raw_source = manifest.input_artifacts["target_raw_source"]
    receipt = store.read_json(manifest.input_artifacts["target_resolution"])
    manifest = workflow.run(manifest, stop_after=RunState.INPUT_VALIDATED)

    assert store.read_bytes(raw_source) == original
    assert receipt["raw_source_artifact"] == raw_source.to_dict()
    validation = store.read_json(
        manifest.stage_records[RunState.INPUT_VALIDATED.value].outputs[0]
    )
    assert validation["structure_resolution"]["raw_source_artifact_id"] == (
        raw_source.artifact_id
    )


def test_workflow_fails_local_index_gate_before_approved_sequence_upload(
    tmp_path,
) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    pharmacophore, _ = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="local-gate-before-network",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
        privacy=PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
            sequence_upload_allowed=True,
        ),
    )
    transport = FakeTransport({})
    workflow = ProtBindWorkflow(workspace)
    workflow.structure_resolver = StructureResolver(
        workspace,
        artifacts=workflow.artifacts,
        transport=transport,
    )

    with pytest.raises(FileNotFoundError, match="library index does not exist"):
        workflow.create(case, tmp_path / "missing-index.sqlite")

    assert transport.calls == []


def test_workflow_rejects_corrupt_index_before_approved_sequence_upload(
    tmp_path,
) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    pharmacophore, _ = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="corrupt-index-before-network",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
        privacy=PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
            sequence_upload_allowed=True,
        ),
    )
    corrupt_index = tmp_path / "corrupt.sqlite"
    corrupt_index.write_bytes(b"not a sqlite index")
    transport = FakeTransport({})
    workflow = ProtBindWorkflow(workspace)
    workflow.structure_resolver = StructureResolver(
        workspace, artifacts=workflow.artifacts, transport=transport
    )

    with pytest.raises(ValueError, match="valid TriPharm index"):
        workflow.create(case, corrupt_index)

    assert transport.calls == []


def test_workflow_rejects_invalid_pharmacophore_before_sequence_upload(
    tmp_path,
) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    _, index_path = _valid_screen_inputs(tmp_path, store)
    pharmacophore = store.put_json(
        {
            "features": [
                {"type": "Donor", "position": [0.0, 0.0, 0.0]},
                {"type": "Acceptor", "position": [3.0, 0.0, 0.0]},
            ]
        },
        producer="invalid-query",
    )
    case = ResearchCase(
        case_id="bad-query-before-network",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
        privacy=PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
            sequence_upload_allowed=True,
        ),
    )
    transport = FakeTransport({})
    workflow = ProtBindWorkflow(workspace)
    workflow.structure_resolver = StructureResolver(
        workspace, artifacts=workflow.artifacts, transport=transport
    )

    with pytest.raises(PipelineStageError, match="requires 3"):
        workflow.create(case, index_path)

    assert transport.calls == []


def test_workflow_rejects_invalid_smiles_before_sequence_upload(tmp_path) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    _, index_path = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="bad-smiles-before-network",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles="not-a-smiles"),
        privacy=PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
            sequence_upload_allowed=True,
        ),
    )
    transport = FakeTransport({})
    workflow = ProtBindWorkflow(workspace)
    workflow.structure_resolver = StructureResolver(
        workspace, artifacts=workflow.artifacts, transport=transport
    )

    with pytest.raises((ChemistryCapabilityError, ValueError, PipelineStageError)):
        workflow.create(case, index_path)

    assert transport.calls == []


@pytest.mark.parametrize("smiles", ["[Zn]", "CC(F)Cl"])
def test_workflow_rejects_metal_or_unassigned_stereo_before_sequence_upload(
    tmp_path, smiles: str
) -> None:
    pytest.importorskip("rdkit")
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    _, index_path = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="bad-chemistry-before-network",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(smiles=smiles),
        privacy=PrivacyPolicy(
            network_allowed=True,
            approved_domains=("search.rcsb.org", "files.rcsb.org"),
            sequence_upload_allowed=True,
        ),
    )
    transport = FakeTransport({})
    workflow = ProtBindWorkflow(workspace)
    workflow.structure_resolver = StructureResolver(
        workspace, artifacts=workflow.artifacts, transport=transport
    )

    with pytest.raises(PipelineStageError):
        workflow.create(case, index_path)

    assert transport.calls == []


def test_workflow_attaches_esmfold_only_through_provenance_receipt(tmp_path) -> None:
    workspace = tmp_path / "workflow"
    store = ArtifactStore(workspace)
    pharmacophore, index_path = _valid_screen_inputs(tmp_path, store)
    case = ResearchCase(
        case_id="esmfold-receipt",
        target=TargetSpec(name="private", sequences=("AC",)),
        mode=ResearchMode.LIGAND_ONLY,
        ligand=LigandHypothesis(pharmacophore=pharmacophore),
    )
    workflow = ProtBindWorkflow(workspace)
    manifest = workflow.create(case, index_path, run_id="esmfold-receipt-run")
    esmfold_input = store.put_json(
        {"sequences": ["AC"]}, producer="protbind.esmfold-v1.smoke-input"
    )
    structure = store.put_bytes(
        _pdb((("A", "AC"),)),
        media_type="chemical/x-pdb",
        producer="fair-esm.esmfold_v1",
        producer_version="2.0.0",
        source=esmfold_input.artifact_id,
    )
    metadata = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.esmfold-v1-result",
            "input": esmfold_input.to_dict(),
            "structure": structure.to_dict(),
            "seed": case.seed,
            "model_revision": "esmfold_3B_v1",
            "weight_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "runtime": {
                "fair_esm_version": "2.0.0",
                "environment_lock_sha256": "c" * 64,
                "runtime_source_sha256": "d" * 64,
            },
            "output_qc": {
                "sequence_identity_sha256": [sha256_bytes(b"AC")],
                "coordinate_finite": True,
                "backbone_complete": True,
                "alternate_locations": False,
            },
        },
        producer="protbind.esmfold-v1-result",
        producer_version="2.0.0",
        source=structure.artifact_id,
    )
    receipt_value = {
        "schema_version": "1.0",
        "kind": "protbind.esmfold-v1-smoke-receipt",
        "job_id": "esmfold-offline-smoke",
        "engine": "esmfold_v1",
        "success": True,
        "input": esmfold_input.to_dict(),
        "sequence_identity_sha256": [sha256_bytes(b"AC")],
        "outputs": [structure.to_dict(), metadata.to_dict()],
        "provenance": {
            "model_revision": "esmfold_3B_v1",
            "weight_sha256": "a" * 64,
            "code_sha256": "b" * 64,
        },
        "hardware_sha256": "e" * 64,
        "error": None,
    }
    receipt_path = tmp_path / "esmfold-receipt.json"
    receipt_path.write_text(json.dumps(receipt_value), encoding="utf-8")

    with pytest.raises(ValueError, match="verified esmfold_receipt"):
        workflow.attach_support(
            manifest,
            "esmfold_structure",
            store.resolve(structure),
            media_type="chemical/x-pdb",
        )
    receipt = workflow.attach_support(
        manifest,
        "esmfold_receipt",
        receipt_path,
        media_type="application/json",
    )

    assert manifest.artifacts["support_esmfold_receipt"] == receipt
    assert manifest.artifacts["support_esmfold_structure"] == structure
    assert manifest.artifacts["support_esmfold_result_metadata"] == metadata
    cached = workflow.structure_resolver.resolve(
        TargetSpec(name="private", sequences=("AC",)), PrivacyPolicy()
    )
    assert cached.structure == structure
    cached_receipt = store.read_json(cached.receipt)
    assert cached_receipt["cache_source_metadata"]["weight_sha256"] == "a" * 64
