#!/usr/bin/env python3
"""Run a path-redacted public 1CRN quick-Vina protocol smoke in AIAA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
WORKER_ROOT = REPOSITORY_ROOT / "workers"
sys.path.insert(0, str(WORKER_ROOT))

import quick_vina_worker as quick  # noqa: E402

from protbind_agent.artifacts import ArtifactStore  # noqa: E402
from protbind_agent.models import ArtifactRef, SiteProvenanceKind  # noqa: E402
from protbind_agent.selection import (  # noqa: E402
    build_quick_vina_input,
    build_selection_preparation,
    validate_docking_box_receipt,
    validate_quick_vina_batch,
)
from protbind_agent.worker_protocol import (  # noqa: E402
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = ArtifactStore(args.workspace)
    receptor = store.import_file(
        args.receptor,
        media_type="chemical/x-pdb",
        producer="protbind.public-smoke-input",
        producer_version="1.0",
        source="pdb:1CRN",
        license="RCSB PDB data policy",
    )
    index = store.import_file(
        args.index,
        media_type="application/vnd.sqlite3",
        producer="protbind.fixture-index",
        producer_version="1.0",
        source="fixture:demo-index",
    )
    environment_lock = store.import_file(
        args.environment_lock,
        media_type="application/json",
        producer="protbind.aiaa-environment-audit",
        producer_version="1.0",
    )
    screen = store.put_json(
        {
            "schema_version": "1.0",
            "kind": "protbind.public-selection-smoke-ranking",
            "hits": [
                {"molecule_id": "demo-001"},
                {"molecule_id": "demo-002"},
            ],
            "semantics": "fixed protocol ordering; not a hit-rate experiment",
        },
        producer="protbind.public-selection-smoke",
        producer_version="1.0",
    )
    preparation = build_selection_preparation(
        store,
        screening=screen,
        library_index=index,
        receptor=receptor,
        protein_chains=(
            ("A", "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"),
        ),
        box_center=(10.5935, 10.2105, 6.079),
        box_size=(20.0, 20.0, 20.0),
        box_source=SiteProvenanceKind.USER_CENTER,
    )
    preparation_value = store.read_json(preparation)
    docking_box_receipt = ArtifactRef.from_dict(
        preparation_value["docking_box_receipt"]
    )
    docking_box_value = validate_docking_box_receipt(
        store,
        docking_box_receipt,
        receptor=receptor,
        center=(10.5935, 10.2105, 6.079),
        size=(20.0, 20.0, 20.0),
        source_kind=SiteProvenanceKind.USER_CENTER.value,
    )
    quick_input = build_quick_vina_input(
        store,
        preparation,
        environment_lock,
        case_id="public-1crn-automatic-selection-smoke",
    )
    parameters = {
        "vina_executable": str((REPOSITORY_ROOT / "tools/bin/vina").resolve()),
        "meeko_prepare_receptor_executable": str(
            (REPOSITORY_ROOT / ".venv-aiaa-protbind/bin/mk_prepare_receptor.py").resolve()
        ),
        "meeko_prepare_ligand_executable": str(
            (REPOSITORY_ROOT / ".venv-aiaa-protbind/bin/mk_prepare_ligand.py").resolve()
        ),
        "vina_version": quick.vina.VINA_VERSION,
        "meeko_version": quick.vina.MEEKO_VERSION,
        "rdkit_version": _version("rdkit"),
        "gemmi_version": _version("gemmi"),
        "numpy_version": _version("numpy"),
        "scipy_version": _version("scipy"),
        "scoring": "vina",
        "cpu": 1,
        "exhaustiveness": 8,
        "num_modes": 1,
        "energy_range": 3.0,
        "command_timeout_seconds": 900.0,
    }
    attestation = quick.vina.runtime_asset_attestation(parameters)
    assets = str(attestation["runtime_assets_sha256"])
    provenance = WorkerProvenance(
        model_revision=quick.quick_model_revision(parameters),
        weight_sha256=assets,
        code_sha256=quick.composite_code_sha256(environment_lock.sha256, assets),
    )
    request = WorkerRequest(
        job_id="public-1crn-quick-vina-smoke",
        engine=quick.ENGINE,
        input=quick_input,
        parameters=parameters,
        seed=20260721,
        provenance=provenance,
    )
    worker = JsonSubprocessWorker(
        (sys.executable, str(WORKER_ROOT / "quick_vina_worker.py")),
        timeout_seconds=1800.0,
        artifact_root=store.root,
        isolate_network=False,
    )
    response, elapsed = worker.run(request)
    if response.error is not None:
        raise RuntimeError(f"{response.error.code}: {response.error.message}")
    evaluations = validate_quick_vina_batch(
        store,
        preparation,
        quick_input,
        response.outputs,
        case_id="public-1crn-automatic-selection-smoke",
        seed=request.seed,
    )
    result = {
        "schema_version": "1.0",
        "kind": "protbind.aiaa-quick-vina-smoke-result",
        "scientific_scope": "protocol-and-environment-smoke-only",
        "inputs": {
            "receptor": receptor.to_dict(),
            "index": index.to_dict(),
            "environment_lock": environment_lock.to_dict(),
            "screening": screen.to_dict(),
            "preparation": preparation.to_dict(),
            "docking_box_receipt": docking_box_receipt.to_dict(),
            "quick_input": quick_input.to_dict(),
        },
        "docking_box_contract": {
            "coordinate_frame": docking_box_value["coordinate_frame"],
            "validation": docking_box_value["validation"],
        },
        "provenance": provenance.to_dict(),
        "worker_outputs": [item.to_dict() for item in response.outputs],
        "evaluations": evaluations,
        "timings_seconds": response.timings_seconds,
        "end_to_end_seconds": elapsed,
        "peak_vram_bytes": response.peak_vram_bytes,
        "warnings": list(response.warnings),
        "isolation": "application-offline-policy-only; direct adapter smoke",
        "limitations": [
            "1CRN/box is not an experimentally established ligand-binding site",
            "demo index has chemistry_verified=false and is fixture-only",
            "box receipt verifies protein-atom overlap only, not biological site derivation",
            "no binding, affinity, activity, enrichment, or hit-rate claim is supported",
            "production workflow still requires bubblewrap OS network isolation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
