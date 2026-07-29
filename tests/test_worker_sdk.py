from __future__ import annotations

import sys
from pathlib import Path

from protbind_agent.artifacts import ArtifactStore
from protbind_agent.worker_protocol import (
    JsonSubprocessWorker,
    WorkerProvenance,
    WorkerRequest,
)


def test_worker_sdk_keeps_third_party_stdout_out_of_json_protocol(tmp_path) -> None:
    source_root = Path(__file__).parents[1] / "src"
    script = tmp_path / "noisy_worker.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from protbind_agent.worker_protocol import WorkerResponse\n"
        "from protbind_agent.worker_sdk import serve_worker\n"
        "def handler(request, store):\n"
        " print('third-party progress that must not reach stdout')\n"
        " output=store.put_json({'ok': True}, producer='noisy-fixture')\n"
        " return WorkerResponse(job_id=request.job_id, engine=request.engine, "
        "outputs=(output,), provenance=request.provenance)\n"
        "raise SystemExit(serve_worker('noisy-fixture', handler))\n",
        encoding="utf-8",
    )
    store = ArtifactStore(tmp_path / "workspace")
    input_artifact = store.put_json({"input": True}, producer="test")
    request = WorkerRequest(
        job_id="noisy-worker-contract",
        engine="noisy-fixture",
        input=input_artifact,
        parameters={},
        seed=1,
        provenance=WorkerProvenance(
            model_revision="fixture-only",
            weight_sha256="a" * 64,
            code_sha256="b" * 64,
        ),
    )

    response, _ = JsonSubprocessWorker(
        (sys.executable, str(script)), artifact_root=store.root
    ).run(request)

    assert response.error is None
    assert store.read_json(response.outputs[0]) == {"ok": True}
