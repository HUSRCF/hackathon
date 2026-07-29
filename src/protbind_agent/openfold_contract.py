"""Shared, immutable OpenFold3 production contract constants."""

from __future__ import annotations

OPENFOLD_ENGINE = "openfold3"
OPENFOLD_RUNTIME_ENGINE = "official-openfold3"
OPENFOLD_REVISION = "openfold3-0.4.3@0bb17be5199846e806b6347b6e17c6249c88ff1b"
OPENFOLD_VERSION = "0.4.3"
OPENFOLD_SCM_NODE = "g0bb17be5199846e806b6347b6e17c6249c88ff1b"
OPENFOLD_BUNDLE_PRODUCER = "protbind.openfold3.bundle"
OPENFOLD_QUERY_MANIFEST_PRODUCER = "protbind.openfold3.query-manifest"
OPENFOLD_RUNNER_PRODUCER = "protbind.openfold3.runner"
OPENFOLD_RUN_METADATA_PRODUCER = "protbind.openfold3.run-metadata"
OFFICIAL_RUNTIME_SHA256 = (
    "742e9bf654b13f67783d095a2327af3ed31163580eaa7b4c548e8a8eb2e68010"
)
OFFICIAL_RUNTIME_FILE_COUNT = 317
OFFICIAL_CHECKPOINT_SIZES = {
    "openfold3-p2-155k": 2_287_928_196,
}
