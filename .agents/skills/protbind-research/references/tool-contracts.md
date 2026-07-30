# MCP tool contracts

All file arguments are project-relative. The server resolves and bounds them to its configured
project root. It runs over stdio and exposes no arbitrary command, raw filesystem, URL, or open
network tool. Its one network-capable method is the bounded identifier-only public registry fetch
described below.

## Read-only tools

- `protbind_doctor()`: return redacted local capability and Radeon admission evidence.
- `protbind_case_status(run_id)`: deep-audit the run and return the current preflight gate.
- `protbind_case_report(run_id, format)`: return bounded `markdown`, `html`, or `degraded` report
  text when available; structure coordinates are never returned.
- `protbind_case_dossier(run_id, format)`: return the detailed current-stage completion,
  acceptance-receipt, timing, warning, failure, artifact, and pose-QA dossier as `json`,
  `markdown`, or `html`. Continuation tokens and coordinate bytes are never returned.
- `protbind_case_pose_view(run_id)`: return coordinate-free pose, validation, docking-box, and
  deterministic Gemmi/RDKit geometry metadata plus loopback viewer paths. It may create an
  immutable derived scene artifact but never changes the scientific manifest or stage state.
- `protbind_artifact_metadata(artifact_ref_json)`: verify a complete serialized `ArtifactRef` and
  return metadata only.
- `protbind_control_history(run_id)`: return content-addressed gate/acceptance receipt references.

## Mutating tools

- `protbind_fetch_public_data(source, identifier, project_path, approved_domain, run_propka=true,
  replace=false)`: download exactly one whitelisted public registry record using a constructed
  HTTPS URL and direct curl. The exact source domain must be explicitly approved. Output paths and
  source-specific suffixes are validated before network access; redirects, ambient proxies,
  credentials, private sequences, arbitrary URLs, and batch queries are unavailable. The tool
  writes the fetched bytes plus a provenance sidecar, but does not attach them to a run or advance
  a scientific stage. Gemmi/RDKit parse checks and optional PROPKA diagnostics are acquisition
  observations, not receptor or ligand acceptance.
- `protbind_case_create(case_path, index_path, run_id?)`: ingest an offline case plus frozen
  TriPharm index and return the initial gate. Nested `structure_file`, `pharmacophore_file`, and
  `site_derivation_source_files` must also be project-relative. Network-enabled privacy policies
  are rejected.
- `protbind_case_advance(run_id, continuation_token)`: preflight and execute exactly the gate's
  named main stage, then postflight it.
- `protbind_case_attach_support(run_id, name, project_path, media_type, replace=false)`: import one
  reviewed local file through core timing/freeze rules, then return a fresh gate.

Never guess an `ArtifactRef`, continuation token, run ID, MIME type, worker digest, or support
name. Obtain it from tool output or the user. Do not set `replace=true` unless the user explicitly
intends to replace an unfrozen support input. Do not call `protbind_fetch_public_data` until the
user has approved its exact domain.

## Failure handling

Report tool errors as control failures, not scientific results. If a tool rejects a path, ask the
user to move or copy the reviewed input under the project root. If a stage requires a worker not
configured at MCP startup, stop and have the operator restart the server with
`--worker-config <reviewed-project-relative-config>`.
