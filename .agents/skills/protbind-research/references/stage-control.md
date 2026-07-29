# Stage-control protocol

ProtBind advances through:

```text
CREATED -> INPUT_VALIDATED -> RECEPTOR_READY -> INDEXED -> SCREENED
        -> SELECTED -> DOCKED -> VALIDATED -> REPORTED
```

`COFOLDED` is optional side evidence. `DEGRADED` and `FAILED` are explicit non-success states.

## Required loop

```text
case_status
  -> preflight gate
  -> explain gate and required actions
  -> case_advance(fresh token)
  -> postflight acceptance
  -> next preflight gate
```

A continuation token binds the run ID, exact manifest SHA-256, next stage, and control-policy
SHA-256. Any manifest change makes it stale. Never reuse a token after an advance or support
attachment.

Each `case_advance` executes at most one main stage. Postflight acceptance re-audits the manifest,
stage/cache binding, configuration, and every declared output artifact. A content-addressed gate
or acceptance receipt is appended to the run's control history.

## Decisions

| Decision | Meaning | Required response |
|---|---|---|
| `READY` | Preflight passed | Advance once with the returned token after user approval |
| `ACCEPTED` | The attempted stage passed postflight | Treat only that stage as completed |
| `NEEDS_ACTION` | A named input/capability is missing | Stop, perform the stated bounded action, re-run status |
| `RETRYABLE` | A recoverable execution failure was recorded | Explain it; retry only explicitly with the fresh token |
| `UNSUPPORTED` | The chemistry/system is outside protocol | Stop; do not silently degrade |
| `FAILED` | Audit or nonrecoverable execution failed | Stop; restore exact inputs or create a revised run |
| `COMPLETE` | All main stages were accepted | Retrieve the report and control history |

Warnings do not erase an accepted result. Repeat them in the handoff and preserve their artifact
references. Never convert `NEEDS_ACTION`, `RETRYABLE`, or `DEGRADED` into success.

## Typical closure actions

- Missing receptor: attach a verified receptor artifact or accepted ESMFold-v1 receipt.
- Missing query: provide the ligand/pocket pharmacophore or bounded pocket hypothesis.
- Missing selection path: configure the pinned quick-Vina worker or attach a frozen selection
  batch.
- Missing docking/validation worker: restart the MCP service with a reviewed, hash-pinned worker
  configuration.
- Worker crash/OOM/GPU busy: preserve the failure record, fix the resource/runtime condition, then
  explicitly request a new gate.
- Artifact/configuration mismatch: do not patch the historical run; restore the exact bound bytes
  or start a new protocol revision.
