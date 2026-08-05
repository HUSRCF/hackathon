# HipFire Qwen3.6-27B counterbalanced tool-routing A/B — 2026-08-05

## Outcome

Qwen3.6-27B completed the requested `doctor` call in all six measured runs. Deterministic
tool routing reduced the exposed tool set from 25 to 2, schema bytes from 9,794 to 454,
and mean prompt tokens from 10,406.7 to 4,694.3. This is direct evidence that ProtBind's
router reduces model context while preserving one-tool execution on the tested workload.

This is an exploratory local Agent comparison, not a scientific protein-ligand benchmark
and not a general DFlash speedup claim. No private sequence, ligand, assay, or knowledge-base
content was sent to the model.

## Provenance

- Hardware: AMD Radeon Pro W7900, `gfx1100`, GPU 0.
- Model: `qwen3.6:27b`; target SHA-256
  `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc`.
- DFlash draft SHA-256:
  `bd8c4f07ae80fe1385bf2606af9a7ba0daa18ca8daec50916f2a489054c44e70`.
- Runtime: Asym3 VMM; DFlash draft loaded; `HIP_VISIBLE_DEVICES=0`.
- HipFire revision: `e2f7dd1a47be23873d72090ac778e4f5197b7fe2`, clean checkout.
- ProtBind revision: `e9f727597765c3ad29406d87aa5ce5ab78196871` before result logging.
- Design: one warm-up per condition, followed by ABBA/BA interleaving.
- Byte-identical request: `Call doctor exactly once. Then state whether it succeeded. Do not call any other tool.`

## Raw data

| Order | Mode | Success | Tools exposed | Schema bytes | Prompt tokens | Tool time (s) | End-to-end (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | routed | 1 | 2 | 454 | 4,693 | 0.527 | 9.643 |
| 2 | full | 1 | 25 | 9,794 | 10,406 | 0.527 | 32.675 |
| 3 | full | 1 | 25 | 9,794 | 10,407 | 0.546 | 28.941 |
| 4 | routed | 1 | 2 | 454 | 4,698 | 0.523 | 9.677 |
| 5 | full | 1 | 25 | 9,794 | 10,407 | 0.531 | 32.711 |
| 6 | routed | 1 | 2 | 454 | 4,692 | 0.530 | 16.434 |

| Mode | n | Success | Elapsed mean ± sample SD (s) | Elapsed median (s) | Mean prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed | 3 | 3/3 | 11.918 ± 3.911 | 9.677 | 4,694.3 |
| full | 3 | 3/3 | 31.442 ± 2.166 | 32.675 | 10,406.7 |

Exact structural deltas for this workload are 92.0% fewer exposed tools, 95.36% fewer
schema bytes, and 54.89% fewer prompt tokens. The observed elapsed reductions were 62.10%
by mean and 70.38% by median (2.64× and 3.38× respectively), but these latency ratios must
remain labelled exploratory.

## Interpretation and boundary

1. The Agent selected and executed exactly one `doctor` call in every run. Routing did not
   reduce functional success in this small workload.
2. Tool execution itself remained approximately 0.52–0.55 seconds in both conditions.
   Most elapsed difference came from the model's two schema-bearing inference calls.
3. The third routed run was a genuine long tail: its second inference showed DFlash
   `tau=1.29` and about 3.9 decode tokens/s in the daemon log, versus about 9.7 seconds for
   the first two routed runs. It must not be discarded.
4. The same-server design still contains prefix-cache, thermal, and DFlash-acceptance
   confounding. These results support the context-reduction claim, not a universal 2.64×
   Agent acceleration or a DFlash kernel-performance claim.
5. This run intentionally excluded `knowledge_search` and `memory_write`; therefore it does
   not validate the three-tool knowledge/memory loop or seekdb. The separate embedded seekdb
   compatibility issue must not be hidden inside this result.

Raw JSON and the machine-readable summary are under
`experiment-results/hipfire-qwen36-agent-20260805/`.

## Scientific-boundary smoke test

A separate routed request exposed only `case_status` and `case_dossier` (681 schema bytes) and
asked the model to call each exactly once, then preserve the fixture/Vina evidence boundaries.
The result is **PARTIAL**, not PASS:

- `case_status` succeeded and reported the run state as `REPORTED`.
- The current host/runtime hardware identity did not match the fixture-bound identity, so the
  gate was `FAILED` and `case_dossier` correctly failed closed instead of returning unbound
  evidence.
- The final answer correctly stated that fixture output is not scientific evidence and that a
  Vina score is not experimental binding free energy.
- The model retried `case_dossier` once after the first failure, despite the explicit
  exactly-once request. The trace therefore contains one successful `case_status` and two failed
  `case_dossier` calls. This should become a runtime policy test: read-only tool failure must not
  be retried automatically unless the user or a declared deterministic retry policy permits it.
- The top-level `tool_calls` value was `1` even though `tool_results` contained three attempts;
  consumers should use the full tool timeline until that summary-field semantics is clarified.

The raw receipt is `case-boundary-routed.json`; its outcome must not be described as a complete
case/dossier workflow validation.

## Post-run runtime correction

The raw run above remains frozen and is not relabelled. After reviewing it, the host runtime was
changed to count every model-emitted invocation per model response instead of deduplicating
`tool_call.id` across the whole session; OpenAI-compatible backends may reuse an ID on a later
turn. An approval pause/resume still counts the pending invocation only once.

ProtBind now also disables identical failed-call re-execution within one Agent session. A repeated
model request remains visible and counted as an attempted call, but its tool result is marked
`automatic_retry_blocked=true` and the handler is not run again. Unit regression tests cover both
reused call IDs and retry blocking. This correction is code-level evidence only until the same
scientific-boundary workload is rerun against the live HipFire backend; it does not retroactively
turn the raw PARTIAL result into PASS.
