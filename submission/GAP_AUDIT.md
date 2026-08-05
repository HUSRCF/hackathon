# Submission gap audit — 2026-08-04

The current codebase is technically substantial and scientifically conservative. The remaining
submission risk is concentrated in packaging, human registration, video evidence, and one missing
performance A/B—not in the number of additional scientific features.

## P0: must close before the deadline

1. **Freeze the final public commit.** The main submission package and Agent retry-audit correction
   are committed. The new one-command demo must now be committed, pushed to the public submission
   branch, and recorded as the final source identity. Existing performance receipts remain bound
   to their explicitly named earlier clean revisions; they must not be silently relabelled.
2. **Record the actual 3–5 minute Radeon demo.** A frozen one-command synthetic workflow now
   builds the HIP path, checks loopback HipFire, requires CPU/HIP ranked-ID parity, reaches an
   accepted `SCREENED` state, and opens a real interactive approval. The video itself still must
   be recorded and published; it must show `rocm-smi`, local HipFire, the tool call, approval,
   Radeon execution, and the final receipts.
3. **Complete human eligibility and PR metadata.** Confirm Luma approval, AMD Developer Program
   membership, exact registered team name, contributions, public URLs, and the required PR title.
4. **Resolve the competition IP decision.** Review the organizer's response to the licensing
   question before submitting code or results that should not be covered by the contest grant.
5. **Run the final clean-tree QA and secret/license scan.** Existing tests pass, but the final
   submission commit must be the one tested and recorded.

## P0-high-value scoring experiment — completed

The routed/full Agent comparison was completed on the same Qwen3.6-27B HipFire daemon and W7900,
using one warm-up per condition followed by an interleaved ABBA/BA measured order:

- routed minimal ToolSpec pack (`--tool-routing`);
- full ToolSpec pack (`--no-tool-routing`).

All six measured `doctor` calls succeeded. Routing reduced exposed tools from 25 to 2, schema bytes
from 9,794 to 454, and mean prompt tokens from 10,406.7 to 4,694.3. Same-server prefix cache,
thermal state, and DFlash acceptance still confound the observed latency ratio, so only the exact
context-reduction claim is submission-safe. See
`refine-logs/EXPERIMENT_RESULTS_20260805_HIPFIRE_ROUTING_AB.md`.

## P1: do only if they cannot destabilize P0

- Run the same exactness smoke on R9700/gfx1201 and report it as cross-architecture verification,
  not as a W7900 speedup.
- The public synthetic `SCREENED` case is complete. A separate precomputed public final-report
  view remains optional and must not replace the live Radeon/Agent path in the video.
- Replace generic `ProtBind Team` metadata with the exact registered team name if different.
- Verify every link from a signed-out browser after the PR is open.

## Do not rush into the submission branch

- resident-GPU exact finalization;
- fpocket/P2Rank automatic consensus;
- a new prospective redocking set;
- Co-IP image segmentation, CETSA thermal fitting, SPR kinetics or active learning;
- OpenFold3 checkpoint integration or OpenMM HIP build changes.

These remain valuable product work, but introducing them immediately before the deadline is more
likely to weaken reproducibility than improve the submitted Agent demo. The current materials
already state these boundaries explicitly.
