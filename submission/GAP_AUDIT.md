# Submission gap audit — 2026-08-04

The current codebase is technically substantial and scientifically conservative. The remaining
submission risk is concentrated in packaging, human registration, video evidence, and one missing
performance A/B—not in the number of additional scientific features.

## P0: must close before the deadline

1. **Freeze a clean public commit.** The current working tree contains the completed TriPharm,
   MMseqs, DrutAI, assay, documentation, figure, and submission changes but HEAD still points to
   the earlier leakage-audit commit. Performance receipts cannot honestly bind the final code
   until this is resolved.
2. **Record the actual 3–5 minute Radeon demo.** The script exists, but the video itself cannot be
   generated from repository evidence. It must show `rocm-smi`, local HipFire, a real tool call,
   approval, Radeon execution, and a final result.
3. **Complete human eligibility and PR metadata.** Confirm Luma approval, AMD Developer Program
   membership, exact registered team name, contributions, public URLs, and the required PR title.
4. **Resolve the competition IP decision.** Review the organizer's response to the licensing
   question before submitting code or results that should not be covered by the contest grant.
5. **Run the final clean-tree QA and secret/license scan.** Existing tests pass, but the final
   submission commit must be the one tested and recorded.

## P0-high-value scoring experiment

Run the existing hash-bound Agent benchmark twice on the same frozen model, daemon, prompt suite,
workspace and GPU:

- routed minimal ToolSpec pack (`--tool-routing`);
- full ToolSpec pack (`--no-tool-routing`).

The benchmark already records exposed schema bytes, TTFT, wall time, tool success and citations.
This is the fairest available inference-optimization A/B for the Track-2 Radeon/ROCm score. It
cannot run as evidence while the ProtBind source tree is dirty because the benchmark correctly
requires a clean exact revision.

## P1: do only if they cannot destabilize P0

- Run the same exactness smoke on R9700/gfx1201 and report it as cross-architecture verification,
  not as a W7900 speedup.
- Prepare one public case that demonstrates a bounded stage transition through the final report.
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
