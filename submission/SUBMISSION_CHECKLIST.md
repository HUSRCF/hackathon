# AMD AI DevMaster submission checklist

Status date: 2026-08-05. Deadline: 2026-08-06 23:59 UTC+8.

## Eligibility and registration — human action required

- [ ] Luma registration approved.
- [ ] Every member registered in the AMD AI Developer Program.
- [ ] Real names and the exact same team name used by all members.
- [ ] Team has at most three members.
- [ ] Final Rules & Conditions and IP response reviewed before submission.

## Track 2 deliverables

- [x] English Project Specification source.
- [x] Project Specification PDF generated and visually inspected.
- [x] Complete source code exists locally.
- [x] English judge quick-start, dependency and reproduction instructions.
- [ ] Confirm the exact registered team name; replace the generic `ProtBind Team` metadata and all
  PR/repository placeholders if it differs.
- [ ] Freeze a clean final Git commit and record its SHA-256 evidence identities.
- [ ] Record and publish a 3–5 minute continuous Radeon demo video.
- [x] Demo script and public-data recording checklist.
- [x] One-command synthetic Radeon demo with fail-closed preflight and CPU/HIP parity receipt.
- [x] Editable PPTX generated and visually inspected.

## PR submission

- [ ] Fork `AMD-DEV-CONTEST/Radeon-hackathon-2026-07`.
- [ ] Add the submission materials in the fork as instructed by the organizer.
- [ ] PR title exactly follows `Track 2, <Team name>, ProtBind`.
- [ ] PR body and every submitted description are in English.
- [ ] Verify every public link in a signed-out browser.
- [ ] Open the PR before the deadline and retain the URL/commit SHA.

## Reproducibility and evidence

- [x] W7900/gfx1100 local Agent receipt exists and is evidence-eligible.
- [x] TriPharm CPU/HIP parity receipts exist.
- [x] Scientific negative results and unsupported claims are explicit.
- [x] Full test suite passes in the AIAA science overlay.
- [ ] Rerun tests, doctor and demo commands on the frozen submission commit.
- [ ] Capture `rocm-smi`, HipFire health and actual end-to-end UI/CLI execution in the video.
- [x] Counterbalanced routed/full Qwen3.6-27B Agent A/B completed on W7900; report exact context
  reduction and keep the observed latency ratio explicitly exploratory.
- [ ] Decide whether R9700/gfx1201 cross-verification can finish without risking the submission.

## Security and licensing

- [x] `.env` is not tracked.
- [x] DrutAI weights are not distributed and require explicit GPL acknowledgement.
- [x] Independently handwritten DrutAI adapter has a file-scoped MIT license.
- [ ] Review every third-party dataset/model/tool license in the final public tree.
- [ ] Run a final secret scan over tracked files and video frames.
- [ ] Confirm that no private sequence, compound, assay value, absolute private path, token or
  unpublished result appears in the repository or video.
