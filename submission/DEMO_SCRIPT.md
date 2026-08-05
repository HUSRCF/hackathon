# 3–5 minute demo script

## Recording rules

- Record one continuous Radeon execution path; do not substitute screenshots for execution.
- Keep `rocm-smi`, the local HipFire endpoint, and the ProtBind UI/terminal legible.
- Use public or synthetic demo data only.
- Do not expose `.env`, tokens, private sequences, absolute private-library paths, or unpublished
  compounds.
- State scientific boundaries verbally; never call a score experimental affinity.

Prepare outside the recording if desired:

```bash
submission/demo/run-demo.sh prepare
```

Then start recording and execute the frozen public workflow:

```bash
submission/demo/run-demo.sh run \
  --gpu 1 \
  --model qwen3.6:27b \
  --base-url http://127.0.0.1:11435/v1
```

The command refuses a dirty Git tree, a non-loopback model endpoint, missing Radeon/HIP evidence,
CPU/HIP ranked-ID disagreement, and any case that does not reach the accepted synthetic
`SCREENED` state. The approval response remains interactive and is never piped by the script.

## Shot list (target: 4 minutes 30 seconds)

### 0:00–0:25 — Problem and product

Show the title and architecture diagram.

Narration: “ProtBind is a local-private protein–ligand research Agent. The LLM runs on an AMD
Radeon GPU, while deterministic tools and content-addressed receipts—not generated prose—control
scientific state.”

### 0:25–0:50 — Radeon and local inference proof

Show `rocm-smi`, the selected W7900/gfx1100 device, HipFire health, and `protbind doctor`.

Narration: “The core Agent is local. This host uses Radeon Pro W7900 and ROCm/HIP. Private data is
not sent to a closed online API.”

### 0:50–1:35 — Agent status and ShadowPlan

Open a prepared public demo case. Ask the Agent to inspect status and advance only one safe stage.
Show the fixed tool call, ShadowPlan, approval prompt, and timeline.

Reject the first approval once, then retry and approve, demonstrating that rejection is not
reported to the model as a fabricated scientific failure.

### 1:35–2:20 — TriPharm CPU/HIP parity

Run a prepared small persisted-index query. Show:

- Radeon HIP execution;
- CPU reference execution;
- complete ranked-ID parity;
- kernel versus full-path timing labels.

Narration: “The HIP prefilter is fast and exact against the CPU reference. We do not claim an
end-to-end speedup because the current CPU exact finalizer dominates the full workflow.”

### 2:20–3:10 — Docking and scientific gates

Resume a precomputed public case to show docking artifacts, PoseBusters, RMSD/IFP metadata, and the
evidence grade. Do not wait for a long docking run during the video; execute a bounded stage whose
inputs are already prepared and show the bound source receipt.

Narration: “Vina proposes poses. PoseBusters and reference-aware metrics evaluate them. A docking
score is not experimental affinity.”

### 3:10–3:40 — Optional orthogonal annotation

Show `drutai-status` and one synthetic annotation. Point to strict Snap confinement and the absence
of network interfaces in the receipt.

Narration: “DrutAI is optional and annotation-only. A discordant prediction requests review; it
never deletes a candidate.”

### 3:40–4:10 — Experimental handoff

Preview a synthetic assay table, show that preview performs no write, approve commit, and run one
explicit curve fit. Emphasize immutable raw/canonical artifacts and no arbitrary SQL/update/delete.

### 4:10–4:30 — Close

Show the final evidence-bounded report and artifact citations.

Narration: “ProtBind connects local Radeon inference, controlled scientific tools, and future
wet-lab feedback without allowing the Agent to overstate evidence.”

## Pre-recording checklist

- [x] Freeze a public synthetic demo case and exact commands under `submission/demo/`.
- Warm HipFire once, then restart the screen recording before the measured demo.
- Close unrelated terminals and notifications.
- Verify font size at 1080p.
- Rehearse failure recovery and approval timing.
- Confirm every shown artifact belongs to the public demo workspace.
