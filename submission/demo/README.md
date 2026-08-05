# ProtBind one-command Radeon demo

This demo uses only committed synthetic protocol fixtures. It performs no network fetch, reads no
`.env` file, and makes no biological, binding, affinity, or end-to-end acceleration claim.

## Final recording command

Start HipFire separately on the loopback interface with the recorded local model, then run:

```bash
submission/demo/run-demo.sh all \
  --gpu 1 \
  --model qwen3.6:27b \
  --base-url http://127.0.0.1:11435/v1
```

The script requires a clean Git worktree for a final run. It creates a fresh output directory under
`experiment-results/submission-demo/`, builds the HIP binaries, captures a redacted doctor receipt,
checks the loopback HipFire model, runs CPU and HIP TriPharm paths, requires complete ranked-ID
parity, advances a synthetic case to `SCREENED`, and opens an interactive approval for one Agent
`case_create` call. Inspect the ShadowPlan before typing `y`.

## Rehearsal without HipFire

The deterministic Radeon/scientific portion can be rehearsed while HipFire is offline:

```bash
submission/demo/run-demo.sh all --gpu 1 --skip-agent --allow-dirty
```

Such a rehearsal is explicitly incomplete and must not be used as the final competition video.

## Separate preparation and recording

To keep compilation outside the screen recording:

```bash
submission/demo/run-demo.sh prepare

# Start recording, then:
submission/demo/run-demo.sh run --gpu 1 --model qwen3.6:27b
```

For the final recording, `run` performs preflight and refuses an existing output directory. Use
`all` instead when showing the HIP build is part of the desired recording.

## Acceptance conditions

- An AMD `gfx*` architecture is present and `HSA_OVERRIDE_GFX_VERSION` is absent.
- `hipcc` and the production `tripharm_hip_query` executable are available.
- HipFire is reachable only through an exact loopback `/v1` URL and advertises the requested model.
- The CPU and HIP runs bind the same index and query and produce the same complete ranked-ID hash.
- The HIP receipt is eligible Radeon evidence.
- The synthetic workflow reaches `SCREENED`, records no failure, and exposes both the screening
  artifact and its backend receipt.
- Agent mutation remains interactive; no approval is piped or synthesized by the script.

Any failed condition terminates the script with a non-zero exit status. Existing output directories
are never overwritten.
