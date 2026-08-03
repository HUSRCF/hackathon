# DrutAI Snap runtime smoke — 2026-08-03

## Scope

This is a one-record execution and control-path smoke test using a synthetic protein sequence and
ethanol (`CCO`). It is not a biological benchmark, binding result, calibration result, or evidence
that DrutAI generalizes to a ProtBind target.

## Runtime

- Snap: `drutai` 1.0.5rc3, revision 13, strict confinement.
- Connected interfaces observed before execution: `home`, `removable-media` unconnected; no
  connected `network*` interface.
- Model: `convmixer64` from pinned source commit
  `5ee6ba7037466609edc06329782dee9298f20f2b`.
- Model SHA-256: `cb373de8c1d79177d06b1a763a749ba2aa0588bd95eb0a5563ce65a786e5cb9a`.
- Worker cache: disabled.
- Threads / batch size: 1 / 1.

## Result

- Status: `COMPLETED`.
- Records: 1.
- Annotation counts: 0 `SUPPORTIVE`, 1 `DISCORDANT`, 0 `ABSTAIN`.
- Bundle artifact SHA-256:
  `25b42cf34bc226b79fd72c6a8abbcd46666320e947920db5fe47a9d5303b3fe8`.
- Raw-output artifact SHA-256:
  `5918b1609d7bf11b9a9c92110e45da5c72871dd80c83c0e1195cc3827bd62bd4`.
- Snap info audit SHA-256:
  `3ff6cbd90036bbd8d7158172b5488a728c492e36e4812245c5d59633767056ec`.
- Snap connections audit SHA-256:
  `cd0b848a9222334c1613b8052e46c40083bf07da9cc31d619a0044ae0a50f60b`.

The synthetic `DISCORDANT` annotation has no biological interpretation. All DrutAI outputs remain
annotation-only with `hard_filter_allowed=false`; the adapter does not convert them into binding,
affinity, activity, pose, or experimental evidence.

## Compatibility finding

Wrapping a strict Snap alias inside the existing bubblewrap profile failed because nested
bubblewrap removed a capability required by `snap-confine`. The adapter now branches by executable
type: ordinary executables retain bubblewrap network namespaces, while canonical `/snap/bin/*`
aliases must pass a fresh strict-confinement and interface audit before direct execution. A
connected network interface, devmode/trymode, broken/disabled package, or failed metadata command
causes a fail-closed control error.
