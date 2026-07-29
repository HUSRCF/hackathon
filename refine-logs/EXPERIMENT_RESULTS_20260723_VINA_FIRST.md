# ProtBind Vina-first implementation results — 2026-07-23

## Outcome

ProtBind now has a schema-2 Vina-first main path in which `SELECTED → DOCKED → VALIDATED`
does not depend on protein–ligand cofolding. OpenFold3 remains an optional candidate-level side
task. A standalone, sealed-reference redocking command was implemented and exercised with real
Meeko, AutoDock Vina, PoseBusters, and sPyRMSD in the AIAA-backed environment.

The completed three-complex pilot recovered 2/3 top-1 poses and 3/3 top-5 poses under the declared
success rule `PoseBusters geometry/chemistry valid AND symmetry-aware RMSD ≤ 2 Å`. This is a
known-site redocking calibration, not blind or prospective docking, and is not a binding/activity
claim.

## Frozen runtime and protocol

- Host: 2× AMD Radeon Pro W7900 (`gfx1100`), each reporting 48,301,604,864 bytes VRAM.
- AIAA/core Python: 3.12.7. The overlay reuses AIAA; no second Torch was installed.
- Scientific tools: AutoDock Vina 1.2.7, Meeko 0.7.1, PoseBusters 0.6.5,
  sPyRMSD 0.9.0, RDKit 2025.9.3, Gemmi 0.7.5, NumPy 2.2.6.
- Vina parameters for all completed cases: seed `20260721`, scoring `vina`, CPU `1`,
  exhaustiveness `8`, requested modes `9`, energy range `3.0`, native heavy-atom bounds plus
  `5 Å` symmetric padding.
- Shared ProtBind source-manifest SHA-256:
  `ed0f8820500ca2dca5fd3c2a0cc616148d5da39197cc48e898667de853559f45`
  over 30 regular `src/protbind_agent/**/*.py` files.
- Shared toolchain SHA-256:
  `545751d2e8a8cc816e65f8828a50acaf32a2207f77a9556c33230ade7d302f86`.
- Vina executable SHA-256:
  `f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644`.
- `HSA_OVERRIDE_GFX_VERSION` was absent. Vina/Meeko/PoseBusters/sPyRMSD ran on CPU; this result is
  not a Radeon docking-speed claim.

Each final `result.json` contains the full source-file manifest, Python executable receipt,
config/toolchain/code hashes, and a composite run identity. Artifact references carry source,
license, producer/version, size, and SHA-256; derived receptor, ligand, and pose artifacts inherit
the input license annotation.

## Real redocking results

| Case | Input source | Top-1 Vina score | Top-1 PB-valid | Top-1 symmetry RMSD | Top-5 best | Scientific status | Vina time |
|---|---|---:|---:|---:|---:|---|---:|
| 1IEP | AutoDock-Vina v1.2.7 official example | -12.440 | yes | 0.827 Å | mode 1, 0.827 Å | `REDOCKING_RECOVERED_TOP1` | 64.559 s |
| 1S19_MC9 | Astex data in Zenodo 8278563 | -12.119 | yes | 0.890 Å | mode 1, 0.890 Å | `REDOCKING_RECOVERED_TOP1` | 31.380 s |
| 1S3V | PoseBusters 0.6.5 package fixture | -9.401 | yes | 6.566 Å | mode 3, 0.371 Å | `REDOCKING_RECOVERED_TOP5_ONLY` | 24.721 s |

Aggregate over these three completed cases: top-1 recovery `2/3`; top-5 oracle recovery `3/3`.
“Top-5” deliberately means reference-aware oracle selection among up to five Vina-ranked modes;
it is not a prospective top-1 result. Vina scores are model tool scores, not experimental binding
free energies and are not compared across scoring functions.

Final receipts:

| Case | Result | Result SHA-256 | Run identity SHA-256 |
|---|---|---|---|
| 1IEP | [result.json](../experiment-results/redock-1iep-20260723-final/result.json) | `9f767a8baa5771bf5eb1a7dce03382661668d87ea81fef71362e4b2c9d32cc67` | `1b91182826b9534482ef03da20528b8e7646d5a8d1e4e0d5d1172b068a48d2ea` |
| 1S19_MC9 | [result.json](../experiment-results/redock-1s19-mc9-20260723-final/result.json) | `af2b5b81e952bd9205760f7f15f9468cb86fb015b4ff101f821736903ec56cae` | `58d7cdc8d5c64a222044901c630b74f51cdef938e042e2cbda8a26992438f3b4` |
| 1S3V | [result.json](../experiment-results/redock-1s3v-20260723-final/result.json) | `0a48d181dff83cfd5b60b013eb13e48314fead6704cef64f6a3ca6f99829acdb` | `de67708271b7bcb7d62417426f1092162f7f045f36a3c6fa00b2d8cbd48f53ce` |

The 1IEP official PDB example contained one SER A438 hydrogen `ATOM` record at the start of the
file. ProtBind reordered intact `ATOM` records by their existing serials before Meeko. Its receipt
proves: 4,412 input/output atom records; zero added or removed; atom-record bytes unchanged;
coordinates unchanged; atom-record multiset SHA-256
`2ed8a4d8e4c7c3727c3e6f6a2d7d934cf4e3656dbbaa636bb25dd25ad470ff8d`.
The normalizer refuses out-of-order inputs containing ANISOU or multi-model sidecars.

## Pre-docking fail-closed evidence

These cases are retained and are not silently removed from the audit trail:

| Case | Stage | Explicit reason | Result |
|---|---|---|---|
| 1UOU | ligand initialization | potentially critical double-bond stereochemistry unspecified (`Bond_Double:11`) | [result.json](../experiment-results/redock-1uou-20260723-final/result.json) |
| 1KE5_LS1 | receptor preparation | Meeko could not match multiple residues with missing side-chain heavy atoms; ProtBind did not use `--allow_bad_res` | [result.json](../experiment-results/redock-1ke5-ls1-20260723-final/result.json) |
| raw Zenodo 1S3V_TQD | receptor validation | retained sulfate (`SO4`); ProtBind did not silently delete it | [result.json](../experiment-results/redock-1s3v-tqd-20260723-final/result.json) |

Their result SHA-256 values are respectively
`01b8f78ba9f157c460b4f361920d1e619a8f6f2e500a1b2ada92f788512464b0`,
`24594c9db67fd5873bb8413826b9534843d03e9acb5490de19c38033c5e912ff`, and
`9e97875a686179bd8c208ec2cf903e9a30c7a2e28ca90df826ed35d78db1a7c1`.

For third-case selection, all 85 Astex entries were screened before Vina. Thirteen passed the v1
chain/residue/metal/heterogen/backbone/stereo/box gates; six of those also passed pinned Meeko
receptor preparation. `1S19_MC9` was the lexicographically first Meeko-eligible case, so it was not
chosen based on docking outcome. The downloaded Zenodo archive was not added to the repository:
size 53,660,397 bytes, SHA-256
`495a8f432ee5612c0dfa3cc582829f112bfca3c29dddc2db2c3a8dc7609e721c`, Zenodo-declared
`CC-BY-4.0`.

## Reference isolation and privacy verification

For all three completed final runs:

- the native coordinate artifact is marked `VALIDATION_ONLY`;
- only coordinate-free ligand identity and a native-derived box receipt reach docking;
- native artifact identity/path is absent from Meeko/Vina argv and the docking-visible case;
- native coordinates are released only after a committed docking pose exists;
- `/home/...` and `/tmp/...` absolute internal paths are absent from `result.json`;
- the final aggregation independently returned `all_native_sealed=true`,
  `all_paths_redacted=true`, and `all_derived_licenses=true`.

This isolation does not turn the experiment into blind docking: the native ligand identity and
native-derived box remain authorized known-site inputs.

## Automated verification

- Ruff: all `src`, `workers`, and `tests` checks pass.
- Compile check: `src`, `workers`, and `tests` pass.
- Pytest: 184 tests pass in the AIAA-backed environment.
- Focused tests cover schema-2 state progression, Vina-without-cofold, minimum-disclosure stage
  envelopes, reference attachment after `DOCKED`, validation-only noninterference, canonical SDF
  round-trip, PoseBusters RMSD separation, PDB record-order integrity, tool/version pins, code/run
  identities, and license lineage.
- Doctor detects two real `gfx1100` devices and no HSA architecture override.

## What this does not yet prove

- This is three-case known-site redocking, not the planned 10-case Astex/PoseBusters regression,
  blind pocket discovery, virtual-screen hit rate, affinity prediction, or biological validation.
- The standalone pilot does not yet report ProLIF IFP recovery, ligand strain decomposition, or
  OpenMM relaxation. Today's AIAA doctor exposes only OpenMM `Reference` and `CPU`; no HIP platform.
- The main workflow still needs automatic execution of all quick-Vina selection requests and a
  real same-complex `both`/`ligand_only`/`pocket_only` end-to-end run through final reports.
- fpocket/P2Rank are absent, so `ligand_only` pocket discovery is not production-ready.
- TriPharm HIP has a real triangle-kernel microbenchmark, but persisted-index top-512 end-to-end
  parity/≥5× is still pending.
- OpenFold3 passed the official ROCm runtime validator but has no real checkpoint inference here;
  ESMFold v1 remains receptor-only. Neither is required for the Vina-first path.
- HipFire/PowerMem integration, local 3Dmol asset, full RAG corpus, and 1/2/4×gfx1100 scheduling
  matrix remain incomplete.
