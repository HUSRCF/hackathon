# ProtBind scientific-gap results — 2026-07-25

## Outcome

This iteration closes six implementation gaps without promoting protocol checks into biological
claims:

1. A hash-bound redocking regression layer now recomputes top-one and top-five pose recovery,
   PoseBusters diagnostics, and ProLIF interaction fingerprints from source artifacts.
2. A target-specific known-site calibration receipt is now consumed before both manual and
   automatic `both`-mode selection.
3. The current selection-2.5/quick-1.3 AIAA direct adapter contract has been rerun successfully.
4. A result-blind, hash-selected ten-complex PoseBusters holdout has been frozen and every case was
   attempted through a resumable, resource-capped batch runner.
5. ProLIF now uses a receipted 8 Å whole-residue receptor crop around both compared ligands, avoiding
   distant receptor topology artifacts without moving or deleting pocket atoms.
6. Missing standard-residue heavy atoms can be repaired only outside a protected pocket, with an
   explicit receipt and no loop rebuilding, hydrogen addition, or `--allow_bad_res` fallback.

The frozen fixed-ten result is **7/10 at top 1 and 7/10 at top-5 oracle** under the primary gate
`PB-valid AND symmetry-RMSD <=2 Å`. Eight cases reached independent metrics and two failed closed at
Meeko receptor preparation. Consequently `gate_complete=false`; this is an honest known-site
redocking result with a full failure denominator, not a prospective hit-rate, binding, affinity, or
Radeon-performance claim. The older six-case section remains below as historical pilot evidence.

## Result-blind fixed-ten PoseBusters holdout

The source candidate list contains 308 PoseBench/PoseBusters IDs. The declared chemistry and
structure filter produced 298 explicit exclusion records and ten selected cases. Selection was
performed before any docking result was read, using
`sha256(namespace + ':' + complex_id), then complex_id`; no case substitution is allowed.

| Frozen artifact | File SHA-256 | Internal identity |
|---|---|---|
| [holdout](../experiment-results/posebusters-redock-holdout-20260725/holdout.json) | `01d9fd57f31ef006601b6a1e982d2cf020d50761bc9c8f5bfe61497ccc064ca3` | selection `242f43398baa76fee2b6b7ab0e53546cd066597cb3f638f81be4378dc8146a89` |
| [run plan](../experiment-results/posebusters-redock-fixed10-formal-20260725/run-plan.json) | `a3eb1736d63fb76086c111fddb56fff75793d307c76cc4f409a2e7fe266ff99c` | plan `fb41f3fd03815f0bd2bf0d2fa1e61f23db4c39797af2da0a34bd6460760fd6f2` |
| [batch result](../experiment-results/posebusters-redock-fixed10-formal-20260725/batch-result.json) | `89deee7806792f120ef23b6cf72aab8bd910c54ac6b929fbfa5b2dbd1a36ad06` | result `0d02ec33db434d302ba72e735338fa8ec0299819e8f2e274f677332332a562c4` |
| [regression manifest](../experiment-results/posebusters-redock-fixed10-formal-20260725/regression-manifest.json) | `34d72fa21156ade868486dda6df791cc4bc2ccc6a44351417882039bb6af8b56` | manifest `86f213ae8d6f48721622680fb616ed3387f1483e35e9d19a1eba4b7172633274` |
| [independent regression v2](../experiment-results/posebusters-redock-fixed10-formal-20260725/regression-v2.json) | `cab37219c7918a852a35b0296199da8cc60e5bdb52c584f72a8d517e232a85a8` | regression `084c8a2b8476a0555a9de454204139bbc1cee06f9feb082ada4e8cd756695e6d` |

The run plan freezes the holdout/input hashes, code and exact tool identities, seed, Vina settings,
and per-case sources before the first case is launched. Resume accepts an existing terminal case
only after all bindings revalidate. At most two single-CPU Vina cases run concurrently, preserving
host and Radeon headroom; this benchmark used no GPU and is not Radeon performance evidence.

| Denominator field | Count |
|---|---:|
| frozen / attempted / terminal | 10 / 10 / 10 |
| redock completed | 8 |
| redock failed closed | 2 |
| metric failed | 0 |
| independent metrics completed | 8 |

| Metric | Numerator / frozen denominator | Rate |
|---|---:|---:|
| PB-valid and symmetry-RMSD <=2 Å, top 1 | 7/10 | 0.7000 |
| PB-valid and symmetry-RMSD <=2 Å, top-5 oracle | 7/10 | 0.7000 |

| Case | Formal status | Top-1 RMSD | Best top-5 RMSD | IFP Jaccard | Interpretation |
|---|---|---:|---:|---:|---|
| 7XFA_D9J | metrics completed | 9.2686 Å | 6.7836 Å | 0.0909 | PB-valid poses, not recovered |
| 7AN5_RDH | metrics completed | 0.5345 Å | 0.5345 Å | 0.6000 | top-1 recovered |
| 7WQQ_5Z6 | metrics completed | 0.4618 Å | 0.4039 Å | 0.8333 | top-1 recovered |
| 7BTT_F8R | redock failed | — | — | — | Meeko rejected missing side chains |
| 7YZU_DO7 | redock failed | — | — | — | Meeko rejected missing side chains |
| 8BOM_QU6 | metrics completed | 1.2914 Å | 1.2914 Å | 0.5714 | top-1 recovered |
| 6YQW_82I | metrics completed | 0.6001 Å | 0.6001 Å | 1.0000 | top-1 recovered |
| 7THI_PGA | metrics completed | 0.7223 Å | 0.7138 Å | 0.7778 | top-1 recovered |
| 7DUA_HJ0 | metrics completed | 1.1377 Å | 1.1377 Å | 0.6250 | top-1 recovered |
| 7ELT_TYM | metrics completed | 0.3960 Å | 0.3960 Å | 0.5000 | top-1 recovered |

The independent evaluator reopened each hash-bound multi-pose SDF and recomputed every top-one and
top-five sPyRMSD/PoseBusters result. Across the eight completed cases, mean/median IFP Jaccard are
0.6248/0.6125, and no top-one pose has a PoseBusters protein-ligand clash. IFP preparation mode is
`RECEIPTED_LIGAND_ADDHS_AND_RECEPTOR_POCKET_CROP`: ligands receive receipted RDKit hydrogen handling,
while the receptor is cropped as the union of complete residues within 8 Å of either docked or
native ligand heavy atoms. The eight crops contain 32–72 residues and 491–1123 atoms; all retained
atom identities match and maximum coordinate delta is 0.0 Å. Cropping is validation-only and never
changes docking input or pose recovery.

The earlier whole-receptor ProLIF pass produced the same PB/RMSD decisions, but distant accidental
proximity bonds affected three IFP values. `regression-v2.json` is therefore authoritative for IFP:
7WQQ changed from 0.7500 to 0.8333, 8BOM from 0.5000 to 0.5714, and 7ELT from 0.5385 to 0.5000.

### Conservative repair remediation is not the formal result

The original fixed-ten artifacts above are immutable. A separate retrospective two-case ablation
tested the newly implemented conservative repair. It may add standard-residue heavy atoms only
when every missing atom lies outside the 6 Å native-ligand protection radius; it never rebuilds a
missing residue/loop, moves an original heavy atom beyond 0.002 Å, adds hydrogens, or invokes
Meeko `--allow_bad_res`.

| Case | Remediation result | Repair receipt summary |
|---|---|---|
| 7BTT_F8R | completed; PB-valid top-1 recovered at 0.9270 Å; IFP Jaccard 0.8000 | 3 outside-pocket residues, 13 heavy atoms added, 0 original-heavy-atom displacement |
| 7YZU_DO7 | still failed closed at receptor preparation | 14 outside-pocket residues, 63 heavy atoms added, 2877 pre-existing H removed so Meeko remains protonation authority, 0 original-heavy-atom displacement; rebuilt geometry still rejected |

The remediation result files are
[7BTT](../experiment-results/posebusters-redock-repair-remediation-20260725/7btt_f8r/result.json)
(`f1efbe057f47a40075d9b18883815efaa73568df217fdf4cda2b51948815bce7`) and
[7YZU](../experiment-results/posebusters-redock-repair-remediation-20260725/7yzu_do7/result.json)
(`f84b19fa767e075594b26652fff445e23df8604633c2553f70e149ae129660b0`). The independent
[remediation regression v2](../experiment-results/posebusters-redock-repair-remediation-20260725/regression-v2.json)
has file SHA-256 `5fba720c25196b3584f4b03c32622f9beb11bcc30315ce7752e9d66f3c5de7aa` and
internal SHA-256 `d258d5d3c55829d48504187940aae9862dcbe5ad2880b79bea6a50d7a067b1b6`.
Its 1/2 rate is an ablation denominator, not an updated 8/10 formal score.

## Six-case public redocking pilot

Frozen manifest:
[`configs/redock-regression-pilot-20260723.json`](../configs/redock-regression-pilot-20260723.json),
file SHA-256 `9bcaa75605ff2687ce332c6206b449093283ad939f09021dd33ce3d10c0d5b8e`,
internal manifest SHA-256
`2f1f55a65ebf8d7717e08f84d46b4575d61335a5aac2ce1799af627608437a13`.

Result:
[`result.json`](../experiment-results/redock-regression-pilot-20260725/result.json), file SHA-256
`241c1dc039802b1445d595d00fa771269dff70602c61a17f70238b7552e0f27b`, internal regression
SHA-256 `ee6784f5288c582d0557687f41064221cfc6f7a056485a0ca0642a75f0fccfab`.
The internal hash was independently recomputed and matched.

| Denominator field | Count |
|---|---:|
| frozen | 6 |
| attempted | 6 |
| redock completed | 3 |
| redock failed closed | 3 |
| metric failed | 0 |
| metrics completed | 3 |

All frozen cases remain in the recovery denominator:

| Metric | Numerator / denominator | Rate |
|---|---:|---:|
| PB-valid and symmetry-RMSD ≤2 Å, top 1 | 2/6 | 0.3333 |
| PB-valid and symmetry-RMSD ≤2 Å, top-5 oracle | 3/6 | 0.5000 |

Case outcomes:

| Case | Status | Top-1 RMSD / result | Top-5 | Explicit failure |
|---|---|---:|---|---|
| 1IEP | metrics completed | 0.8273 Å, pass | pass | — |
| 1KE5-LS1 | redock failed | — | — | Meeko receptor preparation nonzero exit; missing-side-chain input remains fail closed |
| 1S19-MC9 | metrics completed | 0.8898 Å, pass | pass | — |
| 1S3V | metrics completed | 6.5655 Å, fail | pass at mode 3; 0.3705 Å | — |
| 1S3V-TQD | redock failed | — | — | retained sulfate made receptor non-protein-only |
| 1UOU | redock failed | — | — | unspecified potentially critical double-bond stereochemistry |

Top-five is no longer trusted from a historical boolean. For each completed case, the evaluator
opens the hash-bound multi-record Vina SDF, derives modes 1–5, records each derived pose SHA-256,
and independently recomputes sPyRMSD and 27 PoseBusters gate checks. Historical top-one/top-five
fields are used only as consistency checks. Top-five remains an oracle pose-recovery metric, not a
prospective top-one result.

### IFP and conformational diagnostics

| Case | IFP Jaccard | Reference recovery | Predicted precision | Counts docked/ref/intersection/union | PB energy ratio | Protein clashes |
|---|---:|---:|---:|---|---:|---:|
| 1IEP | 0.7692 | 0.7692 | 1.0000 | 10/13/10/13 | 1.7159 | 0 |
| 1S19-MC9 | 0.4167 | 0.5556 | 0.6250 | 8/9/5/12 | 1.4452 | 0 |
| 1S3V | 0.6000 | 0.7500 | 0.7500 | 4/4/3/5 | 2.8234 | 0 |

The median IFP Jaccard is 0.6000 and median reference-interaction recovery is 0.7500, over only the
three completed cases. Missing ligand hydrogens were handled only through the explicitly enabled,
independent ArtifactStore path. RDKit `AddHs(addCoords=True)` added 40 and 29 hydrogens to the
1S19 and 1S3V native references, respectively, while preserving heavy-atom identity and coordinates;
1IEP required no added hydrogens. The preparation artifacts and receipts are referenced by the
regression result.

PoseBusters energy ratio and internal ligand energies are conformational/strain diagnostics. They
are not binding energies or binding free energies. ProLIF agreement is structural interaction
evidence, not affinity or activity evidence.

Runtime identity SHA-256 is
`be0c3b54681b71dd4fb435feb8414c99be55fa9dfda388ef8db154d4c11ba5d4`, binding CPython 3.12.7,
RDKit 2025.9.3, sPyRMSD 0.9.0, PoseBusters 0.6.5, ProLIF 2.2.0, NumPy 2.2.6, pandas 2.3.2,
distribution RECORD hashes, evaluator implementations, and evaluator source hashes.

## Known-site calibration is now a consumed gate

The previous calibration receipt existed but was not causally required by selection. Selection 2.5
now treats calibration as an explicit all-or-none `both`-mode claim. Before manual or automatic
selection it verifies:

- exact target ID, Meeko prepared-receptor SHA-256, and requested box geometry;
- the canonical source redock artifact and its completed status;
- exact source redock receptor-preparation and native-box artifact pointers;
- preparation receipt bindings and fail-closed protein/cofactor policy;
- recomputed source top-one/top-five PB-valid/symmetry-RMSD metrics;
- an independent coordinate-free site-derivation evidence artifact.

A self-rehashed forged receipt cannot replace the source receptor, box, or metrics. User-center and
user-residue hypotheses cannot be promoted by calibration.

Real 1IEP consumer evidence:

| Artifact | SHA-256 |
|---|---|
| [calibration receipt](../experiment-results/known-site-calibration-1iep-20260725/objects/65/0aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82) | `650aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82` |
| [selection preparation 2.5](../experiment-results/known-site-calibration-1iep-20260725/objects/48/9c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17) | `489c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17` |
| [minimal quick input 1.2](../experiment-results/known-site-calibration-1iep-20260725/objects/26/836f7ca9832907924469d0fdc66d6edc753f2d55b1c31a0bd86c2fe15c1d1b) | `26836f7ca9832907924469d0fdc66d6edc753f2d55b1c31a0bd86c2fe15c1d1b` |

The calibration passed top-one at 0.8273 Å and authorized only its known-site box with the exact
prepared receptor. The quick input commits the selection preparation SHA and contains no
`native_reference`. This is target-specific known-site pose-recovery calibration, not evidence that
another ligand binds.

## Current AIAA quick-Vina contract smoke

Current v4 evidence is under
[`experiment-results/aiaa-selection-quick-vina-20260725-v4`](../experiment-results/aiaa-selection-quick-vina-20260725-v4):

| Artifact | SHA-256 |
|---|---|
| `smoke-result.json` | `8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e` |
| `vina-provenance.json` | `b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f` |
| quick worker code | `ed48d5b6884bd4a572cd7315d046b557b2bfd0ea3d1953dd72ee5acfa7458d14` |
| full Vina worker code | `a181906553cc0960a9e51f02856df3b2bc527fd4a0aad24321d61bd92760f56c` |

Profile 1.3 / selection 2.5 / quick input 1.2 / box receipt 2.0 completed 3/3 requests
with the 36-entry output closure and the expected pruning-only scores
`-1.942/-1.896/-1.753`. The exact 1CRN box contains 269 protein heavy atoms, so coordinate-frame
overlap passes. It remains a user-center hypothesis with no biological-site derivation, the demo
index has `chemistry_verified=false`, and the run used the direct application-offline adapter rather
than production bubblewrap isolation. It is protocol/environment evidence only.

Doctor now recognizes the attested bundled Vina executable. The same doctor run reports two
gfx1100/W7900 devices, reserves GPU 1 for other tools while allowing at most one single-GPU
OpenFold job on GPU 0, and confirms VRAM is not pooled. It also reports:

- fpocket unavailable and P2Rank unavailable;
- OpenMM platforms `Reference` and `CPU` only; no HIP platform;
- bubblewrap present but unusable because loopback setup returns `RTM_NEWADDR`;
- production worker isolation remains fail closed.

## Automated verification

- Full pytest suite: 267 tests passed.
- Ruff over the repository: passed.
- `compileall` over `src` and `workers`: passed.
- Regression result internal SHA-256: independently recomputed and matched.

## Remaining scientific gates

1. Preserve the original fixed-ten as the primary result. If a repaired protocol is evaluated,
   preregister it as a new revision and rerun all ten; never splice the repaired 7BTT result into the
   frozen batch. Resolve 7YZU with a separately validated restrained side-chain geometry protocol or
   keep it unsupported. The current formal gate remains incomplete because two runs failed.
2. Install and integrate fpocket/P2Rank consensus before claiming a real `ligand_only` path; do not
   infer a whole-protein box.
3. Run the same public complex through `both`, `ligand_only`, and `pocket_only` to final reports.
4. Add an explicit cofactor-retention/parameterization policy. Conservative outside-pocket heavy-atom
   repair now exists, but metals, covalent systems, missing loops, and rejected repair geometry must
   continue to fail closed.
5. Build and verify OpenMM HIP and report CPU/HIP parity; current AIAA exposes CPU/Reference only.
6. Rerun production quick selection on a host that permits the bubblewrap namespace. Do not disable
   isolation to obtain a pass.
7. Complete persisted-index TriPharm top-512 parity/end-to-end speed and optional OpenFold3 real
   checkpoint evaluation under the existing one-GPU-at-a-time resource policy.
