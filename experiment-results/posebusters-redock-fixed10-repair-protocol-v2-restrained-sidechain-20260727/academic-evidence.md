# Constrained receptor-repair protocol v2: frozen ten-case pose-recovery pilot

> This is a hash-bound computational evidence packet. It is not evidence of experimental binding affinity or clinical effect.

## Study identity

- Study: `posebusters-fixed10-repair-v2-pilot`
- Scope: `PILOT_METHOD_VERIFICATION`
- Analysis timing: `RETROSPECTIVE_AFTER_OUTCOME`
- Protocol: `sha256:0e478850d0be7dfa0eaced3a0c448bdae7dcb456ae4d3d095522c5b49218395d`
- Evidence packet: `sha256:1bf3022935a8be5447f332ddc3cdf3e0b3ec954c1e01eb4219273e9f1a5c3ff8`

## Candidate estimates

| Endpoint | Successes / total | Estimate | Wilson interval |
|---|---:|---:|---:|
| Workflow completion | 10 / 10 | 1.000 | [0.722, 1.000] |
| Top-1 PB-valid + RMSD ≤ 2 Å | 9 / 10 | 0.900 | [0.596, 0.982] |
| Top-5 oracle PB-valid + RMSD ≤ 2 Å | 9 / 10 | 0.900 | [0.596, 0.982] |

## Claim–evidence matrix

| Claim | Role | Status | Observed | Evidence |
|---|---|---|---:|---|
| Repair protocol v2 reaches at least 80% top-1 PB-valid pose recovery on the fixed ten-case pilot. | primary | `SUPPORTED` | 0.900000 | `sha256:5f524930248aebe717a316a0a44c89be95e805b354ef5b00eaaf7436d3c70b35` |
| ↳ rationale |  |  |  | Observed rate 0.900000 is at least the frozen descriptive threshold 0.800000. |
| Repair protocol v2 completes all ten frozen cases without removing failures from the denominator. | secondary | `SUPPORTED` | 1.000000 | `sha256:5f524930248aebe717a316a0a44c89be95e805b354ef5b00eaaf7436d3c70b35` |
| ↳ rationale |  |  |  | Observed rate 1.000000 equals the frozen threshold 1.000000. |
| Repair protocol v2 is superior to repair protocol v1 for paired top-1 pose recovery. | secondary | `INCONCLUSIVE` | 0.100000 | `sha256:5f524930248aebe717a316a0a44c89be95e805b354ef5b00eaaf7436d3c70b35`, `sha256:076643a00d761dc00afc9a81e64ca69c509eba28743e49c8ad0b786e51459abc` |
| ↳ rationale |  |  |  | The exact paired comparison does not cross the frozen alpha; a raw rate difference alone is not evidence of superiority. |
| Repair protocol v2 reaches at least 80% top-5 oracle pose recovery on the fixed ten-case pilot. | secondary | `SUPPORTED` | 0.900000 | `sha256:5f524930248aebe717a316a0a44c89be95e805b354ef5b00eaaf7436d3c70b35` |
| ↳ rationale |  |  |  | Observed rate 0.900000 is at least the frozen descriptive threshold 0.800000. |

## Paired protocol comparison

- Candidate-only successes: 1
- Baseline-only successes: 0
- Absolute top-1 rate difference: 0.100
- Exact two-sided McNemar p-value: 1.000000

A positive raw difference is descriptive. Superiority is only supported when the frozen paired test crosses alpha in the declared direction.

## Registered but not automatically credited

| Type | ID | Status | Purpose |
|---|---|---|---|
| negative control | `perturbed-pocket-box` | `NOT_RUN` | Translate the known-site docking box away from the crystallographic site to test whether recovery depends on correct pocket localization. |
| negative control | `shuffled-vina-ranking` | `NOT_RUN` | Randomize pose order with a frozen seed to quantify how much top-1 recovery depends on ranking rather than top-5 sampling. |
| negative control | `identity-and-stereochemistry-tamper` | `NOT_RUN` | Verify that atom-identity, bond-order, and stereochemistry changes fail closed before being counted as valid recovery. |
| ablation | `repair-disabled` | `NOT_RUN` | Separate receptor-preparation coverage from docking pose ranking. |
| ablation | `conservative-repair-v1` | `PARTIALLY_OBSERVED_AS_BASELINE` | Compare conservative heavy-atom repair against the constrained new-atom geometry optimization in v2. |
| ablation | `fixed-single-iteration-budget` | `NOT_RUN` | Test whether the adaptive 250/1000/5000 iteration ladder changes receptor-preparation success or only runtime. |
| ablation | `posebusters-validity-removed` | `NOT_RUN_DIAGNOSTIC_ONLY` | Measure how RMSD-only reporting would inflate apparent success; this ablation may not replace the primary endpoint. |

## Scientific boundaries

- The ten cases are a pilot method-verification set and cannot establish broad target, chemotype, or apo-to-holo generalisation.
- Known-site redocking measures pose reproduction when the crystallographic pocket is supplied; it does not measure blind pocket finding.
- The reference pose is validation-only and must remain unavailable to docking and ranking.
- Top-5 is an oracle diagnostic and must not be described as prospective top-1 selection accuracy.
- Vina scores are ranking diagnostics, not experimental binding free energies.
- PoseBusters validity, symmetry-aware RMSD, and interaction fingerprints are structural evidence, not proof of biological binding.
- A retrospective protocol can make its analysis reproducible but cannot retroactively become a prospective preregistration.
- All frozen cases remain in the denominator; preparation or metric failures count as failures rather than exclusions.

## Interpretation

SUPPORTED means the frozen rule was met by these bound artifacts. It does not mean biological truth, experimental binding, or universal generalisation was proven.

Generalisation, affinity, and screening hit-rate claims are all `NOT_EVALUATED` by this packet.
