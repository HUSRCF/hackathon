# Isolated scientific workers

ProtBind does not merge the implemented OpenFold3, legacy ESMFold, or core RDKit/Vina stacks into
one Python environment; any future ESMFold2 worker would also require its own isolated environment.
Each configured process receives exactly one JSON request on stdin and must emit exactly one JSON
response line on stdout. Human diagnostics belong on stderr.

The shared contract is implemented in `protbind_agent.worker_protocol`; model wrappers may use
`protbind_agent.worker_sdk.serve_worker`. The host passes `PROTBIND_ARTIFACT_ROOT` and artifact
references, but does not pass API keys or its full environment.

Pipeline workers receive a content-addressed `protbind.stage-input` envelope rather than only the
previous stage's first file. Its path-free JSON shape is:

```json
{
  "schema_version": "2.0",
  "kind": "protbind.stage-input",
  "stage": "DOCKED",
  "case": {"sha256": "...", "media_type": "application/json"},
  "input_artifacts": {"library_index": {"sha256": "..."}},
  "supporting_artifacts": {
    "support_selection_batch": {"sha256": "..."},
    "support_vina_environment_lock": {"sha256": "..."}
  },
  "previous": {
    "stage": "SELECTED",
    "scientific_outputs": [{"sha256": "..."}],
    "receipt": null
  }
}
```

The abbreviated `ArtifactRef` objects above contain all normal reference fields in a real request.
The host hashes and records this envelope as the stage input, separates an upstream worker receipt
from scientific outputs, and verifies the full dependency graph before resume. A worker resolves
the case and every required input through `ArtifactStore`; it must not infer an internal path.

Required response evidence:

- output `ArtifactRef` values written through the shared `ArtifactStore`;
- model revision, weight SHA-256, and worker-code SHA-256 copied from the verified request;
- wall/kernel timings and peak VRAM where the runtime exposes them;
- structured warnings and an explicit error code for OOM, unsupported chemistry, parse errors, or
  model/runtime failure.

Worker configuration rejects `HSA_OVERRIDE_GFX_VERSION` and environment variable names that look
like credentials. Non-secret environment values are bound into the configuration identity by a
canonical hash and are never copied into the manifest or receipt.

Production pipeline workers default to `isolate_network=true`. The host recursively copies only
the stage envelope's declared artifact graph into a temporary exchange store, sets offline runtime
variables, runs the process under `bubblewrap --unshare-net`, hash-verifies its outputs, and then
imports those outputs into the main store. The current bubblewrap command binds `/` read-only and
binds only the exchange store read-write. It therefore provides OS network isolation and write
confinement, but it is **not yet a strict read-confinement boundary** for other host files. Only
hash-pinned, reviewed worker code should be configured. Disabling OS isolation is rejected by the
pipeline except for explicitly marked `fixture-only` tests.

Run `PYTHONPATH=src python -m protbind_agent doctor` before production. Its
`runtime_details.worker_network_isolation.status` is exactly `missing`, `present_but_unusable`, or
`usable`; only `usable` passes the OS-isolation preflight. This host currently reports
`present_but_unusable` with bubblewrap probe return code 1. Offline environment variables are an
application policy and `application_offline_env_is_os_isolation=false`; they are not an OS network
boundary.

Environment boundaries:

- `openfold3`: official 0.4.3 source in a dedicated AIAA-backed overlay, reusing AIAA's validated
  ROCm Torch/Triton, MSA-free/direct-CIF; the upstream `openfold3-rocm7` lock remains a reference
  implementation, and the adapter exposes `low_mem` but production requires it to be `true`;
- `esmfold_v1`: separate AIAA-backed overlay reusing the base fair-esm and ROCm Torch runtime,
  adding only a pinned legacy OpenFold build and small non-Torch dependencies, with complete
  environment/code/weight hashes and chunk retry 128 → 64 → 32;
- `esmfold2`: future-only; no runnable worker is integrated. Any future isolated environment and
  adapter cannot enter the workflow until the documented 3-complex offline `gfx1100` gate passes;
- core docking/validation target: RDKit, Gemmi, PDBFixer, Meeko/Vina, PoseBusters, ProLIF,
  sPyRMSD and OpenMM HIP; concrete Vina and validation adapters exist. Six public inputs were
  attempted: three completed known-site redocking and three failed closed at declared gates. The
  fixed ten-case regression,
  IFP/strain suite, blind site discovery, and OpenMM HIP platform remain outstanding.

No placeholder worker is supplied that emits fake coordinates. Until a **main-path** environment
or adapter is configured, the run stops as `DEGRADED/CAPABILITY_UNAVAILABLE` (or the more specific
structured worker error). OpenFold3 is a schema-2 optional side task: when it is not requested,
unavailable, or fails recoverably, Vina docking and validation continue.

## OpenFold3 adapter

`openfold3_worker.py` is a concrete offline CLI adapter pinned to OpenFold3 0.4.3 commit
`0bb17be5199846e806b6347b6e17c6249c88ff1b`. Its request provenance must use the exact model
revision:

```text
openfold3-0.4.3@0bb17be5199846e806b6347b6e17c6249c88ff1b
```

Only when optional cofold evidence is requested, first stop the automatic main path after
`SELECTED`, construct a batch bound to that frozen selection, and then attach these three immutable
local support artifacts before resuming `DOCKED`. Passing `--name X` stores the key as
`support_X`:

```bash
protbind case run ... --stop-after selected
protbind case attach RUN_ID --name openfold_batch \
  --file openfold-batch.json --media-type application/json
protbind case attach RUN_ID --name openfold_checkpoint \
  --file openfold3.ckpt --media-type application/octet-stream
protbind case attach RUN_ID --name openfold_environment_lock \
  --file experiment-results/aiaa-openfold3-environment.json \
  --media-type application/json
protbind case resume RUN_ID --worker-config configs/protbind-workers.toml
```

The repository currently exposes a strict `protbind.cofold-input-batch` validator, not a public
builder or CLI that derives this JSON from the automatic selection bundle. The attach sequence is
therefore an expert import path for a batch produced by reviewed external tooling; it is not a
turnkey automatic OpenFold path. The host accepts these supports after `SELECTED`, but freezes them
as soon as cofold starts or `DOCKED` completes.

On that resume, cofold runs at the beginning of `DOCKED`. Its schema-2 stage envelope has
`stage=COFOLDED`, `previous.stage=SELECTED`, the frozen selection as the previous scientific output,
and all three OpenFold supports. The subsequent Vina envelope still has `previous.stage=SELECTED`;
when cofold completed, it additionally carries `cofold_evidence_bundle`. Thus the optional task may
move `RUNNING → COMPLETED`, or record `UNAVAILABLE`/`FAILED_RECOVERABLE`, without becoming the
required upstream state for docking.

Launch the adapter from the explicit AIAA-backed OpenFold wrapper. Attach the generated
`aiaa-openfold3-environment.json` (which includes the official source allowlist and ROCm runtime
attestation) rather than trusting an ambient Python environment:

```toml
argv = ["/absolute/repo/scripts/aiaa-openfold3.sh", "python", "/absolute/repo/workers/openfold3_worker.py"]

[workers.cofold.environment]
HIP_VISIBLE_DEVICES = "0"
```

`support_openfold_batch` must be a `protbind.cofold-input-batch` v1.0 receipt bound to this run's
screening artifact, exact library-index artifact, and an allowed receptor artifact. The host checks
that the protein-chain sequences match the case and receptor, the complete screened ranking is
preserved, Bemis–Murcko scaffolds are unique, and every retained molecule has one to four
parent-traceable microstates. Quick Vina must evaluate one or two microstates for every retained
molecule. Each entry binds its finite score, exact receptor, pose artifact, box center, and box size
to typed Vina evidence. Molecules are ranked deterministically by ascending Vina score, then
`molecule_id`, then `microstate_id`; top-16 and the best-microstate top-8 must match that ordering
exactly. The same batch supplies one or two protein chains, canonical isomeric SMILES, and optional
direct-CIF template references. The adapter does not perform those upstream selection or
quick-docking steps itself.

`weight_sha256` must equal the checkpoint artifact's SHA-256. `code_sha256` is deliberately a
composite identity, not merely the adapter file hash:

```text
openfold_package_source_sha256 = SHA256(canonical_json(sorted([
  [allowlisted_installed_relative_path, SHA256(installed_file)], ...
])))

protbind_runtime_sha256 = SHA256(canonical_json([
  ["src/protbind_agent/<relative>.py", SHA256(installed_file)], ...
]))

code_sha256 = SHA256(canonical_json({
  "schema_version": "1.0",
  "adapter_sha256": SHA256(workers/openfold3_worker.py),
  "protbind_runtime_sha256": protbind_runtime_sha256,
  "environment_lock_sha256": SHA256(aiaa-openfold3-environment.json),
  "openfold_package_source_sha256": openfold_package_source_sha256,
  "openfold_revision":
    "openfold3-0.4.3@0bb17be5199846e806b6347b6e17c6249c88ff1b"
}))
```

The OpenFold allowlist walks the actual package root returned by Python for the imported
`openfold3` module; it does not trust the distribution `RECORD`/file list. This supports the
official editable overlay installation while ensuring `openfold3.run_openfold` resolves inside that
same attested root. The predicate includes every file below that root that either ends in `.py`,
occurs below `core/data/resources/`, or ends in `model_setting_presets.yml`. Thus model resources
and the preset are hashed in addition to Python source. The measured official 0.4.3 runtime
contains exactly 317 allowlisted files and has manifest SHA-256
`742e9bf654b13f67783d095a2327af3ed31163580eaa7b4c548e8a8eb2e68010`; both count and hash
must match. Before inference the adapter also requires distribution version `0.4.3`, console entry
point `run_openfold = openfold3.run_openfold:cli`. When `scm_version.json` exists its tag, distance,
node and dirty fields must match exactly. Upstream editable installs do not currently emit that
custom file, so its absence is accepted only when the exact 317-file official allowlist hash
matches; synthetic or modified installs still fail closed. The resulting runtime
attestation and effective OpenFold config artifacts are retained in the query/run metadata.

Because the adapter imports ProtBind code, its composite identity also hashes the canonical
path/SHA-256 manifest of every `src/protbind_agent/**/*.py` file. Editing any imported host runtime
module therefore changes `code_sha256`; hashing only the adapter and OpenFold package is
insufficient.

`PROTBIND_TEST_RUNTIME=1` can relax only the measured release/ROCm checks for deterministic direct
protocol tests. It is a reserved test fixture switch: `WorkerConfig` rejects it, so it cannot be
placed in a pipeline TOML/environment. Test-runtime output is labeled as fixture output, never as
the official engine.

The adapter generates official query JSON with top-level `seeds: [request.seed]` and, on every
query, `use_msas=false`, `use_paired_msas=false`, and `use_main_msas=false`. The generated runner
YAML repeats the seed and disables the MSA server. It invokes the same environment's interpreter as
`python -m openfold3.run_openfold predict` with `--use-msa-server=False`, passing the local
checkpoint, output directory, sample count, and runner YAML as explicit argv. There is no
executable-path override. Production fixes `num_diffusion_samples=1`; protocol fixtures may exercise
multiple samples. Production also fixes `low_mem=true`, ROCm Triton=true, and MSA server=false;
other supported parameters are `command_timeout_seconds` and
`checkpoint_name`, plus the free-VRAM admission threshold `minimum_free_vram_gib`. OpenFold3 0.4.3
production accepts only its documented default `openfold3-p2-155k` checkpoint.
`weight_sha256`, rather than a filename, remains the checkpoint identity.
A successful response begins with a `protbind.cofold-bundle`, followed by imported CIF/confidence,
timing/effective-config JSON, and generated query/runner metadata. Confidence is labeled as model
confidence, never binding affinity. Peak VRAM is returned as unavailable rather than fabricated
because the child CLI does not expose it through this adapter.
For production, the host independently requires the exact `openfold3` worker engine,
`official-openfold3` candidate engine, pinned bundle/metadata producers, checkpoint/lock/envelope
references, official runtime attestation, and an exact raw-output inventory. Missing or extra
returned artifacts and engine aliases fail closed before the optional cofold record is committed;
they do not turn cofold into a DOCKED dependency.

Every predicted model must be a parseable mmCIF with finite coordinates, exact protein chains
`A` (and `B` for a two-chain request) and sequences, one non-empty ligand residue on chain `Z`, no
unexpected molecular chain, and ligand heavy-element counts identical to the requested microstate.
This gate does **not** yet prove per-atom identity, bond connectivity/order, atom mapping, or
stereochemistry preservation. Those remain downstream chemistry/PoseBusters mapping work and are
not inferred from matching element counts.

Resource policy is intentionally conservative: `HIP_VISIBLE_DEVICES` must name exactly one device
with a canonical decimal index, conflicting ROCr/CUDA mask aliases are rejected, and the runner
forces one trainer device and FP32 (`precision: 32-true`). Before launching, the host takes a
same-user host-global OpenFold lease followed by a device lease keyed by `HIP_VISIBLE_DEVICES`. A second
ProtBind worker on the same device returns recoverable `GPU_BUSY`; a second OpenFold job on another
device in another ProtBind workspace also returns recoverable `OPENFOLD_BUSY`. OpenFold `devices: 2`/`4`
is an upstream distribution mode and does not pool VRAM to make one query fit; the current ProtBind
adapter does not expose that multi-device mode. On a host with two `gfx1100` devices, use GPU0 as the scientific
lane (OpenMM only after OpenFold exits) and reserve GPU1 for HipFire. A one-GPU host must pause GPU
LLM/OpenMM during OpenFold; a four-GPU host still defaults to one OpenFold job and reserves GPU1–3
for other tools. These leases coordinate same-user ProtBind workers across host workspaces; HipFire does
not participate, so it must still be explicitly assigned GPU1 while OpenFold uses GPU0. The
required official `p2-155k` byte size is 2,287,928,196 bytes (approximately 2.13 GiB). There is no
documented small/medium/large checkpoint family for OpenFold3 0.4.3. For an official runtime, the
declared `checkpoint_name`, artifact byte size,
and `weight_sha256` must all agree. No official small model is currently established; use an
attested checkpoint plus `low_mem`, and do not advertise a smaller variant without official
research and local validation. Neither the numeric checkpoint suffix nor its file size is a VRAM
tier or a measurement of inference peak memory.
This repository has not frozen an independently trusted upstream checkpoint SHA-256 allowlist.
The local hash gives run-to-run identity, but self-declared hash plus byte size alone is not proof
of byte-for-byte equivalence with the upstream object.

The shared default `minimum_free_vram_gib=28.0` is a pre-run admission floor: the production worker
reads free memory on its assigned GPU and declines the job when less is available. It is neither a
peak VRAM cap nor evidence of actual peak consumption; peak remains unavailable unless
independently measured. It is the conservative shared default for the final 48 GB-class `gfx1100`
platform, but does not validate or guarantee that a real query will fit. The
admission floor does not provide scheduling isolation; that comes from the GPU assignment/lease
described above. The adapter rejects thresholds below 24 GiB.

The adapter protocol contract has been exercised with a deterministic fake CLI, while the official
OpenFold3 0.4.3 source separately passes its ROCm/HIP/Triton/Evoformer validator in the AIAA-backed
overlay. No real OpenFold3 checkpoint inference, 1/2/4×`gfx1100` scheduling comparison,
PoseBusters parsing, or three-complex
bake-off has been run in this repository state, so none of those scientific/performance claims are
established yet.

## Quick-Vina selection adapter

`quick_vina_worker.py` implements the typed `vina-quick` engine between `SCREENED` and `SELECTED`.
It accepts only `protbind.quick-vina-input` schema v1.0, producer version 1.2, never a generic
`DOCKED` envelope. The input contains scalar screening/index/preparation commitments, the exact
receptor and environment lock, the mandatory docking-box receipt, and only the projected
microstate requests; it does not stage the 100k index into the worker. The selection preparation
and finalizer producer version is 2.5.

The semantic profile version is 1.3. It is CPU-only (`cpu=1`, exhaustiveness ≤16, modes ≤3;
defaults 8/1), rejects visible GPU masks, and is labeled `selection-pruning-only`. It calls the shared `vina_worker` implementation
internally, so receptor/ligand preparation, chemical-identity gates, tool attestation, score parsing,
and candidate failures are identical to full Vina. The adapter then wraps successes in purpose-
specific evidence and emits exact success/failure coverage. Failed requests have no score, pose, or
evidence. `WorkerResponse.outputs` is the complete recursively reachable artifact closure because
the host importer does not recursively import undeclared outputs.

The host validates request IDs, box, seed, CPU profile, outer and inner Vina evidence, environment
lock, counts, and the complete output closure before freezing top 16. A mandatory schema/producer
2.0 `protbind.docking-box-receipt` binds the receptor's complete `ArtifactRef` and SHA-256, source
kind, center/size, and `receptor-cartesian-angstrom` frame. Dimensions must each be 4–60 Å and
volume must be ≤27,000 Å³. The host reparses the exact receptor artifact, recomputes atom counts and
distances, and requires at least one standard-protein heavy atom inside the box. Its exact identity
must survive preparation → quick input → every request → inner and outer evidence → evaluation
batch/run metadata → final selection bundle.

Atom overlap is only a coordinate-frame plausibility check; it does not establish a biological
binding site. `user-center` and `user-residues` remain `user-hypothesis-only`. The
`co-crystal-ligand`, `fpocket-p2rank-consensus`, and `public-benchmark-reference` source kinds
require a hash-bound, coordinate-free `protbind.site-derivation-evidence` artifact that binds the
receptor, box, frame, derivation method, and non-empty SHA-256 source commitments while attesting
that validation reference coordinates were not exposed to screening. Even this verifies provenance
of the derivation, not biological-site truth. The host saves the batch and a receipt containing every returned
`ArtifactRef` before finalization, enabling resume without a second quick run. These scores cannot
enter `DOCKED` as final evidence: `vina_worker.py` reruns the selected candidates with its
independent evidence profile.

For an explicitly calibrated `both` case, selection 2.5 also revalidates the stored calibration
against its canonical source redock, exact Meeko prepared receptor/preparation receipt, native-derived
box, target ID, and PB-valid/symmetry-RMSD decision before either manual or automatic selection. The
quick input commits the resulting preparation SHA but never receives native-ligand coordinates.

Automatic v1 selection requires an explicit pocket center and box size. Until fpocket/P2Rank is
connected, ligand-only/residue-only cases fail explicitly with `SITE_DISCOVERY_UNAVAILABLE`; no
whole-protein docking box is inferred.

The historical 2026-07-23 profile 1.2/selection config 2.3/input producer 1.1 direct AIAA v3 smoke
completed 3/3 real Meeko/Vina requests with an exact 36-entry output closure and tool scores
`-1.942/-1.896/-1.753`. It used public 1CRN at a box with no verified biological-site derivation and
a demo index with `chemistry_verified=false`, and it ran under application offline policy rather
than the production bubblewrap boundary. It therefore proves only the historical adapter's direct
protocol/environment path—not binding, docking quality, ranking, throughput, OS isolation, or
Radeon performance. Evidence is under `experiment-results/aiaa-selection-quick-vina-20260723-v3`;
the smoke/provenance SHA-256 values are
`c0e077c2d8c24e59fc4f6d3eece777f1c455b5fd325a7c890152d724339c11ee` and
`b9b226eb718c2435f7450395f1ac40c2b1ae27a42ce40b02b8a316edfbae1536`. The unchanged environment
lock/runtime-assets identities are `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35`
and `e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`; quick/full adapter code
identities are `507fd3ac9d311cacd7df516e66d38043f46045d8727e6b36b58b489c8f742be9` and
`e800f8b94c41a343582742d3a8bfbfacaa44a5fbde868f4bfcb58c7deb054334`. The v2 directory remains
historical evidence and is not the current contract. The site-gate contract and runtime composite
hash have since changed, so neither v3 nor v2 is current evidence. The 2026-07-25 v4 direct smoke
uses profile 1.3/config 2.5/input producer 1.2/box receipt 2.0 and completed 3/3 requests with the
36-entry closure. Its smoke/provenance SHA-256 values are
`8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e` and
`b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f`; quick/full code SHA-256
values are `ed48d5b6884bd4a572cd7315d046b557b2bfd0ea3d1953dd72ee5acfa7458d14` and
`a181906553cc0960a9e51f02856df3b2bc527fd4a0aad24321d61bd92760f56c`. It remains a user-center,
unverified-chemistry, application-offline protocol smoke—not docking-quality, binding, isolation,
throughput, or Radeon-performance evidence.

A separate historical `chemistry_verified=true` v3 workflow bound docking-box receipt
`641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` into production preparation
and quick input, then reached the `SELECTED` worker launch. The host denied bubblewrap
loopback/network-namespace setup with `RTM_NEWADDR: Operation not permitted`. The run failed closed
as recoverable `WORKER_CRASH`, state `DEGRADED`, last completed stage `SCREENED`, with no quick
result. Its production manifest SHA-256 is
`730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746`. This host-capability failure
must not be worked around by disabling `isolate_network` or enabling a fixture bypass; production
isolation remains unproven. The 214-test count belongs to the historical v3 contract.

## Vina adapter

`vina_worker.py` is the offline `DOCKED` adapter. It is pinned to AutoDock Vina 1.2.7 and Meeko
0.7.1, requires exact RDKit, Gemmi, NumPy, and SciPy versions, and defaults to one CPU thread. It does not use or reserve
a Radeon GPU. Three absolute executable paths are mandatory: `vina`, `mk_prepare_receptor.py`, and
`mk_prepare_ligand.py`. Ambient `PATH` is never used to select these tools, and every child command
receives empty CUDA/HIP/ROCr device masks.

Before `DOCKED`, generate and attach the exact AIAA/core environment audit used by the worker:

```bash
scripts/aiaa-protbind.sh scripts/aiaa_environment_audit.py \
  --output experiment-results/aiaa-environment.json
protbind case attach RUN_ID --name vina_environment_lock \
  --file experiment-results/aiaa-environment.json --media-type application/json
```

Configure the worker through the same reviewed AIAA wrapper; do not use a Pixi environment or
install another Torch:

```toml
argv = ["/absolute/repo/scripts/aiaa-protbind.sh", "/absolute/repo/workers/vina_worker.py"]
```

The worker measures a canonical runtime-asset manifest containing the active Python executable,
Vina binary hash, both Meeko entry-point hashes, all six exact tool/package versions, and complete
path-relative file/hash manifests for the imported Meeko, RDKit, Gemmi, NumPy, and SciPy roots.
Request `weight_sha256` is this manifest digest. The
composite `code_sha256` is:

```text
code_sha256 = SHA256(canonical_json({
  "schema_version": "1.0",
  "adapter_sha256": SHA256(workers/vina_worker.py),
  "protbind_runtime_sha256": SHA256(canonical_json([
    ["src/protbind_agent/<relative>.py", SHA256(file)], ...
  ])),
  "environment_lock_sha256": SHA256(aiaa-environment.json),
  "runtime_assets_sha256": weight_sha256,
  "vina_version": "1.2.7",
  "meeko_version": "0.7.1"
}))
```

`model_revision` must be
`autodock-vina-1.2.7+meeko-0.7.1+rdkit-<exact>+gemmi-<exact>+numpy-<exact>+scipy-<exact>`.
The worker refuses relative/ambient
executables, version/hash drift, a missing lock, Vina seeds outside `[1, 2147483647]`, non-Vina
scoring, disconnected/metal/non-organic ligands, and receptors containing water, ligands, metals,
or non-standard residues. The conservative receptor gate requires a separately prepared,
protein-only PDB/mmCIF rather than silently deleting atoms.

Meeko entry points are executed with the same `sys.executable` whose package roots were attested;
their shebang cannot silently select a different Python environment.

For each frozen schema-2 `SELECTED` candidate (up to top 16), the adapter recovers the exact
canonical microstate and box from the selection bundle, creates one deterministic ETKDGv3
conformer, invokes Meeko, and then invokes the explicit Vina binary with the case seed, box,
`--cpu`, exhaustiveness, mode count, and energy range. Legacy schema-1 `COFOLDED` input remains
parseable but is not the schema-2 main path. Prepared receptor/ligand PDBQT files must contain finite atom records; ligand torsion
records are required. Every Vina mode must preserve the Meeko atom signature and contain exactly
one finite `REMARK VINA RESULT` record. The selected score is read only from the first, best-ordered
mode. The prepared ligand's supported PDBQT atom-type counts must also reproduce the frozen
heavy-element identity; Meeko `CG0..CG3` macrocycle carbons count as carbon while `G0..G3` glue
pseudoatoms do not count as chemical atoms. Across every Vina mode, all serial, residue, charge,
atom-type, ROOT/BRANCH, and torsion fields must remain identical to the prepared ligand; only
coordinates and Vina remarks may change. The prepared receptor must preserve the normalized input
protein's complete heavy-atom/residue multiset. Missing or malformed tool scores are never replaced.

The schema-2 `protbind.docking-bundle` binds the upstream selection bundle and complete upstream ID
list, original and prepared receptor, run metadata, successful candidates, and structured
candidate failures. Every success binds its parent candidate/microstate, receptor, prepared ligand,
seed, box, and a `protbind.tool-evidence` artifact. Its canonical `pose`/`pose_sdf` is the top Vina
mode reconstructed as SDF through Meeko; `pose_pdbqt` keeps the raw top mode,
`all_modes_pdbqt` keeps raw Vina output, and `all_modes_sdf` keeps every reconstructed mode. A
`protbind.pose-extraction-receipt` binds these artifacts and attests elements/isotopes, formal
charge, bond order/aromaticity, stereochemistry, hydrogen count, atom mapping, score/mode count and
coordinate mapping. The bundle-level `protbind.receptor-preparation-receipt` binds the input
receptor, normalized PDB, and prepared PDBQT while attesting finite coordinates and complete
heavy-atom/residue identity. Optional cofold structure may be carried as additional evidence but is
not required. The
union of success parent IDs and failure candidate IDs must exactly equal the frozen upstream list;
no candidate can disappear silently. Candidate-level chemistry/preparation/Vina failures carry an
error code and no score. Global provenance, runtime, receptor-preparation, or protocol failures fail
the worker itself.

There is not yet a reviewed upstream binary/package allowlist. Therefore even a successful real
run is labeled `attested-local-autodock-vina`, with trust level
`hash-attested-local-without-reviewed-upstream-allowlist`; it is never called an official runtime
merely because a self-attested executable prints version 1.2.7.

`PROTBIND_TEST_RUNTIME=1` is reserved for direct contract tests and is rejected by `WorkerConfig`.
Fixture output is labeled `test-fixture-vina`, reports no peak VRAM, and is not scientific evidence.
The checked-in tests use controlled fake executables only to exercise protocol validation. In
addition, six public inputs were attempted in a sealed-reference known-site pilot: three completed
real Meeko/Vina/PoseBusters/sPyRMSD redocking and three failed closed at declared input/preparation
gates. Among the three completed cases, top-1 recovery was 2/3 and reference-aware top-5 oracle
recovery was 3/3. This is not a ten-case regression, blind docking, virtual-screen success rate,
affinity evidence, or a six-input 3/3 success claim.

## Validation adapter

`validation_worker.py` is the offline `VALIDATED` adapter. In schema 2 the host normally derives
both support artifacts automatically: `build_validation_input_batch` consumes the exact completed
`protbind.docking-bundle`, while `build_validation_toolchain` hashes the installed AIAA packages
and creates a matching `WorkerProvenance`. The configured provenance must equal that locally
measured binding; placeholders or ambient package drift fail closed. Configure the worker through
the reviewed core wrapper:

```toml
argv = ["/absolute/repo/scripts/aiaa-protbind.sh", "/absolute/repo/workers/validation_worker.py"]
```

Manually attached `support_validation_batch`/`support_validation_toolchain` remain accepted as
frozen expert inputs, but are not required for the normal DOCKED→VALIDATED path.

The toolchain artifact itself is the model identity: `model_revision` is
`validation-toolchain:<artifact-sha256>` and `weight_sha256` is that same SHA-256. PoseBusters is
mandatory and the generated manifest pins both `dock` and `redock` configurations; an unpinned optional tool is written into
`unsupported_reasons`, while a tool that is explicitly pinned but absent/mismatched fails closed.
Each installed package is checked
against its exact distribution version and a path/hash manifest of its imported package root.
The adapter and all `src/protbind_agent/**/*.py` sources form the composite `code_sha256`.

The input batch binds the exact prior docking bundle and every successful DOCKED candidate. The
canonical docked ligand is the attested SDF and the receptor is the normalized PDB from the Vina
bundle; optional cofold structure is carried only when present. PoseBusters always gates the docked
pose and independently reports an optional cofold result; neither engine reports an experimental
truth. An independent reference pose can be attached only after `DOCKED` and before `VALIDATED` as
`support_reference_pose`; the generated batch marks it `VALIDATION_ONLY`, selects PoseBusters
`redock`, and enables symmetry-aware RMSD. A ligand hypothesis is never treated as an aligned
reference pose.
ProLIF reports a Jaccard/Tanimoto similarity over residue/interaction labels. OpenMM accepts only a
pre-serialized System plus coordinates and an explicit `CPU` or `HIP` platform, runs at most 1000
local-minimization iterations, and reports a bounded geometry gate. It does **not** call this a
stability simulation or treat particle-count equality as attested parameterization. HIP uses one
leased/masked physical GPU and explicit logical `DeviceIndex=0`; CPU uses one thread.

Every metric has a returned `protbind.tool-evidence` artifact with exact input refs and a runtime
attestation. The host additionally checks full candidate coverage, toolchain/runtime versions and
the prepared refs against the frozen validation batch. `preparation_attested=true` requires the
exact non-fixture `protbind.pose-extraction-receipt` and
`protbind.receptor-preparation-receipt` emitted by Vina, with every identity check true and all
artifact references matching. Missing, incomplete, mismatched, or fixture receipts cap a passing
pose at `HYPOTHESIS_ONLY`; hard validation failures remain `REJECTED`. With an authorized reference,
PB-valid plus symmetry RMSD ≤2 Å is `REDOCKING_RECOVERED`; without reference, independently valid
method/IFP agreement may be `METHOD_CONSENSUS`. Neither term is experimental binding support.
Real PoseBusters and sPyRMSD were exercised in the three completed known-site redocking cases above.
ProLIF IFP, strain analysis, OpenMM relaxation, a fixed ten-case regression, and complete
failure/exclusion denominators remain pending.

## Other worker status

`esmfold_v1_worker.py` uses the fair-esm API and requires a complete local three-file set: ESMFold,
the frozen ESM2 3B backbone, and its contact-regression checkpoint. Their ordered hashes and sizes
form one composite `weight_sha256`; merely pinning the small ESMFold wrapper checkpoint is
insufficient. The worker forces PyTorch's restricted `weights_only` loader, statically rejects
checkpoint globals outside the reviewed argparse/OmegaConf allowlist, points fair-esm's hub helper
at a verified `<torch-hub>/checkpoints` layout, and runs inside the disconnected worker namespace.
Its composite code identity binds the environment lock, fair-esm/OmegaConf/legacy OpenFold sources,
the OpenFold attention extension, selected PyTorch loader/core ROCm files, the Python executable,
and ProtBind runtime. It rejects architecture
spoofing and conflicting device masks, requires at least 12 GiB free by default, validates the
returned PDB's exact sequences/N-CA-C/finite coordinates/altloc state, and retries chunk sizes
128 → 64 → 32. The workflow/smoke runner—not the standalone worker process—acquires the shared
same-user host GPU lease before launch. User-site Python packages are excluded from both provenance
calculation and the isolated worker runtime.

A post-hardening 24-residue offline AIAA smoke run completed on one W7900/gfx1100 while leaving the
second card unused: 8,496,247,808 peak allocated bytes, 26.112 s model load, 3.653 s inference,
and 37.425 s end to end (warm local filesystem/cache conditions). Repeated setup runs returned the
same byte-identical structure artifact. The worker applied an exact-hash
Python 3.12 compatibility shim to fair-esm 2.0.0's two dataclass defaults; it did not alter tensor
math or weights. This validates the local
protocol/runtime path only and is not a benchmark. The imported checkpoint set is
hash-pinned but has not been established as byte-identical to every object currently served by the
upstream download endpoint, so the result is not an official-release equivalence or accuracy
claim. ESMFold predicts a receptor from one/two canonical protein chains; it is not a ligand-pose
or complex-cofolding engine and is not substituted for OpenFold3 evidence.

ESMFold2 is future-only: no runnable worker is integrated, and it cannot enter the workflow unless
the documented offline three-complex gate is completed. AlphaFold DB import is likewise future-only;
the implemented authorized structure-import path is RCSB. The Vina and validation adapters have
deterministic protocol coverage plus the limited real redocking pilot described above.

Re-run the ESMFold protocol check without logging the sequence or internal checkpoint paths in its
JSON result:

```bash
scripts/aiaa-esmfold-v1.sh python scripts/esmfold_v1_smoke.py \
  --model <local-esmfold.pt> \
  --esm2-model <checkpoint-dir>/esm2_t36_3B_UR50D.pt \
  --esm2-regression <checkpoint-dir>/esm2_t36_3B_UR50D-contact-regression.pt \
  --environment-lock requirements/aiaa-esmfold-v1-overlay.lock.txt \
  --workspace <private-smoke-workspace> \
  --receipt-output <private-receipt.json> --sequence <protein> --device 0

protbind case attach RUN_ID --name esmfold_receipt \
  --file <private-receipt.json> --media-type application/json
```

Create or stop the case at `INPUT_VALIDATED`, then attach this receipt before `RECEPTOR_READY`; once
`RECEPTOR_READY` is recorded, receptor inputs are frozen. Attaching a generic `esmfold_structure`
is refused. The verified receipt is an import receipt, not an in-workflow folding request: the host
validates its exact target-sequence identity, imports/preserves the original worker-produced
structure and result metadata, and registers their complete seed/weight/code/environment/hardware
lineage in the exact-sequence cache.
