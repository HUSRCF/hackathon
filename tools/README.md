# Local third-party runtime assets

`scripts/bootstrap-aiaa-protbind.sh --download-vina` downloads the official
AutoDock Vina 1.2.7 Linux x86_64 release binary to `tools/bin/vina`.

- Source: <https://github.com/ccsb-scripps/AutoDock-Vina/releases/tag/v1.2.7>
- Expected SHA-256: `f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644`
- The binary is ignored by Git and must not be redistributed as part of this repository.

The worker records and re-verifies the runtime asset hash for every evidence-grade docking run.
