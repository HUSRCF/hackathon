# TriPharm HIP triangle matcher

This directory contains the first ROCm-native kernel boundary. It compares original-order
three-point feature types and all six atom correspondences, atomically accumulating a 64-bit
query-triangle mask and per-query minimum normalized error per molecule.

The executable is intentionally labelled a **triangle-match microbenchmark**. It is not yet the
full persisted-index top-512 pipeline and must not be reported as the planned 100k end-to-end
screening number.

```bash
cmake -S kernels/tripharm_hip -B build/tripharm_hip -DCMAKE_BUILD_TYPE=Release
cmake --build build/tripharm_hip -j
./build/tripharm_hip/tripharm_hip_benchmark \
  --candidates 100000 --molecules 100000 --queries 64 --repetitions 7
```

The benchmark rejects `HSA_OVERRIDE_GFX_VERSION`, reports the actual `gcnArchName`, separates CPU,
H2D, kernel, and D2H timing, and exits nonzero unless CPU/HIP match masks are exact and recall is at
least 0.999.
