# Benchmarks

Different files can be run to verify timings and statistics. The security level
and the number of runs can be set inside the files.

- Example usage (new implementation Dartois and Duparc, 2026): `example_qt_Fp.py`
- Example usage (former implementation): `example.py`
- Example usage with sage ideals (former implementation): `example_sage_ideal.py`
- Benchmarks (former implementation vs new implementation): `compare.py`
- Timing and statistics for norm equation (former implementation): `time_neq.py`
- Timings qt-Pegasis (former implementation): `time_qtp.py`
- Timings qt-Pegasis from random curve (former implementation): `time_rand_curve.py`
- Verify correctness agains PEGASIS (former implementation of qt-Pegasis vs. PEGASIS): `test_qtp.py` (requires PEGASIS; to
  download it, run in the `benchmark/` folder `git clone --recurse-submodules
  git@github.com:pegasis4d/pegasis.git .old_pegasis`

To skip assertions, run timings with `sage --python -O [script.py]`.
