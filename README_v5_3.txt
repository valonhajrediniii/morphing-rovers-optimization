TAMRO v5.3 - Paired Coordinate / Adaptive Trust Regions

Run from repository root:
  python -m py_compile solution/tamro_v5_3.py run_v5_3.py
  python run_v5_3.py --rounds 1 --trials 24 --seed 2036 --target 4.0

Starts from solution/results/tamro_v5_best.npy, or solution/checkpoints/tamro_v5_best.npy.
Resumes from solution/results/tamro_v5_3_state.npz if present.
Never accepts a record based solely on quick/balanced scores.
The original ESA simulator/data are expected at external/esa/ (not included in this package).
This package is an independent experiment; no score improvement is guaranteed.
