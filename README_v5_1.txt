TAMRO v5.1 Adaptive Local Refinement
====================================
Install in the root of morphing-rovers-optimization:
  solution/tamro_v5_1.py
  run_v5_1.py

Dependencies: existing virtual environment; local external/esa/morphing_udp.py

Initial best: solution/results/tamro_v5_best.npy or solution/checkpoints/tamro_v5_best.npy

Run a first test:
  python run_v5_1.py --rounds 1 --trials 24 --seed 2034 --target 4.0

Resume with additional rounds:
  python run_v5_1.py --rounds 5 --trials 32 --target 4.0

Results (local, ignored by git):
  solution/results/tamro_v5_1_best.npy
  solution/results/tamro_v5_1_state.npz
  solution/results/tamro_v5_1_history.csv

Official ESA fitness is checked on all 30 scenarios before accepting a new record.
There is no guarantee of reaching 4.0. A dry-run mock test validates the
Python control flow and resume behavior; real ESA performance must be measured
in the user's environment.
