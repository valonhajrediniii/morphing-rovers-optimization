TAMRO v5.6 - winner-direction line search and local fusion
=========================================================

This optimization is original continuation of TAMRO v5.5. It relies on
independently sourced ESA SpOC2 Morphing Rovers simulator in external/esa,
with the repository's solution/tamro_v4_4.py compatibility helpers.

Fresh starting checkpoint:
  solution/results/tamro_v5_5_best.npy
  OR solution/checkpoints/tamro_v5_5_best.npy

Previous best anchor:
  solution/results/tamro_v5_best.npy
  OR solution/checkpoints/tamro_v5_best.npy

From the project root, after activating the same Python venv as before:
  python -m py_compile solution/tamro_v5_6.py run_v5_6.py
  python run_v5_6.py --rounds 1 --trials 24 --seed 2039 --target 4.0

Resume:
  python run_v5_6.py --rounds 3 --trials 32 --target 4.0

Writes to (local, gitignored):
  solution/results/tamro_v5_6_best.npy
  solution/results/tamro_v5_6_state.npz
  solution/results/tamro_v5_6_history.csv

State includes last best, prior best direction, step sizes and PRNG state.
Checkpointing happens at round end and immediately upon an official
improvement. If interrupted mid-round, its other attempts are not logged.

Only a score returned from ESA's 30-scenario official evaluator and
confirmed by the corresponding 30-scenario scores is considered a record.
Improvement below 4.0 is a goal, not a guarantee.
