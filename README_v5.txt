TAMRO v5 Hybrid: local search (v4.4), block differential evolution,
diagonal evolution strategy for biases, and elite crossover.

Place files in repo:
  solution/tamro_v5_hybrid.py
  run_v5_hybrid.py

Requirements: same existing ESA simulator and Python venv as TAMRO v4.4.
Starts from local v4.4 elite state if available; otherwise from
solution/checkpoints/tamro_v4_4_best.npy.

Run from Git Bash repo root:
  python -m py_compile solution/tamro_v5_hybrid.py run_v5_hybrid.py
  python run_v5_hybrid.py --rounds 1 --trials 24 --seed 2033 --target 4.0
  python run_v5_hybrid.py --rounds 5 --trials 36 --target 4.0

State/checkpoint/history go into solution/results, which is gitignored.
Only official ESA 30-scenario evaluation confirms new best; 4.0 is a target,
NOT a promise of improvement.
