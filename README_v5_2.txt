TAMRO v5.2: Scenario-aware adaptive local refinement

Requirements: existing Morphing Rovers optimization repo, ESA simulator under
external/esa/, existing packages numpy and torch, and TAMRO v5 checkpoint.

In Git Bash from the repo, with virtual environment activated:
    python -m py_compile solution/tamro_v5_2.py run_v5_2.py
    python run_v5_2.py --rounds 1 --trials 24 --seed 2035 --target 4.0
    python run_v5_2.py --rounds 5 --trials 32 --target 4.0

The initial candidate is loaded from solution/results/tamro_v5_best.npy,
falling back to solution/checkpoints/tamro_v5_best.npy.
The simulator's official fitness is used to confirm every improvement.
The quick 12-scenario and balanced 22-scenario scores are screening-only.

Local outputs (keep these!):
    solution/results/tamro_v5_2_best.npy
    solution/results/tamro_v5_2_state.npz
    solution/results/tamro_v5_2_history.csv

For reproducibility, checkpoint and log can be copied to
solution/checkpoints/ after independent verification, then committed.

TAMRO v5.2 does NOT modify TAMRO v5.0 checkpoint/state/history.
