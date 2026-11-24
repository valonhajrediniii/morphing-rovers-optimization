TAMRO v5.8 - Local Mask Sensitivity Search

Start from the best chromosome produced by TAMRO v5.6 (~3.998486).
Probe index 138, its eight neighbors in the same 11x11 terrain mask,
and two-coordinate interactions. Select promising candidates with
partial scenario screening; validate every new record using the official
30-scenario morphing_rover_UDP.fitness.

Files:
  solution/tamro_v5_8.py
  run_v5_8.py

Requirements: local ESA simulator at external/esa/morphing_udp.py,
including its terrain maps and existing Python 3.12 virtual environment.

Run:
  python run_v5_8.py --rounds 1 --trials 24 --seed 2041 --target 3.9

Resume with the same command; state saved in
  solution/results/tamro_v5_8_state.json
  solution/results/tamro_v5_8_best.npy
  solution/results/tamro_v5_8_history.csv

Checkpoint acceptance is always based on official full fitness, not
partial score. This method is exploratory and has no guaranteed result.
