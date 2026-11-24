TAMRO v5.5 – Active-Parameter Paired Search

Files:
  solution/tamro_v5_5.py
  run_v5_5.py

The optimizer requires the project's original tamro_v4_4.py and
external ESA simulator to be installed locally as with older TAMRO versions.

Starter (Git Bash, from project root):
  python -m py_compile solution/tamro_v5_5.py run_v5_5.py
  python run_v5_5.py --rounds 1 --trials 24 --seed 2038 --target 4.0

Resume (after first run):
  python run_v5_5.py --rounds 3 --trials 40 --target 4.0

It loads the local solution/results/tamro_v5_best.npy or the committed
solution/checkpoints/tamro_v5_best.npy if there is no v5.5 state.

Outputs under ignored solution/results/:
  tamro_v5_5_state.npz - atomic checkpoint plus RNG state
  tamro_v5_5_best.npy  - best solution
  tamro_v5_5_history.csv - trial history

This file does not provide or claim an official ESA record. All fitness
claims must be independently checked with udp.fitness(chromosome.tolist())[0].
