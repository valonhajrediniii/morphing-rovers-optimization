TAMRO v5.7 – Winner-Direction Trust Region

Requires the ESA SpOC-2 simulator in external/esa/morphing_udp.py and its Maps.
Install the project's Python dependencies and activate its .venv first.

Starting point: solution/results/tamro_v5_6_best.npy, falling back to
solution/checkpoints/tamro_v5_6_best.npy. Direction anchor: v5.5 checkpoint.

Run a short test:
    python -m py_compile solution/tamro_v5_7.py run_v5_7.py
    python run_v5_7.py --rounds 1 --trials 24 --seed 2040 --target 3.9

For additional rounds (resumes from tamro_v5_7_state.npz):
    python run_v5_7.py --rounds 3 --trials 32 --target 3.9

All results are local to solution/results/ (ignored by Git).
Any improvement is accepted only after verifying ESA's official fitness
across 30 scenarios. Preserve checkpoints before making any merge.
