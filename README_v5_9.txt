TAMRO v5.9 -- Scenario-balanced island search

Starts from tamro_v5_6_best.npy; builds a small elite archive from earlier original checkpoints.
Samples 24 perturbations with 8 operators, screens using difficult cases + diverse cases,
then validates selected candidates with official 30-scenario fitness.

Git Bash:
    python -m py_compile solution/tamro_v5_9.py run_v5_9.py
    python run_v5_9.py --rounds 1 --trials 24 --seed 2042 --target 3.9

Resume:
    python run_v5_9.py --rounds 3 --trials 32 --target 3.9

Only accepting improvements on official ESA full fitness. Keeps own checkpoint in
solution/results/. Never delete solution/results/ while optimization is active.
The original ESA reference implementation/data is expected under external/esa/.
Not an official ESA solution. Research and educational use; see ESA licensing.
