TAMRO v5.4 – paired micro local search, adaptive trust regions.

Requires original ESA code/data installed under external/esa and a valid
v5 checkpoint in solution/results/tamro_v5_best.npy or solution/checkpoints/tamro_v5_best.npy.

Test (Git Bash, activated ESA virtual environment):
python -m py_compile solution/tamro_v5_4.py run_v5_4.py
python run_v5_4.py --rounds 1 --trials 24 --seed 2037 --target 4.0

Resume:
python run_v5_4.py --rounds 3 --trials 24 --target 4.0

Checkpoints under solution/results/; keep that directory local and backed up.
Only official 30-case ESA fitness can set a record. No guarantee of success.
