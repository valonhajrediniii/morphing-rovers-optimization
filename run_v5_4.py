"""Run TAMRO v5.4 using the locally installed ESA simulator."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESA = ROOT / 'external' / 'esa'
SOLUTION = ROOT / 'solution'
if not (ESA / 'morphing_udp.py').exists():
    raise SystemExit('Simulator missing: external/esa/morphing_udp.py')
sys.path.insert(0, str(ESA))
sys.path.insert(0, str(SOLUTION))
os.chdir(ESA)
script = SOLUTION / 'tamro_v5_4.py'
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name='__main__')
