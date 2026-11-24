"""Run TAMRO v5.1 from the repository root using the local ESA simulator."""
import os
import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESA = ROOT / 'external' / 'esa'
SOLUTION = ROOT / 'solution'
TARGET = SOLUTION / 'tamro_v5_1.py'
if not (ESA / 'morphing_udp.py').exists():
    raise SystemExit('ESA simulator missing: external/esa/morphing_udp.py')
if not TARGET.exists():
    raise SystemExit('Missing optimizer: solution/tamro_v5_1.py')
sys.path.insert(0, str(ESA))
sys.path.insert(0, str(SOLUTION))
os.chdir(ESA)
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name='__main__')
