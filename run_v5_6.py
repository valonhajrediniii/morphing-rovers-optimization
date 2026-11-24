"""Launch TAMRO v5.6 from the repository root with ESA's data directory."""
import os
import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESA = ROOT / 'external' / 'esa'
SOLUTION = ROOT / 'solution'
if not (ESA / 'morphing_udp.py').exists():
    raise SystemExit('ESA simulator missing: external/esa/morphing_udp.py')
sys.path.insert(0, str(ESA))
sys.path.insert(0, str(SOLUTION))
os.chdir(ESA)
sys.argv[0] = str(SOLUTION / 'tamro_v5_6.py')
runpy.run_path(str(SOLUTION / 'tamro_v5_6.py'), run_name='__main__')
