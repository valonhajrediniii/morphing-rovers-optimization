"""Run TAMRO v5.3 using the locally installed ESA simulator."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESA = ROOT / "external" / "esa"
SOLUTION = ROOT / "solution"

if not (ESA / "morphing_udp.py").exists():
    raise SystemExit("Missing external/esa/morphing_udp.py")

sys.path.insert(0, str(ESA))
sys.path.insert(0, str(SOLUTION))
os.chdir(ESA)
sys.argv[0] = str(SOLUTION / "tamro_v5_3.py")
runpy.run_path(str(SOLUTION / "tamro_v5_3.py"), run_name="__main__")
