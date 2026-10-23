import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESA = ROOT / "external" / "esa"
SOLUTION = ROOT / "solution"

if not (ESA / "morphing_udp.py").exists():
    raise SystemExit(
        "ESA simulator not found in external/esa/"
    )

sys.path.insert(0, str(ESA))
sys.path.insert(0, str(SOLUTION))

# Kodi i ESA-s perdor relative paths per data/.
os.chdir(ESA)

from main import main

if __name__ == "__main__":
    main()
