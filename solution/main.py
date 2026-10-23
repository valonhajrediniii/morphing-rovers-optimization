
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morphing_udp import morphing_rover_UDP
from tamro import optimize


def main():
    udp = morphing_rover_UDP()
    chromosome = udp.example()

    print("=== TAMRO Optimization ===")

    best, fitness, history = optimize(
        udp,
        chromosome,
        iterations=6,
        seed=2026
    )

    results_dir = ROOT / "solution" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "tamro_best.npy", best)

    np.savetxt(
        results_dir / "tamro_history.csv",
        np.array(history),
        delimiter=",",
        header="best_fitness",
        comments=""
    )

    print(f"\nFinal best fitness: {fitness:.6f}")
    print("Results saved successfully.")


if __name__ == "__main__":
    main()
