from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morphing_udp import (
    MASK_SIZE, MASK_CENTRES, NUM_MODE_PARAMETERS,
    NUM_BIASES, NUM_WEIGHTS, morphing_rover_UDP
)
from evaluator import RoverEvaluator


def mutate(parent, rng, sigma):
    candidate = parent.copy()
    operation = int(rng.integers(0, 3))

    if operation == 0:
        mode = int(rng.integers(0, 4))
        row = int(rng.integers(0, MASK_SIZE - 2))
        col = int(rng.integers(0, MASK_SIZE - 2))

        start = mode * MASK_SIZE**2
        mask = candidate[start:start + MASK_SIZE**2]
        mask = mask.reshape(MASK_SIZE, MASK_SIZE)

        mask[row:row+3, col:col+3] += rng.normal(
            0, sigma, (3, 3)
        )

    elif operation == 1:
        start = NUM_MODE_PARAMETERS + NUM_BIASES + NUM_WEIGHTS - 80
        indices = rng.choice(
            np.arange(start, start + 80),
            size=12, replace=False
        )
        candidate[indices] += rng.normal(0, sigma, len(indices))

    else:
        start = NUM_MODE_PARAMETERS
        indices = rng.choice(
            np.arange(start, start + NUM_BIASES),
            size=8, replace=False
        )
        candidate[indices] += rng.normal(0, sigma, len(indices))

    candidate = np.clip(candidate, -100.0, 100.0)

    for centre in MASK_CENTRES:
        candidate[centre] = max(candidate[centre], 1e-16)

    return candidate


def optimize(udp, initial, trials=12, top_k=3, seed=2026):
    rng = np.random.default_rng(seed)
    evaluator = RoverEvaluator(udp)

    best = np.asarray(initial, dtype=np.float64).copy()
    best_full = evaluator.full_fitness(best.tolist())

    candidates = []
    print(f"Starting full fitness: {best_full:.6f}")

    for i in range(trials):
        sigma = float(rng.choice([0.03, 0.1, 0.3, 1.0]))
        candidate = mutate(best, rng, sigma)
        quick = evaluator.quick_fitness(candidate.tolist())

        candidates.append((quick, i, candidate))
        print(f"Quick {i+1}/{trials}: {quick:.6f}", flush=True)

    candidates.sort(key=lambda item: item[0])

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(parents=True, exist_ok=True)

    for rank, (quick, index, candidate) in enumerate(
        candidates[:top_k], start=1
    ):
        full = evaluator.full_fitness(candidate.tolist())

        print(
            f"Full {rank}/{top_k}: "
            f"trial={index+1}, quick={quick:.6f}, "
            f"full={full:.6f}",
            flush=True
        )

        if np.isfinite(full) and full < best_full:
            best = candidate.copy()
            best_full = full
            np.save(results / "tamro_v2_best.npy", best)
            print("NEW BEST SOLUTION SAVED!", flush=True)

    return best, best_full


if __name__ == "__main__":
    udp = morphing_rover_UDP()

    results = Path(__file__).resolve().parent / "results"
    source = results / "tamro_best.npy"

    if not source.exists():
        raise FileNotFoundError(
            "Run TAMRO v1 first to generate tamro_best.npy"
        )

    initial = np.load(source)
    best, fitness = optimize(udp, initial)

    np.save(results / "tamro_v2_best.npy", best)
    print(f"\nFinal verified fitness: {fitness:.6f}")
