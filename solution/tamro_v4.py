import argparse
import csv
import time
from pathlib import Path

import numpy as np

from morphing_udp import (
    Rover,
    morphing_rover_UDP,
    MASK_CENTRES,
    NUM_MODE_PARAMETERS,
    NUM_BIASES,
    NUM_WEIGHTS,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"
CHECKPOINT = RESULTS / "tamro_v4_best.npy"
HISTORY = RESULTS / "tamro_v4_history.csv"

N_MASK = NUM_MODE_PARAMETERS
N_BIAS = NUM_BIASES
N_WEIGHTS = NUM_WEIGHTS
N_CONTINUOUS = N_MASK + N_BIAS + N_WEIGHTS


def repair(x, lower, upper):
    x = np.clip(x, lower, upper)
    x[-7:] = np.rint(x[-7:])

    for centre in MASK_CENTRES:
        x[centre] = max(x[centre], 1e-16)

    return x


def mutate(parent, rng, sigma, lower, upper):
    x = parent.copy()
    operation = int(rng.integers(0, 5))

    if operation == 0:
        # Local 3x3 terrain-mask mutation
        mode = int(rng.integers(0, 4))
        row = int(rng.integers(0, 9))
        col = int(rng.integers(0, 9))

        start = mode * 121
        mask = x[start:start + 121].reshape(11, 11)

        mask[row:row + 3, col:col + 3] += (
            rng.normal(0, sigma, (3, 3))
        )
        name = "TERRAIN PATCH"

    elif operation == 1:
        # Sparse neural-network weight mutation
        start = N_MASK + N_BIAS
        indices = rng.choice(
            np.arange(start, N_CONTINUOUS),
            size=64,
            replace=False,
        )
        x[indices] += rng.normal(0, sigma, 64)
        name = "NEURAL SPARSE"

    elif operation == 2:
        # Contiguous neural-network weight block
        start_min = N_MASK + N_BIAS
        start = int(rng.integers(
            start_min, N_CONTINUOUS - 127
        ))
        x[start:start + 128] += rng.normal(
            0, sigma, 128
        )
        name = "NEURAL BLOCK"

    elif operation == 3:
        # Network output layer: 80 weights and 2 biases
        weight_start = N_CONTINUOUS - 80
        bias_start = N_MASK + N_BIAS - 2

        indices = rng.choice(
            np.arange(weight_start, N_CONTINUOUS),
            size=16,
            replace=False,
        )
        x[indices] += rng.normal(0, sigma, 16)
        x[bias_start:bias_start + 2] += (
            rng.normal(0, sigma, 2)
        )
        name = "OUTPUT CONTROL"

    else:
        # Discrete architecture hyperparameter
        index = int(rng.integers(
            N_CONTINUOUS, len(x)
        ))
        x[index] = rng.integers(
            int(lower[index]),
            int(upper[index]) + 1
        )
        name = "ARCHITECTURE"

    return repair(x, lower, upper), name


def partial_fitness(udp, chromosome, scenarios):
    rover = Rover(chromosome.tolist())
    scores = []

    for map_id, scenario_id in scenarios:
        result = udp.run_single_scenario(
            rover, map_id, scenario_id
        )
        scores.append(
            float((1 + result[0]) * result[1])
        )

    return float(np.mean(scores))


def select_scenarios(rng):
    quick = []
    balanced = []

    for map_id in range(6):
        selected = rng.choice(
            5, size=2, replace=False
        )
        first, second = map(int, selected)

        quick.append((map_id, first))
        balanced.extend([
            (map_id, first),
            (map_id, second),
        ])

    return quick, balanced


def append_history(record):
    fields = [
        "round", "trial", "operation", "sigma",
        "quick", "balanced", "full",
        "best_full", "improved",
    ]

    exists = HISTORY.exists()

    with HISTORY.open("a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def optimize(rounds=1, trials=12, seed=2028):
    RESULTS.mkdir(parents=True, exist_ok=True)

    udp = morphing_rover_UDP()
    rng = np.random.default_rng(seed)

    lower, upper = udp.get_bounds()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    source = CHECKPOINT
    if not source.exists():
        source = RESULTS / "tamro_best.npy"

    if not source.exists():
        raise FileNotFoundError(
            "Run python run.py first to generate "
            "solution/results/tamro_best.npy"
        )

    best = np.asarray(
        np.load(source), dtype=float
    ).copy()

    if len(best) != len(lower):
        raise ValueError("Invalid chromosome length")

    best = repair(best, lower, upper)

    best_full = float(
        udp.fitness(best.tolist())[0]
    )

    if not np.isfinite(best_full):
        raise ValueError("Invalid starting fitness")

    np.save(CHECKPOINT, best)

    print("=== TAMRO v4 Optimization ===")
    print(f"Starting fitness: {best_full:.6f}")
    print(f"Rounds: {rounds}")
    print(f"Trials per round: {trials}", flush=True)

    sigma_pool = np.array([
        0.01, 0.03, 0.1, 0.3, 1.0, 3.0
    ])

    start_time = time.perf_counter()

    for round_id in range(1, rounds + 1):
        print(
            f"\n=== ROUND {round_id}/{rounds} ===",
            flush=True
        )

        quick_cases, balanced_cases = (
            select_scenarios(rng)
        )

        candidates = []
        records = {}

        for trial in range(1, trials + 1):
            sigma = float(rng.choice(sigma_pool))

            candidate, operation = mutate(
                best, rng, sigma, lower, upper
            )

            quick = partial_fitness(
                udp, candidate, quick_cases
            )

            records[trial] = {
                "round": round_id,
                "trial": trial,
                "operation": operation,
                "sigma": sigma,
                "quick": quick,
                "balanced": "",
                "full": "",
                "best_full": best_full,
                "improved": False,
            }

            if np.isfinite(quick):
                candidates.append(
                    (quick, trial, candidate)
                )

            print(
                f"Quick {trial}/{trials} | "
                f"{operation} | "
                f"Fitness: {quick:.6f}",
                flush=True
            )

        candidates.sort(key=lambda x: x[0])

        balanced_candidates = []

        for quick, trial, candidate in candidates[:4]:
            balanced = partial_fitness(
                udp, candidate, balanced_cases
            )

            records[trial]["balanced"] = balanced

            if np.isfinite(balanced):
                balanced_candidates.append(
                    (balanced, trial, candidate)
                )

            print(
                f"Balanced trial {trial}: "
                f"{balanced:.6f}",
                flush=True
            )

        balanced_candidates.sort(
            key=lambda x: x[0]
        )

        for balanced, trial, candidate in (
            balanced_candidates[:2]
        ):
            full = float(
                udp.fitness(candidate.tolist())[0]
            )

            improved = (
                np.isfinite(full)
                and full < best_full
            )

            if improved:
                best = candidate.copy()
                best_full = full

                np.save(CHECKPOINT, best)

                print(
                    f"NEW RECORD: {best_full:.6f}",
                    flush=True
                )

            records[trial]["full"] = full
            records[trial]["improved"] = improved

            print(
                f"Full trial {trial}: {full:.6f}",
                flush=True
            )

        for record in records.values():
            record["best_full"] = best_full
            append_history(record)

        print(
            f"Round {round_id} best: "
            f"{best_full:.6f}",
            flush=True
        )

    elapsed = time.perf_counter() - start_time

    print("\n=== FINAL RESULT ===")
    print(f"Best full fitness: {best_full:.6f}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"History: {HISTORY}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds", type=int, default=1
    )
    parser.add_argument(
        "--trials", type=int, default=12
    )
    parser.add_argument(
        "--seed", type=int, default=2028
    )

    args = parser.parse_args()

    if args.rounds < 1 or args.trials < 4:
        parser.error(
            "rounds >= 1 and trials >= 4 required"
        )

    optimize(
        rounds=args.rounds,
        trials=args.trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
