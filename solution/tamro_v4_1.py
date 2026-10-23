import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from morphing_udp import morphing_rover_UDP, Rover
from tamro_v4 import mutate, repair

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"

CHECKPOINT = RESULTS / "tamro_v4_1_best.npy"
HISTORY = RESULTS / "tamro_v4_1_history.csv"

# Skenarët më të vështirë sipas analizës sonë.
HARD_SCENARIOS = [
    (3, 4),  # Map 4, Scenario 5
    (2, 4),  # Map 3, Scenario 5
    (3, 3),  # Map 4, Scenario 4
    (2, 1),  # Map 3, Scenario 2
    (1, 4),  # Map 2, Scenario 5
]


def evaluate(udp, chromosome, scenarios):
    rover = Rover(chromosome.tolist())
    scores = []

    for map_id, scenario_id in scenarios:
        result = udp.run_single_scenario(
            rover, map_id, scenario_id
        )
        scores.append(float((1 + result[0]) * result[1]))

    return float(np.mean(scores))


def choose_scenarios(rng):
    # Nga një skenar për çdo hartë.
    quick = []
    for map_id in range(6):
        scenario = int(rng.integers(0, 5))
        quick.append((map_id, scenario))

    # Të gjithë skenarët e vështirë + skenarë të tjerë.
    balanced = list(HARD_SCENARIOS)

    for scenario in quick:
        if scenario not in balanced:
            balanced.append(scenario)

    # Plotësojmë deri në 12 skenarë unikë.
    while len(balanced) < 12:
        candidate = (
            int(rng.integers(0, 6)),
            int(rng.integers(0, 5)),
        )
        if candidate not in balanced:
            balanced.append(candidate)

    return quick, balanced


def optimize(rounds=1, trials=15, seed=2029):
    RESULTS.mkdir(parents=True, exist_ok=True)

    udp = morphing_rover_UDP()
    rng = np.random.default_rng(seed)

    lb, ub = udp.get_bounds()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    source = CHECKPOINT
    if not source.exists():
        source = RESULTS / "tamro_v4_best.npy"

    if not source.exists():
        source = RESULTS / "tamro_best.npy"

    best = np.load(source).astype(float)
    best = repair(best, lb, ub)
    best_score = float(udp.fitness(best.tolist())[0])

    print("=== TAMRO v4.1 ===", flush=True)
    print(f"Starting fitness: {best_score:.6f}", flush=True)

    np.save(CHECKPOINT, best)

    fieldnames = [
        "round", "trial", "operation",
        "sigma", "quick", "balanced",
        "full", "best_full", "improved"
    ]

    for round_id in range(1, rounds + 1):
        print(f"\nRound {round_id}/{rounds}", flush=True)

        quick_cases, balanced_cases = choose_scenarios(rng)
        candidates = []
        records = {}

        # Secili kandidat është mutacion i konfigurimit më të mirë.
        for trial in range(1, trials + 1):
            sigma = float(rng.choice([
                0.01, 0.03, 0.1, 0.3, 1.0, 3.0
            ]))

            candidate, operation = mutate(
                best, rng, sigma, lb, ub
            )

            quick = evaluate(udp, candidate, quick_cases)

            records[trial] = {
                "round": round_id,
                "trial": trial,
                "operation": operation,
                "sigma": sigma,
                "quick": quick,
                "balanced": "",
                "full": "",
                "best_full": best_score,
                "improved": False
            }

            if np.isfinite(quick):
                candidates.append(
                    (quick, trial, operation, candidate)
                )

            print(
                f"Quick {trial}/{trials}: "
                f"{operation} = {quick:.6f}",
                flush=True
            )

        # Ruajmë diversitet: një kandidat për çdo lloj mutacioni.
        candidates.sort(key=lambda x: x[0])
        diverse = []
        used_operations = set()

        for item in candidates:
            if item[2] not in used_operations:
                diverse.append(item)
                used_operations.add(item[2])

        # Plotësojmë edhe me kandidatë të tjerë të mirë.
        selected_ids = {item[1] for item in diverse}
        for item in candidates:
            if len(diverse) >= 7:
                break
            if item[1] not in selected_ids:
                diverse.append(item)
                selected_ids.add(item[1])

        balanced_candidates = []

        for quick, trial, operation, candidate in diverse:
            balanced = evaluate(
                udp, candidate, balanced_cases
            )

            records[trial]["balanced"] = balanced

            if np.isfinite(balanced):
                balanced_candidates.append(
                    (balanced, trial, candidate)
                )

            print(
                f"Balanced {trial}: {balanced:.6f}",
                flush=True
            )

        balanced_candidates.sort(key=lambda x: x[0])

        # Tre kandidatë verifikohen me 30 skenarë.
        for balanced, trial, candidate in balanced_candidates[:3]:
            full = float(udp.fitness(candidate.tolist())[0])
            improved = np.isfinite(full) and full < best_score

            if improved:
                best = candidate.copy()
                best_score = full
                np.save(CHECKPOINT, best)

                print(
                    f"NEW RECORD: {best_score:.6f}",
                    flush=True
                )

            records[trial]["full"] = full
            records[trial]["improved"] = improved

            print(f"Full {trial}: {full:.6f}", flush=True)

        with HISTORY.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if file.tell() == 0:
                writer.writeheader()

            for record in records.values():
                record["best_full"] = best_score
                writer.writerow(record)

        print(
            f"Round {round_id} best: {best_score:.6f}",
            flush=True
        )

    print(f"\nFINAL FITNESS: {best_score:.6f}")
    print(f"Checkpoint saved: {CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2029)

    args = parser.parse_args()

    if args.rounds < 1 or args.trials < 5:
        parser.error("rounds >= 1 and trials >= 5 required")

    optimize(args.rounds, args.trials, args.seed)
