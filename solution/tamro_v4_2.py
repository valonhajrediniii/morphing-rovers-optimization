import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import (
    Rover, morphing_rover_UDP, MASK_CENTRES,
    NUM_MODE_PARAMETERS, NUM_BIASES, NUM_WEIGHTS
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"

BEST_FILE = RESULTS / "tamro_v4_2_best.npy"
STATE_FILE = RESULTS / "tamro_v4_2_state.json"
LOG_FILE = RESULTS / "tamro_v4_2_history.csv"

MASKS = NUM_MODE_PARAMETERS
BIASES = NUM_BIASES
CONT = MASKS + BIASES + NUM_WEIGHTS

OPERATIONS = [
    "TERRAIN_PATCH",
    "NEURAL_SPARSE",
    "NEURAL_BLOCK",
    "OUTPUT_CONTROL",
    "ARCHITECTURE",
]

HARD = [
    (3, 4), (2, 4), (3, 3),
    (2, 1), (1, 4)
]

FIELDS = [
    "round", "trial", "operation", "sigma",
    "quick", "balanced", "full",
    "best_full", "improved"
]


def repair(x, lower, upper):
    x = np.clip(x, lower, upper)
    x[CONT:] = np.rint(x[CONT:])

    for index in MASK_CENTRES:
        x[index] = max(float(x[index]), 1e-16)

    return x


def mutate(parent, operation, sigma, rng, lower, upper):
    x = parent.copy()

    if operation == 0:
        mode = int(rng.integers(0, 4))
        row, col = rng.integers(0, 9, size=2)

        mask = x[
            mode * 121:(mode + 1) * 121
        ].reshape(11, 11)

        mask[row:row+3, col:col+3] += rng.normal(
            0, sigma, (3, 3)
        )

    elif operation == 1:
        indices = rng.choice(
            np.arange(MASKS + BIASES, CONT),
            64, replace=False
        )
        x[indices] += rng.normal(0, sigma, 64)

    elif operation == 2:
        start = int(rng.integers(
            MASKS + BIASES, CONT - 127
        ))
        x[start:start+128] += rng.normal(
            0, sigma, 128
        )

    elif operation == 3:
        indices = rng.choice(
            np.arange(CONT - 80, CONT),
            16, replace=False
        )
        x[indices] += rng.normal(0, sigma, 16)

        x[MASKS+BIASES-2:MASKS+BIASES] += (
            rng.normal(0, sigma, 2)
        )

    else:
        index = int(rng.integers(CONT, CONT + 7))

        choices = [
            k for k in range(
                int(lower[index]),
                int(upper[index]) + 1
            )
            if k != int(x[index])
        ]

        if choices:
            x[index] = rng.choice(choices)

    return repair(x, lower, upper)


def select_scenarios(rng):
    quick = [
        (m, int(rng.integers(0, 5)))
        for m in range(6)
    ]

    balanced = list(dict.fromkeys(quick + HARD))

    while len(balanced) < 12:
        candidate = (
            int(rng.integers(0, 6)),
            int(rng.integers(0, 5))
        )

        if candidate not in balanced:
            balanced.append(candidate)

    return quick, balanced


def partial(udp, chromosome, scenarios, cache):
    rover = Rover(chromosome.tolist())

    for scenario in scenarios:
        if scenario not in cache:
            a, b = udp.run_single_scenario(
                rover, scenario[0], scenario[1]
            )
            cache[scenario] = float((1 + a) * b)

    return float(np.mean([
        cache[s] for s in scenarios
    ]))


def atomic_numpy(path, array):
    temp = path.with_suffix(".tmp")

    with temp.open("wb") as handle:
        np.save(handle, array)

    os.replace(temp, path)


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    os.replace(temp, path)


def append_log(records):
    exists = LOG_FILE.exists()

    with LOG_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        if not exists:
            writer.writeheader()

        writer.writerows(records)


def optimize(rounds, trials, seed):
    RESULTS.mkdir(parents=True, exist_ok=True)

    udp = morphing_rover_UDP()

    lower, upper = [
        np.asarray(v, dtype=float)
        for v in udp.get_bounds()
    ]

    if STATE_FILE.exists() and BEST_FILE.exists():
        state = json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )

        best = np.load(BEST_FILE).astype(float)

        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]

        print(
            f"Resuming after {state['rounds_done']} rounds.",
            flush=True
        )

    else:
        source = RESULTS / "tamro_v4_1_best.npy"

        if not source.exists():
            source = (
                ROOT / "solution" / "checkpoints"
                / "tamro_v4_1_best.npy"
            )

        if not source.exists():
            raise FileNotFoundError(
                "Missing TAMRO v4.1 checkpoint"
            )

        best = np.load(source).astype(float)
        rng = np.random.default_rng(seed)

        state = {
            "rounds_done": 0,
            "attempts": [0] * 5,
            "successes": [0] * 5,
            "scales": [1.0] * 5,
            "seed": seed
        }

    if len(best) != len(lower):
        raise ValueError("Invalid chromosome length")

    if not np.all(np.isfinite(best)):
        raise ValueError("Invalid chromosome values")

    if np.any(best < lower) or np.any(best > upper):
        raise ValueError("ESA bounds violated")

    if not np.all(
        best[CONT:] == np.rint(best[CONT:])
    ):
        raise ValueError("Invalid hyperparameters")

    best_full = float(udp.fitness(best.tolist())[0])

    print("=== TAMRO v4.2 ===", flush=True)
    print(
        f"Starting full fitness: {best_full:.6f}",
        flush=True
    )

    for _ in range(rounds):
        round_id = state["rounds_done"] + 1

        quick_cases, balanced_cases = (
            select_scenarios(rng)
        )

        base_quick = partial(
            udp, best, quick_cases, {}
        )

        base_balanced = partial(
            udp, best, balanced_cases, {}
        )

        print(f"\n=== ROUND {round_id} ===", flush=True)
        print(
            f"Baseline quick: {base_quick:.6f}",
            flush=True
        )
        print(
            f"Baseline balanced: {base_balanced:.6f}",
            flush=True
        )

        # Probabilitete adaptive sipas sukseseve.
        probabilities = np.array([
            0.3 + (
                state["successes"][i] + 1
            ) / (
                state["attempts"][i] + 2
            )
            for i in range(5)
        ])

        probabilities /= probabilities.sum()

        candidates = []
        records = {}

        for trial in range(1, trials + 1):
            # Provojme secilin operator te pakten nje here.
            if trial <= 5:
                op = trial - 1
            else:
                op = int(rng.choice(
                    5, p=probabilities
                ))

            sigma = float(rng.choice([
                0.03, 0.1, 0.3, 1.0
            ])) * state["scales"][op]

            candidate = mutate(
                best, op, sigma,
                rng, lower, upper
            )

            cache = {}

            quick = partial(
                udp, candidate, quick_cases, cache
            )

            records[trial] = {
                "round": round_id,
                "trial": trial,
                "operation": OPERATIONS[op],
                "sigma": sigma,
                "quick": quick,
                "balanced": "",
                "full": "",
                "best_full": "",
                "improved": False
            }

            if np.isfinite(quick):
                candidates.append((
                    quick, trial, op, candidate, cache
                ))

            print(
                f"Quick {trial}/{trials}: "
                f"{OPERATIONS[op]} = {quick:.6f}",
                flush=True
            )

        candidates.sort(key=lambda x: x[0])

        # Tre kandidatet me te mire ne Quick.
        selected = candidates[:3]

        # Shtojme edhe operatore te ndryshem.
        seen = {c[2] for c in selected}

        for candidate in candidates:
            if len(selected) >= 8:
                break

            if candidate[2] not in seen:
                selected.append(candidate)
                seen.add(candidate[2])

        for candidate in candidates:
            if len(selected) >= 8:
                break

            if candidate[1] not in {
                s[1] for s in selected
            }:
                selected.append(candidate)

        balanced_list = []

        for quick, trial, op, candidate, cache in selected:
            balanced = partial(
                udp, candidate, balanced_cases, cache
            )

            records[trial]["balanced"] = balanced

            if np.isfinite(balanced):
                balanced_list.append((
                    balanced, trial, op, candidate
                ))

            print(
                f"Balanced {trial}: {balanced:.6f}",
                flush=True
            )

        balanced_list.sort(key=lambda x: x[0])

        finalists = balanced_list[:2]

        if finalists:
            winner_ops = {c[2] for c in finalists}

            extra = next(
                (
                    c for c in balanced_list[2:]
                    if c[2] not in winner_ops
                ),
                None
            )

            if extra:
                finalists.append(extra)
            elif len(balanced_list) > 2:
                finalists.append(balanced_list[2])

        # Full Evaluation: 30 skenaret zyrtare.
        for _, trial, op, candidate in finalists:
            score = float(
                udp.fitness(candidate.tolist())[0]
            )

            improved = bool(
                np.isfinite(score)
                and score < best_full
            )

            state["attempts"][op] += 1

            if improved:
                state["successes"][op] += 1
                state["scales"][op] = min(
                    5.0, state["scales"][op] * 1.15
                )

                best = candidate.copy()
                best_full = score

                print(
                    f"NEW VERIFIED RECORD: "
                    f"{best_full:.6f}",
                    flush=True
                )

            else:
                state["scales"][op] = max(
                    0.15, state["scales"][op] * 0.9
                )

            records[trial]["full"] = score
            records[trial]["improved"] = improved

            print(
                f"Full {trial}: {score:.6f}",
                flush=True
            )

        for record in records.values():
            record["best_full"] = best_full

        state["rounds_done"] = round_id
        state["rng_state"] = rng.bit_generator.state

        atomic_numpy(BEST_FILE, best)
        atomic_json(STATE_FILE, state)
        append_log(list(records.values()))

        print(
            f"Round {round_id} best: "
            f"{best_full:.6f}",
            flush=True
        )

    print("\n=== FINAL RESULT ===")
    print(f"Best full fitness: {best_full:.6f}")
    print(f"Checkpoint: {BEST_FILE}")
    print(f"History: {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds", type=int, default=1
    )
    parser.add_argument(
        "--trials", type=int, default=15
    )
    parser.add_argument(
        "--seed", type=int, default=2030
    )

    args = parser.parse_args()

    if args.rounds < 1 or args.trials < 5:
        parser.error(
            "rounds >= 1 and trials >= 5"
        )

    optimize(
        args.rounds,
        args.trials,
        args.seed
    )
