"""TAMRO 4.3: paired, scenario-aware adaptive optimization."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import Rover, morphing_rover_UDP
from tamro_v4_2 import OPERATIONS, mutate, repair

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"
SNAPSHOT = RESULTS / "tamro_v4_3_state.npz"
LOG = RESULTS / "tamro_v4_3_history.csv"

ALL_SCENARIOS = [
    (m, s) for m in range(6) for s in range(5)
]

FIELDS = [
    "round", "trial", "operator", "sigma",
    "quick", "balanced", "full",
    "best_full", "improved"
]


def evaluate_cases(udp, x, cases, cache=None):
    cache = {} if cache is None else cache
    rover = Rover(x.tolist())

    for m, s in cases:
        if (m, s) not in cache:
            a, b = udp.run_single_scenario(rover, m, s)
            cache[(m, s)] = float((1 + a) * b)

    score = np.mean([cache[key] for key in cases])
    return float(score), cache


def initial_chromosome():
    paths = [
        RESULTS / "tamro_v4_2_best.npy",
        ROOT / "solution" / "checkpoints"
        / "tamro_v4_2_best.npy",
    ]

    for path in paths:
        if path.exists():
            return np.load(path).astype(float), path

    raise FileNotFoundError(
        "Cannot find TAMRO v4.2 checkpoint"
    )


def save_snapshot(best, state):
    temp = SNAPSHOT.with_suffix(".tmp")

    with temp.open("wb") as file:
        np.savez_compressed(
            file,
            chromosome=best,
            metadata=json.dumps(state)
        )

    os.replace(temp, SNAPSHOT)


def append_rows(rows):
    existing = LOG.exists()

    with LOG.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        if not existing:
            writer.writeheader()

        writer.writerows(rows)


def choose_cases(rng, baseline_scores):
    quick = []

    for m in range(6):
        hardest = max(
            range(5),
            key=lambda s: baseline_scores[(m, s)]
        )

        if rng.random() < 0.5:
            s = hardest
        else:
            s = int(rng.integers(0, 5))

        quick.append((m, s))

    balanced = list(quick)

    ranked_hard = sorted(
        ALL_SCENARIOS,
        key=lambda key: baseline_scores[key],
        reverse=True
    )

    for case in ranked_hard[:6]:
        if case not in balanced:
            balanced.append(case)

    while len(balanced) < 15:
        case = ALL_SCENARIOS[
            int(rng.integers(0, 30))
        ]

        if case not in balanced:
            balanced.append(case)

    return quick, balanced


def choose_finalists(ranked, count=4):
    finalists = ranked[:2]

    selected_ids = {
        item[1] for item in finalists
    }

    seen_ops = {
        item[2] for item in finalists
    }

    for item in ranked[2:]:
        if len(finalists) >= count:
            break

        if (
            item[1] not in selected_ids
            and item[2] not in seen_ops
        ):
            finalists.append(item)
            selected_ids.add(item[1])
            seen_ops.add(item[2])

    for item in ranked[2:]:
        if len(finalists) >= count:
            break

        if item[1] not in selected_ids:
            finalists.append(item)
            selected_ids.add(item[1])

    return finalists


def optimize(rounds, trials, seed):
    RESULTS.mkdir(parents=True, exist_ok=True)

    udp = morphing_rover_UDP()

    lower, upper = (
        np.asarray(v, dtype=float)
        for v in udp.get_bounds()
    )

    if SNAPSHOT.exists():
        with np.load(
            SNAPSHOT, allow_pickle=False
        ) as saved:
            best = saved["chromosome"].astype(float)
            state = json.loads(
                str(saved["metadata"].item())
            )

        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]

        print(
            f"Resuming after round "
            f"{state['rounds_done']}",
            flush=True
        )

    else:
        best, source = initial_chromosome()
        rng = np.random.default_rng(seed)

        state = {
            "rounds_done": 0,
            "attempts": [0] * 5,
            "successes": [0] * 5,
            "scales": [1.0] * 5,
            "stagnation": 0,
            "seed": seed
        }

        print(f"Starting from: {source}", flush=True)

    if (
        len(best) != len(lower)
        or not np.isfinite(best).all()
    ):
        raise ValueError("Invalid checkpoint")

    if (
        (best < lower).any()
        or (best > upper).any()
    ):
        raise ValueError("ESA bounds violated")

    if not np.array_equal(
        best[-7:], np.rint(best[-7:])
    ):
        raise ValueError("Invalid architecture")

    best_full = float(
        udp.fitness(best.tolist())[0]
    )

    if not np.isfinite(best_full):
        raise ValueError("Invalid initial fitness")

    print(
        f"Verified starting fitness: "
        f"{best_full:.6f}",
        flush=True
    )

    # Profile the current best over all 30 cases.
    profile, base_cache = evaluate_cases(
        udp, best, ALL_SCENARIOS
    )

    if abs(profile - best_full) > 1e-4:
        raise ValueError(
            "Scenario and official scores disagree"
        )

    for _ in range(rounds):
        round_num = state["rounds_done"] + 1

        quick_cases, balanced_cases = (
            choose_cases(rng, base_cache)
        )

        quick_base, _ = evaluate_cases(
            udp, best, quick_cases, dict(base_cache)
        )

        balanced_base, _ = evaluate_cases(
            udp, best, balanced_cases,
            dict(base_cache)
        )

        attempts = np.asarray(
            state["attempts"], dtype=float
        )

        successes = np.asarray(
            state["successes"], dtype=float
        )

        weights = (
            0.35
            + (successes + 1) / (attempts + 2)
        )

        weights /= weights.sum()

        candidates = []
        rows = {}

        print(
            f"\nROUND {round_num} | "
            f"Quick baseline: {quick_base:.4f} | "
            f"Balanced baseline: {balanced_base:.4f}",
            flush=True
        )

        for pair in range((trials + 1) // 2):

            if pair < 5:
                op = pair
            else:
                op = int(
                    rng.choice(
                        len(OPERATIONS), p=weights
                    )
                )

            scale = float(rng.choice([
                0.01, 0.03, 0.10,
                0.30, 1.0, 3.0
            ]))

            sigma = (
                scale * state["scales"][op]
            )

            if state["stagnation"] >= 3:
                sigma *= 2.0

            positive = mutate(
                best, op, sigma,
                rng, lower, upper
            )

            # Opposite-direction perturbation.
            negative = repair(
                best - (positive - best),
                lower, upper
            )

            for candidate in [
                positive, negative
            ]:
                trial = len(rows) + 1

                if trial > trials:
                    break

                q, cache = evaluate_cases(
                    udp, candidate, quick_cases
                )

                rows[trial] = {
                    "round": round_num,
                    "trial": trial,
                    "operator": OPERATIONS[op],
                    "sigma": sigma,
                    "quick": q,
                    "balanced": "",
                    "full": "",
                    "best_full": best_full,
                    "improved": False
                }

                if np.isfinite(q):
                    candidates.append((
                        q, trial, op,
                        candidate, cache
                    ))

                print(
                    f"Quick {trial}/{trials}: "
                    f"{OPERATIONS[op]} "
                    f"{q:.5f} "
                    f"(delta {q-quick_base:+.5f})",
                    flush=True
                )

        candidates.sort(key=lambda c: c[0])

        shortlist = choose_finalists(
            candidates,
            count=min(8, len(candidates))
        )

        balanced = []

        for q, trial, op, candidate, cache in shortlist:
            score, _ = evaluate_cases(
                udp, candidate,
                balanced_cases, cache
            )

            rows[trial]["balanced"] = score

            if np.isfinite(score):
                balanced.append((
                    score, trial, op, candidate
                ))

            print(
                f"Balanced {trial}: "
                f"{score:.5f} "
                f"(delta {score-balanced_base:+.5f})",
                flush=True
            )

        balanced.sort(key=lambda c: c[0])

        finalists = choose_finalists(
            balanced, count=4
        )

        improved_any = False

        for _, trial, op, candidate in finalists:
            score = float(
                udp.fitness(candidate.tolist())[0]
            )

            improved = bool(
                np.isfinite(score)
                and score < best_full - 1e-8
            )

            state["attempts"][op] += 1

            if improved:
                state["successes"][op] += 1

                state["scales"][op] = min(
                    5.0,
                    state["scales"][op] * 1.12
                )

                best = candidate.copy()
                best_full = score
                improved_any = True

                print(
                    f"NEW VERIFIED RECORD: "
                    f"{best_full:.6f}",
                    flush=True
                )

            else:
                state["scales"][op] = max(
                    0.1,
                    state["scales"][op] * 0.92
                )

            rows[trial]["full"] = score
            rows[trial]["improved"] = improved

            print(
                f"Full {trial}: {score:.6f}",
                flush=True
            )

        if improved_any:
            profile, base_cache = evaluate_cases(
                udp, best, ALL_SCENARIOS
            )

            if abs(profile - best_full) > 1e-4:
                raise ValueError(
                    "Official fitness verification failed"
                )

            state["stagnation"] = 0

        else:
            state["stagnation"] += 1

        state["rounds_done"] = round_num
        state["rng_state"] = (
            rng.bit_generator.state
        )

        for row in rows.values():
            row["best_full"] = best_full

        save_snapshot(best, state)
        append_rows(list(rows.values()))

        print(
            f"Round {round_num} best: "
            f"{best_full:.6f}",
            flush=True
        )

        if best_full < 5.0:
            print(
                "TARGET ACHIEVED: FITNESS BELOW 5.0",
                flush=True
            )
            break

    print("\n=== FINAL RESULT ===")
    print(
        f"Final verified fitness: "
        f"{best_full:.6f}"
    )
    print(f"Snapshot: {SNAPSHOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds", type=int, default=1
    )
    parser.add_argument(
        "--trials", type=int, default=20
    )
    parser.add_argument(
        "--seed", type=int, default=2031
    )

    args = parser.parse_args()

    if args.rounds < 1 or args.trials < 10:
        parser.error(
            "rounds >= 1 and trials >= 10 required"
        )

    optimize(
        args.rounds,
        args.trials,
        args.seed
    )
