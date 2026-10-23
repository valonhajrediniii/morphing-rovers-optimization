"""TAMRO v4.4: elite-guided, multi-fidelity mixed-integer optimization.
Independent optimizer for the ESA SpOC2 Morphing Rovers UDP.
Minimizes the official mean fitness across 30 scenarios.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import (
    Rover, morphing_rover_UDP, NUM_MODE_PARAMETERS,
    NUM_BIASES, NUM_WEIGHTS, MASK_CENTRES,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"
BEST_PATH = RESULTS / "tamro_v4_4_best.npy"
STATE_PATH = RESULTS / "tamro_v4_4_state.npz"
LOG_PATH = RESULTS / "tamro_v4_4_history.csv"
ALL = [(m, s) for m in range(6) for s in range(5)]
MASKS = NUM_MODE_PARAMETERS
BIAS_END = MASKS + NUM_BIASES
CONT = BIAS_END + NUM_WEIGHTS
OPS = [
    "ARCH_SINGLE", "ARCH_DOUBLE", "OUTPUT_HEAD", "BIAS_GROUP",
    "NEURAL_BLOCK", "NEURAL_SPARSE", "MASK_TEMPLATE", "MASK_PATCH",
]
FIELDS = [
    "round", "trial", "operator", "sigma", "quick", "balanced",
    "full", "best_full", "improved", "parent",
]


def repair(x, lo, hi):
    x = np.clip(x, lo, hi)
    x[CONT:] = np.rint(x[CONT:])
    for i in MASK_CENTRES:
        x[i] = max(x[i], 1e-16)
    return x


def validate(x, lo, hi):
    if x.shape != lo.shape or not np.isfinite(x).all():
        raise ValueError("Invalid chromosome shape or nonfinite value")
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError("Chromosome violates ESA bounds")
    if not np.array_equal(x[CONT:], np.rint(x[CONT:])):
        raise ValueError("Seven architecture values must be integers")


def evaluate(udp, x, scenarios, cache):
    missing = [case for case in scenarios if case not in cache]
    if missing:
        rover = Rover(x.tolist())
        for m, s in missing:
            a, b = udp.run_single_scenario(rover, m, s)
            cache[(m, s)] = float((1 + a) * b)
    return float(np.mean([cache[c] for c in scenarios]))


def sample_cases(rng, base):
    # One per terrain map; sometimes the hardest one, sometimes random.
    quick = []
    for m in range(6):
        hardest = max(range(5), key=lambda s: base[(m, s)])
        s = hardest if rng.random() < 0.55 else int(rng.integers(5))
        quick.append((m, s))
    # More reliable screening of promising candidates.
    balanced = list(quick)
    hardest_cases = sorted(ALL, key=lambda c: base[c], reverse=True)
    for c in hardest_cases[:5]:
        if c not in balanced:
            balanced.append(c)
    while len(balanced) < 16:
        c = ALL[int(rng.integers(30))]
        if c not in balanced:
            balanced.append(c)
    return quick, balanced


def architecture_variant(parent, rng, lo, hi, number):
    x = parent.copy()
    slots = rng.choice(np.arange(CONT, CONT + 7), number, replace=False)
    for i in slots:
        values = [v for v in range(int(lo[i]), int(hi[i]) + 1) if v != int(x[i])]
        if values:
            x[i] = rng.choice(values)
    return x


def propose(parent, op, sigma, rng, lo, hi):
    """Produce at most two distinct nearby chromosomes around a parent."""
    if op in (0, 1):
        count = 1 if op == 0 else 2
        out = []
        for _ in range(2):
            x = repair(architecture_variant(parent, rng, lo, hi, count), lo, hi)
            if not np.array_equal(x, parent) and not any(np.array_equal(x, y) for y in out):
                out.append(x)
        return out

    delta = np.zeros_like(parent)
    if op == 2:  # Output layer weights and 2 output biases
        indices = rng.choice(np.arange(CONT - 80, CONT), 30, replace=False)
        delta[indices] = rng.normal(0, sigma, len(indices))
        delta[BIAS_END - 2:BIAS_END] = rng.normal(0, sigma, 2)
    elif op == 3:  # Various neural biases, small subset
        indices = rng.choice(np.arange(MASKS, BIAS_END), 12, replace=False)
        delta[indices] = rng.normal(0, sigma, len(indices))
    elif op == 4:  # Contiguous NN-weight block, multi-scale
        width = int(rng.choice([16, 64, 128, 384, 768]))
        start = int(rng.integers(BIAS_END, CONT - width + 1))
        delta[start:start + width] = rng.normal(0, sigma, width)
    elif op == 5:  # Sparse NN parameters
        width = int(rng.choice([16, 64, 128, 384]))
        indices = rng.choice(np.arange(MASKS, CONT), width, replace=False)
        delta[indices] = rng.normal(0, sigma, width)
    elif op == 6:  # Direct morphology optimization toward flat normalized terrain
        modes = rng.choice(4, int(rng.choice([1, 2, 4])), replace=False)
        gain = float(rng.choice([0.05, 0.15, 0.35, 0.65, 1.0]))
        for mode in modes:
            start = int(mode) * 121
            delta[start:start + 121] = gain * (1.0 - parent[start:start + 121])
    else:  # Local terrain mask patch
        mode = int(rng.integers(4))
        row, col = rng.integers(0, 9, size=2)
        patch = np.zeros((11, 11), dtype=float)
        patch[row:row + 3, col:col + 3] = rng.normal(0, sigma, (3, 3))
        delta[mode * 121:(mode + 1) * 121] = patch.ravel()

    variants = [repair(parent + delta, lo, hi), repair(parent - delta, lo, hi)]
    return [v for v in variants if not np.array_equal(v, parent)]


def atomic_npz(best, elites, elite_scores, state):
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, best=best, elites=np.stack(elites),
                            elite_scores=np.array(elite_scores),
                            state=json.dumps(state))
    os.replace(tmp, STATE_PATH)
    tmp = BEST_PATH.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f, best)
    os.replace(tmp, BEST_PATH)


def log_round(rows):
    has_header = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)


def starting_best():
    choices = [
        ROOT / "solution" / "checkpoints" / "tamro_v4_3_best.npy",
        RESULTS / "tamro_v4_3_best.npy",
    ]
    for path in choices:
        if path.exists():
            return np.load(path).astype(float), path
    # v4.3 also saved an NPZ, even if its .npy was never copied.
    old_state = RESULTS / "tamro_v4_3_state.npz"
    if old_state.exists():
        with np.load(old_state, allow_pickle=False) as data:
            return data["chromosome"].astype(float), old_state
    raise FileNotFoundError("Missing TAMRO v4.3 best checkpoint")


def shortlist_diverse(ranked, wanted):
    picked = []
    used = set()
    seen_ops = set()
    for cand in ranked:
        if len(picked) >= wanted:
            break
        if cand["op"] not in seen_ops:
            picked.append(cand)
            used.add(cand["trial"])
            seen_ops.add(cand["op"])
    for cand in ranked:
        if len(picked) >= wanted:
            break
        if cand["trial"] not in used:
            picked.append(cand)
            used.add(cand["trial"])
    return picked


def archive_add(elites, scores, x, score, best_score):
    if score > best_score + 0.7:
        return
    for v in elites:
        if np.array_equal(v, x):
            return
    elites.append(x.copy())
    scores.append(float(score))
    order = np.argsort(scores)[:5]
    elites[:] = [elites[int(i)] for i in order]
    scores[:] = [scores[int(i)] for i in order]


def optimize(rounds, trials, seed, target):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(v, dtype=float) for v in udp.get_bounds())
    if len(lo) != CONT + 7:
        raise RuntimeError("Unexpected ESA chromosome layout")

    if STATE_PATH.exists():
        with np.load(STATE_PATH, allow_pickle=False) as data:
            best = data["best"].astype(float)
            elites = [x.copy() for x in data["elites"]]
            scores = list(map(float, data["elite_scores"]))
            state = json.loads(str(data["state"].item()))
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        print(f"Resumed after round {state['rounds_done']}", flush=True)
    else:
        best, source = starting_best()
        rng = np.random.default_rng(seed)
        elites = [best.copy()]
        scores = []
        state = {
            "rounds_done": 0, "attempts": [0] * len(OPS),
            "successes": [0] * len(OPS), "stagnation": 0, "seed": seed,
        }
        print(f"Starting from {source}", flush=True)

    validate(best, lo, hi)
    best_score = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(best_score):
        raise ValueError("Nonfinite initial official fitness")
    if not scores:
        scores = [best_score]
    print(f"Official starting fitness: {best_score:.9f}", flush=True)

    base_cache = {}
    base_profile = evaluate(udp, best, ALL, base_cache)
    if abs(base_profile - best_score) > 1e-4:
        raise RuntimeError("Scenario scores disagree with ESA fitness")

    for _ in range(rounds):
        r = state["rounds_done"] + 1
        quick, balanced = sample_cases(rng, base_cache)
        qbase = float(np.mean([base_cache[c] for c in quick]))
        bbase = float(np.mean([base_cache[c] for c in balanced]))
        print(f"\nROUND {r}: best={best_score:.6f} | quick baseline={qbase:.4f} | balanced baseline={bbase:.4f}", flush=True)

        attempts = np.array(state["attempts"], dtype=float)
        successes = np.array(state["successes"], dtype=float)
        probs = 0.35 + (successes + 1) / (attempts + 2)
        probs[0:2] *= 1.5  # Discrete architecture achieved last major gain.
        probs /= probs.sum()

        candidates = []
        rows = {}
        fingerprints = {best.tobytes()}
        for pair_id in range((trials + 1) // 2):
            op = pair_id % len(OPS) if pair_id < len(OPS) else int(rng.choice(len(OPS), p=probs))
            sigma = float(rng.choice([0.01, 0.03, 0.1, 0.3, 1.0, 3.0]))
            if state["stagnation"] >= 4:
                sigma *= 2
            parent_id = 0
            if len(elites) > 1 and rng.random() < 0.28:
                parent_id = int(rng.integers(1, len(elites)))
            parent = elites[parent_id]
            variants = propose(parent, op, sigma, rng, lo, hi)
            for x in variants:
                if len(rows) >= trials:
                    break
                key = x.tobytes()
                if key in fingerprints:
                    continue
                fingerprints.add(key)
                trial = len(rows) + 1
                cache = {}
                q = evaluate(udp, x, quick, cache)
                record = dict(round=r, trial=trial, operator=OPS[op], sigma=sigma,
                              quick=q, balanced="", full="", best_full=best_score,
                              improved=False, parent=parent_id)
                rows[trial] = record
                if np.isfinite(q):
                    candidates.append(dict(trial=trial, op=op, x=x, cache=cache, quick=q))
                print(f"Quick {trial}/{trials} {OPS[op]}: {q:.5f} (delta {q-qbase:+.5f})", flush=True)

        candidates.sort(key=lambda c: c["quick"])
        shortlisted = shortlist_diverse(candidates, min(12, len(candidates)))
        for cand in shortlisted:
            b = evaluate(udp, cand["x"], balanced, cand["cache"])
            cand["balanced"] = b
            rows[cand["trial"]]["balanced"] = b
            print(f"Balanced {cand['trial']}: {b:.5f} (delta {b-bbase:+.5f})", flush=True)
        shortlisted = [c for c in shortlisted if np.isfinite(c["balanced"])]
        shortlisted.sort(key=lambda c: c["balanced"])
        finalists = shortlist_diverse(shortlisted, min(5, len(shortlisted)))

        improved_round = False
        for cand in finalists:
            # Evaluate only missing scenarios: no expensive duplicated full simulation.
            full = evaluate(udp, cand["x"], ALL, cand["cache"])
            op = cand["op"]
            state["attempts"][op] += 1
            improved = bool(np.isfinite(full) and full < best_score - 1e-8)
            if improved:
                # Recheck independently with official 30-scenario function.
                official = float(udp.fitness(cand["x"].tolist())[0])
                if not np.isfinite(official) or abs(official - full) > 1e-4:
                    raise RuntimeError("Official ESA verification failed")
                full = official
                improved = official < best_score - 1e-8
            rows[cand["trial"]]["full"] = full
            rows[cand["trial"]]["improved"] = improved
            if improved:
                state["successes"][op] += 1
                archive_add(elites, scores, best, best_score, best_score)
                best = cand["x"].copy()
                best_score = full
                archive_add(elites, scores, best, best_score, best_score)
                base_cache = dict(cand["cache"])
                improved_round = True
                state["stagnation"] = 0
                state["rng_state"] = rng.bit_generator.state
                atomic_npz(best, elites, scores, state)
                print(f"*** NEW VERIFIED RECORD: {best_score:.9f} ***", flush=True)
            elif np.isfinite(full):
                archive_add(elites, scores, cand["x"], full, best_score)
            print(f"Full {cand['trial']} {OPS[op]}: {full:.6f}", flush=True)
            if best_score < target:
                break

        if not improved_round:
            state["stagnation"] += 1
        state["rounds_done"] = r
        state["rng_state"] = rng.bit_generator.state
        for row in rows.values():
            row["best_full"] = best_score
        atomic_npz(best, elites, scores, state)
        log_round(list(rows.values()))
        print(f"ROUND {r} FINISHED | best full fitness = {best_score:.9f} | elites={len(elites)}", flush=True)
        if best_score < target:
            print(f"TARGET HIT: fitness < {target}", flush=True)
            break
    print(f"\nFINAL OFFICIAL VERIFIED BEST: {best_score:.9f}", flush=True)
    print(f"Checkpoint: {BEST_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--target", type=float, default=4.0)
    args = parser.parse_args()
    if args.rounds < 1 or args.trials < 16 or not np.isfinite(args.target):
        parser.error("Need --rounds >= 1, --trials >= 16, finite target")
    optimize(args.rounds, args.trials, args.seed, args.target)
