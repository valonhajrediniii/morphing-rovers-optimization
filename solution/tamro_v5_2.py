"""TAMRO v5.2: multi-fidelity, paired micro-mutation search.

Optimize ESA Morphing Rovers only by the 30-scenario official objective.
Quick and balanced subsets are screens, never the reported fitness.
This is an independent incremental experiment; it preserves TAMRO v5 files.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from morphing_udp import morphing_rover_UDP
from tamro_v4_4 import (ALL, BIAS_END, CONT, MASKS, RESULTS, ROOT,
                         evaluate, repair, validate)

STATE = RESULTS / "tamro_v5_2_state.npz"
BEST = RESULTS / "tamro_v5_2_best.npy"
HISTORY = RESULTS / "tamro_v5_2_history.csv"
OPS = ("BIAS_COORD", "BIAS_GROUP", "BIAS_ELITE", "WEIGHT_SPARSE",
       "WEIGHT_BLOCK", "MASK_MICRO", "OUTPUT_HEAD")
FIELDS = ("round", "trial", "operator", "sigma", "quick", "balanced",
          "full", "best_full", "improved")


def atomic_save(best, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, best=best, metadata=json.dumps(state))
    os.replace(tmp, STATE)
    tmp = BEST.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f, best)
    os.replace(tmp, BEST)


def append_history(records):
    first = not HISTORY.exists() or HISTORY.stat().st_size == 0
    with HISTORY.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if first:
            writer.writeheader()
        writer.writerows(records)


def load_best(seed, lo, hi):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as data:
            best = data["best"].astype(float)
            state = json.loads(str(data["metadata"].item()))
        rng.bit_generator.state = state["rng_state"]
        print(f"Resumed v5.2 after round {state['rounds_done']}", flush=True)
    else:
        options = [RESULTS / "tamro_v5_best.npy",
                   ROOT / "solution/checkpoints/tamro_v5_best.npy"]
        source = next((p for p in options if p.exists()), None)
        if source is None:
            raise FileNotFoundError("Missing v5 checkpoint: tamro_v5_best.npy")
        best = np.load(source).astype(float)
        state = {"rounds_done": 0, "attempts": [0] * len(OPS),
                 "successes": [0] * len(OPS), "scales": [1.0] * len(OPS),
                 "stagnation": 0, "seed": seed}
        print(f"Starting from {source}", flush=True)
    validate(best, lo, hi)
    return best, state, rng


def load_elites(best, lo, hi):
    elites = [best.copy()]
    old = RESULTS / "tamro_v5_state.npz"
    if old.exists():
        with np.load(old, allow_pickle=False) as data:
            saved = data["elites"].astype(float) if "elites" in data else []
        for x in saved:
            try:
                validate(x, lo, hi)
            except ValueError:
                continue
            if not any(np.array_equal(x, e) for e in elites):
                elites.append(x.copy())
    return elites


def generate_delta(op, x, elites, sigma, rng):
    delta = np.zeros_like(x)
    if op == 0:  # One to two neural biases: coordinate refinement.
        ids = rng.choice(np.arange(MASKS, BIAS_END),
                         size=int(rng.choice([1, 2])), replace=False)
        delta[ids] = sigma * rng.choice([-1.0, 1.0], size=len(ids))
    elif op == 1:  # Sparse group of biases.
        n = int(rng.choice([4, 8, 16]))
        ids = rng.choice(np.arange(MASKS, BIAS_END), n, replace=False)
        delta[ids] = rng.normal(0, sigma, n)
    elif op == 2:  # A small step along an archived elite's bias difference.
        available = [e for e in elites if not np.array_equal(e, x)]
        if available:
            donor = available[int(rng.integers(len(available)))]
            possible = np.where(np.abs(donor[MASKS:BIAS_END] -
                                       x[MASKS:BIAS_END]) > 1e-12)[0] + MASKS
            if len(possible):
                n = min(len(possible), int(rng.choice([4, 12, 32])))
                ids = rng.choice(possible, n, replace=False)
                gain = float(rng.choice([0.1, 0.25, 0.5]))
                delta[ids] = np.clip((donor[ids] - x[ids]) * gain,
                                     -2 * sigma, 2 * sigma)
        if not np.any(delta):
            ids = rng.choice(np.arange(MASKS, BIAS_END), 8, replace=False)
            delta[ids] = rng.normal(0, sigma, len(ids))
    elif op == 3:  # Sparse neural weights.
        n = int(rng.choice([4, 16, 32]))
        ids = rng.choice(np.arange(BIAS_END, CONT), n, replace=False)
        delta[ids] = rng.normal(0, sigma, n)
    elif op == 4:  # A compact contiguous weight block.
        n = int(rng.choice([8, 32, 64]))
        start = int(rng.integers(BIAS_END, CONT - n + 1))
        delta[start:start+n] = rng.normal(0, sigma, n)
    elif op == 5:  # One pixel or 2x2 area in one terrain mask.
        mode = int(rng.integers(4))
        side = int(rng.choice([1, 2]))
        row = int(rng.integers(0, 12-side))
        col = int(rng.integers(0, 12-side))
        patch = np.zeros((11, 11))
        patch[row:row+side, col:col+side] = rng.normal(0, sigma, (side, side))
        delta[mode*121:(mode+1)*121] = patch.ravel()
    elif op == 6:  # Output head and final control biases.
        n = int(rng.choice([4, 12]))
        ids = rng.choice(np.arange(CONT - 64, CONT), n, replace=False)
        delta[ids] = rng.normal(0, sigma, n)
        if rng.random() < 0.5:
            delta[BIAS_END-2:BIAS_END] = rng.normal(0, sigma, 2)
    return delta


def choose_cases(rng, profile):
    # Two cases per map: its current hardest + one other. Same cases for
    # all candidates in this round so subset comparisons are fair.
    quick = []
    for m in range(6):
        hard_s = max(range(5), key=lambda s: profile[(m, s)])
        quick.append((m, hard_s))
        other = [s for s in range(5) if s != hard_s]
        quick.append((m, int(rng.choice(other))))
    # Expand to 22 cases: prioritize hard remaining cases but preserve
    # map coverage and rotate the held-out cases across rounds.
    balanced = list(quick)
    for m in rng.permutation(6):
        left = [(int(m), s) for s in range(5) if (int(m), s) not in balanced]
        pick = max(left, key=lambda c: profile[c])
        balanced.append(pick)
    while len(balanced) < 22:
        remaining = [c for c in ALL if c not in balanced]
        weights = np.array([max(profile[c], 0.01) for c in remaining], float)
        weights /= weights.sum()
        balanced.append(remaining[int(rng.choice(len(remaining), p=weights))])
    return quick, balanced


def shortlist(ranked, n):
    # Keep strongest scores, reserving a few slots for operator diversity.
    out = ranked[:min(n, max(2, n//2))]
    ids = {c["trial"] for c in out}
    used = {c["op"] for c in out}
    for c in ranked:
        if len(out) >= n:
            break
        if c["op"] not in used and c["trial"] not in ids:
            out.append(c)
            ids.add(c["trial"])
            used.add(c["op"])
    for c in ranked:
        if len(out) >= n:
            break
        if c["trial"] not in ids:
            out.append(c)
            ids.add(c["trial"])
    return out


def optimize(rounds, trials, seed, target, finalists):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(v, float) for v in udp.get_bounds())
    if len(lo) != CONT+7:
        raise ValueError("Unexpected ESA chromosome dimensions")
    best, state, rng = load_best(seed, lo, hi)
    elites = load_elites(best, lo, hi)
    best_score = float(udp.fitness(best.tolist())[0])
    profile = {}
    mean = evaluate(udp, best, ALL, profile)
    if not np.isfinite(best_score) or abs(mean-best_score)>1e-4:
        raise ValueError("Official fitness and scenario score do not agree")
    print(f"Official starting fitness: {best_score:.9f}; elites={len(elites)}", flush=True)
    print("Hardest:", ", ".join(f"M{m+1}S{s+1}={profile[(m,s)]:.3f}"
          for m,s in sorted(ALL, key=lambda c:profile[c], reverse=True)[:5]), flush=True)

    for _ in range(rounds):
        if best_score < target:
            print(f"Target < {target} already reached", flush=True)
            break
        rnd = state["rounds_done"] + 1
        quick, balanced = choose_cases(rng, profile)
        qbase = float(np.mean([profile[c] for c in quick]))
        bbase = float(np.mean([profile[c] for c in balanced]))
        print(f"\nROUND {rnd} | official={best_score:.9f} | quick={qbase:.5f} | balanced={bbase:.5f}", flush=True)
        attempts = np.array(state["attempts"], float)
        wins = np.array(state["successes"], float)
        probs = 0.25 + (wins + 1) / (attempts + 2)
        probs /= probs.sum()
        found = {best.tobytes()}
        candidates, records = [], {}
        pairs = 0
        while len(records) < trials and pairs < trials*5:
            op = pairs if pairs < len(OPS) else int(rng.choice(len(OPS), p=probs))
            pairs += 1
            sigma = float(rng.choice([0.003, 0.01, 0.03, 0.10, 0.30])) * float(state["scales"][op])
            if state["stagnation"] >= 4:
                sigma *= 1.75
            delta = generate_delta(op, best, elites, sigma, rng)
            for direction in (1, -1):
                if len(records) >= trials:
                    break
                x = repair(best + direction*delta, lo, hi)
                key = x.tobytes()
                if key in found:
                    continue
                found.add(key)
                trial = len(records) + 1
                cache = {}
                q = evaluate(udp, x, quick, cache)
                records[trial] = dict(round=rnd, trial=trial, operator=OPS[op],
                                      sigma=sigma, quick=q, balanced="", full="",
                                      best_full=best_score, improved=False)
                if np.isfinite(q):
                    candidates.append(dict(trial=trial, op=op, x=x, cache=cache, quick=q))
                print(f"Quick {trial}/{trials} {OPS[op]}: {q:.6f} (delta {q-qbase:+.6f})", flush=True)
        candidates.sort(key=lambda c: c["quick"])
        stage = shortlist(candidates, min(12, len(candidates)))
        for c in stage:
            b = evaluate(udp, c["x"], balanced, c["cache"])
            c["balanced"] = b
            records[c["trial"]]["balanced"] = b
            print(f"Balanced {c['trial']} {OPS[c['op']]}: {b:.6f} (delta {b-bbase:+.6f})", flush=True)
        stage = sorted((c for c in stage if np.isfinite(c["balanced"])),
                       key=lambda c: c["balanced"])
        finals = shortlist(stage, min(finalists, len(stage)))
        improved_round = False
        for c in finals:
            score = evaluate(udp, c["x"], ALL, c["cache"])
            op = c["op"]
            state["attempts"][op] += 1
            improved = bool(np.isfinite(score) and score < best_score - 1e-8)
            if improved:
                official = float(udp.fitness(c["x"].tolist())[0])
                if not np.isfinite(official) or abs(official-score) > 1e-4:
                    raise ValueError("Independent ESA fitness recheck failed")
                score = official
                improved = official < best_score - 1e-8
            records[c["trial"]]["full"] = score
            records[c["trial"]]["improved"] = improved
            if improved:
                old = best.copy()
                best = c["x"].copy()
                best_score = score
                profile = dict(c["cache"])
                state["successes"][op] += 1
                state["scales"][op] = min(5.0, state["scales"][op]*1.10)
                improved_round = True
                if not any(np.array_equal(old, e) for e in elites):
                    elites.append(old)
                if len(elites) > 8:
                    elites = elites[-8:]
                state["rng_state"] = rng.bit_generator.state
                atomic_save(best, state)
                print(f"*** NEW VERIFIED RECORD: {best_score:.9f} ***", flush=True)
            else:
                state["scales"][op] = max(0.12, state["scales"][op]*0.94)
            print(f"Full {c['trial']} {OPS[op]}: {score:.9f}", flush=True)
            if best_score < target:
                break
        state["rounds_done"] = rnd
        state["stagnation"] = 0 if improved_round else state["stagnation"] + 1
        state["rng_state"] = rng.bit_generator.state
        for r in records.values():
            r["best_full"] = best_score
        atomic_save(best, state)
        append_history(list(records.values()))
        print(f"ROUND {rnd} DONE | best={best_score:.9f}", flush=True)
        if best_score < target:
            print(f"TARGET ACHIEVED: fitness < {target}", flush=True)
            break
    print(f"\nFINAL OFFICIAL FITNESS: {best_score:.9f}", flush=True)
    print(f"Checkpoint: {BEST}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=2035)
    p.add_argument("--target", type=float, default=4.0)
    p.add_argument("--finalists", type=int, default=6)
    args = p.parse_args()
    if args.rounds < 1 or args.trials < 14 or args.finalists < 1 or not np.isfinite(args.target):
        p.error("rounds >= 1, trials >= 14, finalists >= 1, target finite")
    optimize(args.rounds, args.trials, args.seed, args.target, args.finalists)
