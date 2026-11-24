"""TAMRO v5.3: paired-coordinate trust-region search for ESA SpOC2.

Uses a validated v5 checkpoint. All decisions to accept a new best use
morphing_rover_UDP.fitness (the official 30-scenario evaluator).
Quick/balanced scores are only candidate-screening heuristics.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from morphing_udp import morphing_rover_UDP, Rover, NUM_MODE_PARAMETERS, NUM_BIASES, NUM_WEIGHTS, MASK_CENTRES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"
STATE = RESULTS / "tamro_v5_3_state.npz"
BEST = RESULTS / "tamro_v5_3_best.npy"
HISTORY = RESULTS / "tamro_v5_3_history.csv"
ALL = [(m, s) for m in range(6) for s in range(5)]
MASK_END = NUM_MODE_PARAMETERS
BIAS_END = MASK_END + NUM_BIASES
CONT_END = BIAS_END + NUM_WEIGHTS
OPS = ("BIAS_COORD", "BIAS_PAIR", "BIAS_GROUP", "WEIGHT_COORD", "WEIGHT_BLOCK", "MASK_COORD", "ELITE_DIRECTION")
BASE_RADII = np.array([0.08, 0.06, 0.04, 0.04, 0.025, 0.10, 0.15], dtype=float)
FIELDS = ("round", "trial", "operator", "radius", "quick", "balanced", "full", "accepted", "best_full")


def validate(x, lo, hi):
    if x.shape != lo.shape or x.size != 19133 or not np.isfinite(x).all():
        raise ValueError("Invalid chromosome size or non-finite values")
    if (x < lo).any() or (x > hi).any() or not np.array_equal(x[CONT_END:], np.rint(x[CONT_END:])):
        raise ValueError("Chromosome violates ESA bounds or discrete constraints")


def repair(x, lo, hi):
    x = np.clip(x, lo, hi)
    x[CONT_END:] = np.rint(x[CONT_END:])
    for i in MASK_CENTRES:
        x[i] = max(float(x[i]), max(1e-16, float(lo[i])))
    return x


def assess(udp, x, scenarios, cache):
    missing = [s for s in scenarios if s not in cache]
    if missing:
        rover = Rover(x.tolist())
        for m, s in missing:
            a, b = udp.run_single_scenario(rover, m, s)
            cache[(m, s)] = float((1 + a) * b)
    return float(np.mean([cache[s] for s in scenarios]))


def load_state(seed, lo, hi):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as z:
            best = z["best"].astype(float)
            state = json.loads(str(z["metadata"].item()))
        rng.bit_generator.state = state["rng_state"]
        print(f"Resuming TAMRO v5.3 after round {state['rounds_done']}", flush=True)
    else:
        paths = [RESULTS / "tamro_v5_best.npy", ROOT / "solution/checkpoints/tamro_v5_best.npy"]
        source = next((p for p in paths if p.exists()), None)
        if source is None:
            raise FileNotFoundError("Cannot find v5 checkpoint; do not start from a random chromosome")
        best = np.load(source).astype(float)
        state = {"rounds_done": 0, "radii": BASE_RADII.tolist(), "attempts": [0]*len(OPS),
                 "wins": [0]*len(OPS), "stagnation": 0, "seed": seed}
        print(f"Starting from {source}", flush=True)
    validate(best, lo, hi)
    return best, state, rng


def load_elites(best, lo, hi):
    pool = [best.copy()]
    file = RESULTS / "tamro_v5_state.npz"
    if file.exists():
        with np.load(file, allow_pickle=False) as z:
            elites = z["elites"] if "elites" in z else []
        for x in elites:
            try:
                x = np.asarray(x, dtype=float)
                validate(x, lo, hi)
            except (ValueError, TypeError):
                continue
            if not any(np.array_equal(x, y) for y in pool):
                pool.append(x.copy())
    return pool


def generate_delta(op, best, elites, radius, rng):
    delta = np.zeros_like(best)
    if op == 0:
        i = int(rng.integers(MASK_END, BIAS_END))
        delta[i] = radius
    elif op == 1:
        ids = rng.choice(np.arange(MASK_END, BIAS_END), size=2, replace=False)
        delta[ids] = radius * rng.choice([-1., 1.], size=2)
    elif op == 2:
        count = int(rng.choice([4, 8, 12]))
        ids = rng.choice(np.arange(MASK_END, BIAS_END), count, replace=False)
        delta[ids] = rng.normal(0, radius, count)
    elif op == 3:
        i = int(rng.integers(BIAS_END, CONT_END))
        delta[i] = radius
    elif op == 4:
        count = int(rng.choice([4, 8, 16, 32]))
        start = int(rng.integers(BIAS_END, CONT_END - count + 1))
        delta[start:start+count] = rng.normal(0, radius, count)
    elif op == 5:
        i = int(rng.integers(MASK_END))
        delta[i] = radius
    elif op == 6:
        donor_list = [e for e in elites if not np.array_equal(e, best)]
        if donor_list:
            donor = donor_list[int(rng.integers(len(donor_list)))]
            # Bias-only interpolated elite direction, capped to trust region.
            different = np.flatnonzero(np.abs(donor[MASK_END:BIAS_END]-best[MASK_END:BIAS_END]) > 1e-12) + MASK_END
            if len(different):
                n = min(len(different), 8)
                ids = rng.choice(different, size=n, replace=False)
                delta[ids] = np.clip(0.2*(donor[ids]-best[ids]), -radius, radius)
        if not np.any(delta):
            ids = rng.choice(np.arange(MASK_END, BIAS_END), 4, replace=False)
            delta[ids] = rng.normal(0, radius, 4)
    return delta


def choose_cases(rng, profile):
    # Same screening cases for all candidates in a round.
    quick = []
    for m in range(6):
        hardest = max(range(5), key=lambda s: profile[(m, s)])
        other = int(rng.choice([s for s in range(5) if s != hardest]))
        quick.extend([(m, hardest), (m, other)])
    balanced = list(quick)
    # Cover at least three scenarios per map, then add difficult and random cases.
    for m in range(6):
        left = [(m, s) for s in range(5) if (m, s) not in balanced]
        balanced.append(max(left, key=lambda s: profile[s]))
    while len(balanced) < 22:
        remaining = [s for s in ALL if s not in balanced]
        # Mixed hard/rotating sample to avoid overfitting to the current worst few.
        if rng.random() < 0.6:
            chosen = max(remaining, key=lambda s: profile[s])
        else:
            chosen = remaining[int(rng.integers(len(remaining)))]
        balanced.append(chosen)
    return quick, balanced


def diversified_shortlist(items, n, rank_key, explore=1):
    """Prioritize screening results while retaining diverse and exploratory candidates."""
    ranked = sorted(items, key=rank_key)
    chosen = ranked[:max(1, n//2)]
    ids = {x["trial"] for x in chosen}
    ops = {x["op"] for x in chosen}
    for x in ranked:
        if len(chosen) >= n-explore:
            break
        if x["op"] not in ops and x["trial"] not in ids:
            chosen.append(x)
            ids.add(x["trial"])
            ops.add(x["op"])
    for x in ranked:
        if len(chosen) >= n-explore:
            break
        if x["trial"] not in ids:
            chosen.append(x)
            ids.add(x["trial"])
    # One exploratory candidate from outside the ranked shortlist, when possible.
    if len(chosen) < n:
        for x in reversed(ranked):
            if x["trial"] not in ids:
                chosen.append(x)
                break
    return chosen


def write_state(best, state):
    tmp = STATE.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, best=best, metadata=json.dumps(state))
    os.replace(tmp, STATE)
    tmp = BEST.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f, best)
    os.replace(tmp, BEST)


def log_rows(rows):
    first = not HISTORY.exists() or HISTORY.stat().st_size == 0
    with HISTORY.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if first:
            writer.writeheader()
        writer.writerows(rows)


def optimize(rounds, trials, seed, target, finalists):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(a, dtype=float) for a in udp.get_bounds())
    if len(lo) != CONT_END+7:
        raise ValueError("Unexpected ESA decision-variable count")
    best, state, rng = load_state(seed, lo, hi)
    elites = load_elites(best, lo, hi)
    incumbent = float(udp.fitness(best.tolist())[0])
    profile = {}
    measured = assess(udp, best, ALL, profile)
    if not np.isfinite(incumbent) or abs(incumbent-measured) > 1e-4:
        raise ValueError("Official fitness disagrees with scenario evaluation")
    print(f"Official starting fitness: {incumbent:.9f}; elite pool={len(elites)}", flush=True)
    if incumbent < target:
        print(f"Already under target {target}", flush=True)
        return
    for _ in range(rounds):
        rid = state["rounds_done"] + 1
        quick, balanced = choose_cases(rng, profile)
        qbase = float(np.mean([profile[s] for s in quick]))
        bbase = float(np.mean([profile[s] for s in balanced]))
        print(f"\nROUND {rid} | best={incumbent:.9f} | quick baseline={qbase:.5f} | balanced baseline={bbase:.5f}", flush=True)
        candidates = []
        records = []
        used = {best.tobytes()}
        pairs = 0
        max_pairs = trials * 6
        while len(records) < trials and pairs < max_pairs:
            if pairs < len(OPS):
                op = pairs
            else:
                # Soft operator preference: all operators remain possible.
                probs = np.array([0.3+(state["wins"][i]+1)/(state["attempts"][i]+2) for i in range(len(OPS))])
                probs /= probs.sum()
                op = int(rng.choice(len(OPS), p=probs))
            pairs += 1
            radius = float(state["radii"][op]) * float(rng.choice([0.25, 0.5, 1, 2, 4]))
            delta = generate_delta(op, best, elites, radius, rng)
            for sign in (1, -1):
                if len(records) >= trials:
                    break
                cand = repair(best + sign*delta, lo, hi)
                key = cand.tobytes()
                if key in used:
                    continue
                used.add(key)
                cache = {}
                qscore = assess(udp, cand, quick, cache)
                trial = len(records) + 1
                row = dict(round=rid, trial=trial, operator=OPS[op], radius=radius,
                           quick=qscore, balanced="", full="", accepted=False, best_full=incumbent)
                records.append(row)
                if np.isfinite(qscore):
                    candidates.append(dict(trial=trial, op=op, x=cand, cache=cache, quick=qscore))
                print(f"Quick {trial}/{trials} {OPS[op]}: {qscore:.6f} (delta {qscore-qbase:+.6f})", flush=True)
        if not candidates:
            raise RuntimeError("No finite candidate scores")
        stage = diversified_shortlist(candidates, min(14, len(candidates)), lambda c: c["quick"])
        for c in stage:
            score = assess(udp, c["x"], balanced, c["cache"])
            c["balanced"] = score
            records[c["trial"]-1]["balanced"] = score
            print(f"Balanced {c['trial']} {OPS[c['op']]}: {score:.6f} (delta {score-bbase:+.6f})", flush=True)
        valid = [c for c in stage if np.isfinite(c["balanced"])]
        if not valid:
            raise RuntimeError("No finite balanced candidates")
        stage = diversified_shortlist(valid, min(finalists, len(valid)), lambda c: c["balanced"])
        improved_round = False
        for c in stage:
            # Official ESA full evaluation; only this value can become a record.
            official = float(udp.fitness(c["x"].tolist())[0])
            op = c["op"]
            state["attempts"][op] += 1
            improved = bool(np.isfinite(official) and official < incumbent-1e-8)
            if improved:
                old = best.copy()
                best = c["x"].copy()
                incumbent = official
                state["wins"][op] += 1
                state["radii"][op] = min(2.0, state["radii"][op]*1.15)
                improved_round = True
                if not any(np.array_equal(old, e) for e in elites):
                    elites.append(old)
                if len(elites) > 10:
                    elites = [best.copy()] + elites[-9:]
                # Persist a new record immediately (even if a later trial fails).
                state["rng_state"] = rng.bit_generator.state
                write_state(best, state)
                print(f"*** NEW VERIFIED RECORD: {incumbent:.9f} ***", flush=True)
            else:
                state["radii"][op] = max(1e-5, state["radii"][op]*0.85)
            row = records[c["trial"]-1]
            row["full"] = official
            row["accepted"] = improved
            print(f"Full {c['trial']} {OPS[op]}: {official:.9f}", flush=True)
            if incumbent < target:
                break
        if improved_round:
            profile = {}
            measured = assess(udp, best, ALL, profile)
            if abs(measured-incumbent) > 1e-4:
                raise ValueError("Record failed independent scenario check")
            state["stagnation"] = 0
        else:
            state["stagnation"] += 1
            if state["stagnation"] % 3 == 0:
                # If stuck, widen trust regions modestly without altering best.
                state["radii"] = [min(2.0, r*1.5) for r in state["radii"]]
        state["rounds_done"] = rid
        state["rng_state"] = rng.bit_generator.state
        for r in records:
            r["best_full"] = incumbent
        write_state(best, state)
        log_rows(records)
        print(f"ROUND {rid} DONE | best={incumbent:.9f}", flush=True)
        if incumbent < target:
            print(f"TARGET REACHED: fitness below {target}", flush=True)
            break
    print(f"\nFINAL OFFICIAL FITNESS: {incumbent:.9f}", flush=True)
    print(f"Checkpoint: {BEST}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=2036)
    p.add_argument("--target", type=float, default=4.0)
    p.add_argument("--finalists", type=int, default=6)
    args = p.parse_args()
    if args.rounds < 1 or args.trials < 14 or args.finalists < 1 or args.finalists > args.trials or not np.isfinite(args.target):
        p.error("Require rounds>=1, trials>=14, 1<=finalists<=trials and finite target")
    optimize(args.rounds, args.trials, args.seed, args.target, args.finalists)
