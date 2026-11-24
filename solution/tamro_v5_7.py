"""TAMRO v5.7: winner-direction trust-region search with full-fitness safeguards.

Original extension to the TAMRO research workflow. Requires the ESA SpOC-2
Morphing Rovers simulator and the existing tamro_v4_4.py compatibility helpers.
Only the official 30-scenario fitness may establish a new record.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from morphing_udp import morphing_rover_UDP
from tamro_v4_4 import ALL, MASKS, BIAS_END, CONT, RESULTS, ROOT, evaluate, repair, validate

STATE = RESULTS / "tamro_v5_7_state.npz"
BEST = RESULTS / "tamro_v5_7_best.npy"
HISTORY = RESULTS / "tamro_v5_7_history.csv"
COLUMNS = ["round", "trial", "kind", "indices", "step", "quick", "balanced",
           "full30", "accepted", "best_full"]


def load_saved(paths):
    for path in paths:
        if path.exists():
            return np.load(path).astype(float), path
    raise FileNotFoundError("No checkpoint found among: " + ", ".join(str(x) for x in paths))


def load_state(seed, lo, hi):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as z:
            best = z["best"].astype(float)
            anchor = z["anchor"].astype(float)
            state = json.loads(str(z["metadata"].item()))
        rng.bit_generator.state = state["rng_state"]
        print(f"Resuming after round {state['rounds_done']}", flush=True)
    else:
        best, from_best = load_saved([
            RESULTS / "tamro_v5_6_best.npy",
            ROOT / "solution" / "checkpoints" / "tamro_v5_6_best.npy",
        ])
        anchor, from_anchor = load_saved([
            RESULTS / "tamro_v5_5_best.npy",
            ROOT / "solution" / "checkpoints" / "tamro_v5_5_best.npy",
        ])
        state = {"rounds_done": 0, "stagnation": 0, "radius": 1.0,
                 "rng_state": rng.bit_generator.state, "seed": seed}
        print(f"Starting from: {from_best}", flush=True)
        print(f"Winning-direction anchor: {from_anchor}", flush=True)
    if len(lo) != 19133 or CONT != 19126:
        raise RuntimeError("Unexpected ESA decision-variable layout")
    validate(best, lo, hi)
    validate(anchor, lo, hi)
    return best, anchor, state, rng


def save_state(best, anchor, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, best=best, anchor=anchor,
                            metadata=json.dumps(state))
    os.replace(temp, STATE)
    temp_best = BEST.with_suffix(".tmp")
    with temp_best.open("wb") as f:
        np.save(f, best)
    os.replace(temp_best, BEST)


def append_history(rows):
    exists = HISTORY.exists() and HISTORY.stat().st_size > 0
    with HISTORY.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def pick_scenarios(rng, baseline):
    """Same baseline and candidate subsets for fair incremental comparisons."""
    rank = sorted(ALL, key=lambda key: baseline[key], reverse=True)
    quick = list(rank[:5])
    for m in range(6):
        case = (m, int(rng.integers(0, 5)))
        if case not in quick:
            quick.append(case)
    while len(quick) < 12:
        case = ALL[int(rng.integers(0, 30))]
        if case not in quick:
            quick.append(case)
    balanced = list(quick)
    for case in rank:
        if len(balanced) >= 22:
            break
        if case not in balanced:
            balanced.append(case)
    while len(balanced) < 22:
        case = ALL[int(rng.integers(0, 30))]
        if case not in balanced:
            balanced.append(case)
    return quick, balanced


def choose_index(rng, changed):
    # Prioritize the successful direction but retain broader exploration.
    mode = int(rng.choice(5, p=[0.34, 0.28, 0.16, 0.13, 0.09]))
    if mode == 0 and len(changed):
        return int(rng.choice(changed))
    if mode == 1:
        return int(rng.integers(MASKS, BIAS_END))
    if mode == 2:
        return int(rng.integers(max(BIAS_END, CONT - 128), CONT))
    if mode == 3:
        return int(rng.integers(0, MASKS))
    return int(rng.integers(BIAS_END, CONT))


def merge_unique(*sequences, maximum):
    chosen = []
    seen = set()
    for seq in sequences:
        for cand in seq:
            idx = cand["trial"]
            if idx not in seen and len(chosen) < maximum:
                chosen.append(cand)
                seen.add(idx)
    return chosen


def optimize(rounds, trials, seed, target):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(bounds, dtype=float) for bounds in udp.get_bounds())
    best, anchor, state, rng = load_state(seed, lo, hi)
    official_best = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(official_best):
        raise RuntimeError("Starting official fitness was not finite")
    baseline = {}
    by_case = evaluate(udp, best, ALL, baseline)
    if abs(by_case - official_best) > 1e-4:
        raise RuntimeError("Official fitness and scenario mean differ")
    print(f"Official starting fitness: {official_best:.9f}", flush=True)

    for _ in range(rounds):
        if official_best < target:
            print(f"Target reached: {official_best:.9f} < {target}", flush=True)
            break
        rid = state["rounds_done"] + 1
        quick, balanced = pick_scenarios(rng, baseline)
        bq = float(np.mean([baseline[c] for c in quick]))
        bb = float(np.mean([baseline[c] for c in balanced]))
        print(f"\nROUND {rid} | best={official_best:.9f} | quick={bq:.6f} | balanced={bb:.6f}", flush=True)

        delta = best - anchor
        delta[CONT:] = 0.0
        changed = np.flatnonzero(abs(delta[:CONT]) > 1e-12)
        rows, pool, probes = [], [], []
        seen = {best.tobytes()}
        radius = float(state["radius"])
        def add(kind, indices, step, x):
            if len(rows) >= trials:
                return None
            x = repair(x, lo, hi)
            signature = x.tobytes()
            if signature in seen:
                return None
            seen.add(signature)
            cache = {}
            q = evaluate(udp, x, quick, cache)
            if not np.isfinite(q):
                return None
            n = len(rows) + 1
            rows.append({"round": rid, "trial": n, "kind": kind,
                         "indices": indices, "step": step, "quick": q,
                         "balanced": "", "full30": "", "accepted": False,
                         "best_full": official_best})
            cand = {"trial": n, "kind": kind, "x": x, "cache": cache,
                    "quick": q}
            pool.append(cand)
            print(f"Quick {n}/{trials} {kind}: {q:.6f} ({q-bq:+.6f})", flush=True)
            return cand

        # 1. Profile the proven winning direction, including very small moves.
        if len(changed):
            gains = [.01, .025, .05, .10, .20, -.025, -.10, .40]
            for gain in gains[:min(8, trials // 3)]:
                add("WINNER_LINE", "|".join(map(str, changed.tolist())),
                    gain, best + gain * delta)

        # 2. Probe both signs on exactly the same coordinate.
        reserve = max(3, trials // 6)
        n_pairs = max(0, (trials - len(rows) - reserve) // 2)
        visited = set()
        for _pair in range(n_pairs):
            idx = choose_index(rng, changed)
            for _retry in range(25):
                if idx not in visited:
                    break
                idx = choose_index(rng, changed)
            visited.add(idx)
            step = radius * float(rng.choice([0.002, .005, .015, .04, .10]))
            step *= max(1.0, abs(float(best[idx])))
            plus = best.copy()
            minus = best.copy()
            plus[idx] += step
            minus[idx] -= step
            a = add("COORD_PLUS", str(idx), step, plus)
            b = add("COORD_MINUS", str(idx), -step, minus)
            for candidate, signed_step in ((a, step), (b, -step)):
                if candidate is not None:
                    probes.append((candidate["quick"]-bq, idx, signed_step))

        # 3. Combine individually promising directions with winner momentum.
        probes.sort(key=lambda item: item[0])
        kept = []
        selected_indices = set()
        for improvement, idx, step in probes:
            if idx not in selected_indices:
                kept.append((idx, step))
                selected_indices.add(idx)
        fusion_count = max(0, trials - len(rows))
        for j in range(fusion_count):
            if not kept and not len(changed):
                break
            x = best.copy()
            if kept:
                how_many = min(len(kept), 1 + j % 3)
                scale = [.2, .5, 1.0, -.4][j % 4]
                for idx, step in kept[:how_many]:
                    x[idx] += scale * step
            if len(changed):
                x += [.01, .03, -.01, .08][j % 4] * delta
            add("FUSION", "mixed", j, x)

        # Evaluate candidate quality on the SAME 22 scenarios.
        ranked_quick = sorted(pool, key=lambda c: c["quick"])
        mandatory_line = sorted(
            [c for c in pool if c["kind"] == "WINNER_LINE"],
            key=lambda c: c["trial"]
        )[:3]
        shortlist = merge_unique(mandatory_line, ranked_quick,
                                 maximum=min(14, len(pool)))
        for c in shortlist:
            score = evaluate(udp, c["x"], balanced, c["cache"])
            c["balanced"] = score
            rows[c["trial"]-1]["balanced"] = score
            print(f"Balanced {c['trial']} {c['kind']}: {score:.6f} ({score-bb:+.6f})", flush=True)

        ranked_balanced = sorted(shortlist, key=lambda c: c["balanced"])
        # Near-zero quick changes made v5.6's record: explicitly test the
        # smallest winner-line moves even if their partial rank is mediocre.
        direct = [c for c in mandatory_line if "balanced" in c]
        finalists = merge_unique(direct[:2], ranked_balanced,
                                 maximum=min(8, len(shortlist)))
        improved = False
        for c in finalists:
            full30 = evaluate(udp, c["x"], ALL, c["cache"])
            won = bool(np.isfinite(full30) and full30 < official_best - 1e-8)
            if won:
                official = float(udp.fitness(c["x"].tolist())[0])
                if not np.isfinite(official) or abs(official-full30) > 1e-4:
                    raise RuntimeError("Mismatch in official ESA verification")
                won = bool(official < official_best - 1e-8)
                full30 = official
            row = rows[c["trial"]-1]
            row["full30"] = full30
            row["accepted"] = won
            print(f"Full30 {c['trial']} {c['kind']}: {full30:.9f}", flush=True)
            if won:
                anchor = best.copy()
                best = c["x"].copy()
                official_best = full30
                baseline = dict(c["cache"])
                state["radius"] = min(3.0, radius * 1.12)
                state["stagnation"] = 0
                state["rng_state"] = rng.bit_generator.state
                save_state(best, anchor, state)
                improved = True
                print(f"*** NEW OFFICIAL RECORD: {official_best:.9f} ***", flush=True)
            if official_best < target:
                break
        if not improved:
            state["stagnation"] += 1
            state["radius"] = max(0.08, radius * 0.75)
            if state["stagnation"] % 4 == 0:
                state["radius"] = min(2.0, radius * 1.3)
        state["rounds_done"] = rid
        state["rng_state"] = rng.bit_generator.state
        for row in rows:
            row["best_full"] = official_best
        save_state(best, anchor, state)
        append_history(rows)
        print(f"ROUND {rid} DONE | best={official_best:.9f} | radius={state['radius']:.5f}", flush=True)
    print(f"\nFINAL OFFICIAL FITNESS: {official_best:.9f}", flush=True)
    print(f"Checkpoint: {BEST}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--target", type=float, default=3.9)
    args = parser.parse_args()
    if args.rounds < 1 or args.trials < 16:
        parser.error("Require --rounds >= 1 and --trials >= 16")
    optimize(args.rounds, args.trials, args.seed, args.target)
