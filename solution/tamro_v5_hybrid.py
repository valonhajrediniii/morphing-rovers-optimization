"""TAMRO v5: mixed-integer hybrid DE, diagonal ES and elite crossover.

Optimizes an ESA Morphing Rovers chromosome and accepts a new best only
when the original 30-scenario official fitness independently confirms it.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import morphing_rover_UDP, NUM_BIASES
from tamro_v4_4 import (ALL, BIAS_END, CONT, MASKS, RESULTS, ROOT,
                         archive_add, evaluate, propose, repair, sample_cases,
                         validate)

STATE = RESULTS / "tamro_v5_state.npz"
BEST = RESULTS / "tamro_v5_best.npy"
HISTORY = RESULTS / "tamro_v5_history.csv"
OPERATORS = ("V44_LOCAL", "ES_BIASES", "DE_MASKS", "DE_BIASES",
             "DE_WEIGHTS", "ELITE_CROSSOVER")
FIELDS = ("round", "trial", "operator", "quick", "balanced",
          "full", "best_full", "improved")


def save(best, elites, scores, diag, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, best=best, elites=np.stack(elites),
                            scores=np.asarray(scores), diag=diag,
                            metadata=json.dumps(state))
    os.replace(temp, STATE)
    temp = BEST.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.save(f, best)
    os.replace(temp, BEST)


def load(seed, lo, hi):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as data:
            best = data["best"].astype(float)
            elites = [x.copy() for x in data["elites"]]
            scores = data["scores"].astype(float).tolist()
            diag = data["diag"].astype(float)
            state = json.loads(str(data["metadata"].item()))
        rng.bit_generator.state = state["rng_state"]
        print(f"Resuming TAMRO v5 after round {state['rounds_done']}", flush=True)
    else:
        source = ROOT / "solution" / "checkpoints" / "tamro_v4_4_best.npy"
        old = RESULTS / "tamro_v4_4_state.npz"
        if old.exists():
            with np.load(old, allow_pickle=False) as data:
                best = data["best"].astype(float)
                elites = [x.copy() for x in data["elites"]]
                scores = data["elite_scores"].astype(float).tolist()
            print(f"Loaded v4.4 elite pool: {len(elites)}", flush=True)
        elif source.exists():
            best = np.load(source).astype(float)
            elites = [best.copy()]
            scores = []
            print(f"Loaded checkpoint: {source}", flush=True)
        else:
            raise FileNotFoundError("Missing v4.4 checkpoint")
        diag = np.ones(NUM_BIASES, dtype=float)
        state = {"rounds_done": 0, "attempts": [0] * len(OPERATORS),
                 "successes": [0] * len(OPERATORS), "stagnation": 0,
                 "seed": seed}
    validate(best, lo, hi)
    for x in elites:
        validate(x, lo, hi)
    if diag.shape != (NUM_BIASES,):
        raise ValueError("Invalid diagonal ES state")
    return best, elites, scores, diag, state, rng


def propose_hybrid(op, best, elites, diag, rng, lo, hi, stagnation):
    parent = best if (len(elites) == 1 or rng.random() < 0.75) else elites[int(rng.integers(len(elites)))]
    sigma = float(rng.choice([0.015, 0.04, 0.12, 0.35, 1.0]))
    if stagnation >= 5:
        sigma *= 1.8

    if op == 0:
        variants = propose(parent, int(rng.integers(8)), sigma, rng, lo, hi)
        if not variants:
            return None, None
        return variants[int(rng.integers(len(variants)))], None

    x = parent.copy()
    if op == 1:  # diagonal (1+lambda) evolution strategy on neural biases
        indices = rng.choice(np.arange(MASKS, BIAS_END),
                             size=int(rng.choice([12, 32, 64, 100])), replace=False)
        x[indices] += sigma * diag[indices - MASKS] * rng.standard_normal(len(indices))
        return repair(x, lo, hi), indices

    if op in (2, 3, 4):  # differential evolution in restricted parameter blocks
        if len(elites) >= 3:
            a, b = rng.choice(len(elites), size=2, replace=False)
            differential = elites[int(a)] - elites[int(b)]
        else:
            differential = rng.standard_normal(len(best)) * sigma
        gain = float(rng.choice([0.35, 0.6, 0.85, 1.1]))
        if op == 2:
            m = int(rng.integers(4))
            indices = np.arange(m * 121, (m + 1) * 121)
        elif op == 3:
            indices = rng.choice(np.arange(MASKS, BIAS_END),
                                 size=int(rng.choice([12, 32, 64])), replace=False)
        else:
            width = int(rng.choice([64, 128, 256, 512]))
            start = int(rng.integers(BIAS_END, CONT - width + 1))
            indices = np.arange(start, start + width)
        x[indices] += gain * differential[indices]
        # Small random component allows exploration when elites are identical.
        x[indices] += rng.normal(0, sigma * 0.2, len(indices))
        return repair(x, lo, hi), None

    # Genetic segment crossover between independently evaluated elite solutions.
    if len(elites) < 2:
        return propose_hybrid(0, best, elites, diag, rng, lo, hi, stagnation)
    donor = elites[int(rng.integers(len(elites)))].copy()
    mode = int(rng.integers(4))
    if mode == 0:
        k = int(rng.integers(4))
        indices = np.arange(k * 121, (k + 1) * 121)
    elif mode == 1:
        indices = rng.choice(np.arange(MASKS, BIAS_END), size=40, replace=False)
    elif mode == 2:
        width = int(rng.choice([128, 512, 1024]))
        start = int(rng.integers(BIAS_END, CONT - width + 1))
        indices = np.arange(start, start + width)
    else:
        indices = np.arange(CONT, CONT + 7)
    x[indices] = donor[indices]
    if mode != 3:
        x[indices] += rng.normal(0, sigma * 0.1, len(indices))
    return repair(x, lo, hi), None


def select_diverse(ranked, number):
    # Keep best two by score; then add operator diversity for exploration.
    out = list(ranked[:min(2, number)])
    chosen = {x["trial"] for x in out}
    seen = {x["op"] for x in out}
    for item in ranked[2:]:
        if len(out) >= number:
            break
        if item["op"] not in seen:
            out.append(item)
            chosen.add(item["trial"])
            seen.add(item["op"])
    for item in ranked[2:]:
        if len(out) >= number:
            break
        if item["trial"] not in chosen:
            out.append(item)
            chosen.add(item["trial"])
    return out


def append_history(rows):
    exists = HISTORY.exists() and HISTORY.stat().st_size > 0
    with HISTORY.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def optimize(rounds, trials, seed, target, finalists):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(v, dtype=float) for v in udp.get_bounds())
    if len(lo) != CONT + 7:
        raise RuntimeError("Unexpected ESA chromosome length")
    best, elites, scores, diag, state, rng = load(seed, lo, hi)
    best_score = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(best_score):
        raise ValueError("Nonfinite initial fitness")
    if not scores:
        scores = [best_score]
    # Ensure current best is in the elite archive, even after resumption.
    if not any(np.array_equal(best, e) for e in elites):
        archive_add(elites, scores, best, best_score, best_score)
    print(f"Official starting fitness: {best_score:.9f}; elites={len(elites)}", flush=True)
    baseline_cache = {}
    baseline = evaluate(udp, best, ALL, baseline_cache)
    if abs(baseline - best_score) > 1e-4:
        raise RuntimeError("ESA official and scenario-by-scenario scores disagree")

    for _ in range(rounds):
        if best_score < target:
            print(f"Already below target {target}", flush=True)
            break
        round_id = state["rounds_done"] + 1
        quick, balanced = sample_cases(rng, baseline_cache)
        qbase = evaluate(udp, best, quick, baseline_cache)
        bbase = evaluate(udp, best, balanced, baseline_cache)
        attempts = np.array(state["attempts"], dtype=float)
        successes = np.array(state["successes"], dtype=float)
        probabilities = 0.25 + (successes + 1) / (attempts + 2)
        probabilities /= probabilities.sum()
        print(f"\nROUND {round_id} | best={best_score:.6f} | quick={qbase:.4f} | balanced={bbase:.4f}", flush=True)
        seen, candidates, records = {best.tobytes()}, [], {}
        tries = 0
        while len(records) < trials and tries < trials * 12:
            tries += 1
            op = (tries - 1) if tries <= len(OPERATORS) else int(rng.choice(len(OPERATORS), p=probabilities))
            x, bias_indices = propose_hybrid(op, best, elites, diag, rng, lo, hi, state["stagnation"])
            if x is None or x.tobytes() in seen:
                continue
            seen.add(x.tobytes())
            trial = len(records) + 1
            cache = {}
            quick_score = evaluate(udp, x, quick, cache)
            records[trial] = dict(round=round_id, trial=trial, operator=OPERATORS[op],
                                  quick=quick_score, balanced="", full="", best_full=best_score,
                                  improved=False)
            if np.isfinite(quick_score):
                candidates.append(dict(trial=trial, op=op, x=x, cache=cache,
                                       quick=quick_score, bias_indices=bias_indices))
            print(f"Quick {trial}/{trials} {OPERATORS[op]}: {quick_score:.5f} (delta {quick_score-qbase:+.5f})", flush=True)

        candidates.sort(key=lambda item: item["quick"])
        shortlist = select_diverse(candidates, min(12, len(candidates)))
        for item in shortlist:
            score = evaluate(udp, item["x"], balanced, item["cache"])
            item["balanced"] = score
            records[item["trial"]]["balanced"] = score
            print(f"Balanced {item['trial']} {OPERATORS[item['op']]}: {score:.5f} (delta {score-bbase:+.5f})", flush=True)
        shortlist = [s for s in shortlist if np.isfinite(s["balanced"])]
        shortlist.sort(key=lambda item: item["balanced"])
        finalist_items = select_diverse(shortlist, min(finalists, len(shortlist)))
        improved_round = False
        for item in finalist_items:
            score = evaluate(udp, item["x"], ALL, item["cache"])
            op = item["op"]
            state["attempts"][op] += 1
            improvement = bool(np.isfinite(score) and score < best_score - 1e-8)
            if improvement:
                official = float(udp.fitness(item["x"].tolist())[0])
                if not np.isfinite(official) or abs(official - score) > 1e-4:
                    raise RuntimeError("ESA official recheck failed")
                improvement = official < best_score - 1e-8
                score = official
            records[item["trial"]]["full"] = score
            records[item["trial"]]["improved"] = improvement
            if improvement:
                state["successes"][op] += 1
                archive_add(elites, scores, best, best_score, best_score)
                best, best_score = item["x"].copy(), score
                archive_add(elites, scores, best, best_score, best_score)
                baseline_cache = dict(item["cache"])
                if item["bias_indices"] is not None:
                    indices = item["bias_indices"] - MASKS
                    diag[indices] = np.clip(diag[indices] * 1.07, 0.2, 4.0)
                improved_round = True
                state["rng_state"] = rng.bit_generator.state
                save(best, elites, scores, diag, state)
                print(f"*** NEW OFFICIAL RECORD: {best_score:.9f} ***", flush=True)
            elif np.isfinite(score):
                archive_add(elites, scores, item["x"], score, best_score)
                if item["bias_indices"] is not None:
                    indices = item["bias_indices"] - MASKS
                    diag[indices] = np.clip(diag[indices] * 0.995, 0.2, 4.0)
            print(f"Full {item['trial']} {OPERATORS[op]}: {score:.6f}", flush=True)
            if best_score < target:
                break
        state["stagnation"] = 0 if improved_round else state["stagnation"] + 1
        state["rounds_done"] = round_id
        state["rng_state"] = rng.bit_generator.state
        for record in records.values():
            record["best_full"] = best_score
        save(best, elites, scores, diag, state)
        append_history(list(records.values()))
        print(f"ROUND {round_id} DONE | best={best_score:.9f} | elites={len(elites)}", flush=True)
        if best_score < target:
            print(f"TARGET ACHIEVED: < {target}", flush=True)
            break
    print(f"\nFINAL OFFICIAL FITNESS: {best_score:.9f}", flush=True)
    print(f"Checkpoint: {BEST}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--seed", type=int, default=2033)
    p.add_argument("--target", type=float, default=4.0)
    p.add_argument("--finalists", type=int, default=5)
    args = p.parse_args()
    if args.rounds < 1 or args.trials < 12 or args.finalists < 1 or not np.isfinite(args.target):
        p.error("rounds >= 1, trials >= 12, finalists >= 1, target finite")
    optimize(args.rounds, args.trials, args.seed, args.target, args.finalists)
