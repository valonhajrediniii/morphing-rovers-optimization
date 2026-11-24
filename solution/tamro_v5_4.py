"""TAMRO v5.4: small-step paired search with sequential scenario screening.

Original local search for ESA Morphing Rovers. Accepts improvements solely using
morphing_rover_UDP.fitness on all 30 scenarios. No third-party chromosomes.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from morphing_udp import (morphing_rover_UDP, Rover, NUM_MODE_PARAMETERS,
                         NUM_BIASES, NUM_WEIGHTS, MASK_CENTRES)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'solution' / 'results'
STATE = RESULTS / 'tamro_v5_4_state.npz'
BEST = RESULTS / 'tamro_v5_4_best.npy'
HISTORY = RESULTS / 'tamro_v5_4_history.csv'
ALL_CASES = [(m, s) for m in range(6) for s in range(5)]
MASK_END = NUM_MODE_PARAMETERS
BIAS_END = MASK_END + NUM_BIASES
CONT_END = BIAS_END + NUM_WEIGHTS
OPS = ('BIAS_COORD', 'BIAS_GROUP', 'WEIGHT_COORD', 'WEIGHT_BLOCK',
       'MASK_COORD', 'MASK_PATCH', 'ELITE_DIRECTION')
INITIAL_STEP = np.array([0.012, 0.018, 0.008, 0.016, 0.020, 0.030, 0.045])
COLUMNS = ('round', 'trial', 'operator', 'step', 'quick', 'balanced',
           'official_full', 'accepted', 'best_full')


def validate(x, lo, hi):
    if x.shape != lo.shape or x.size != 19133 or not np.isfinite(x).all():
        raise ValueError('Invalid ESA chromosome or non-finite values')
    if (x < lo).any() or (x > hi).any():
        raise ValueError('ESA parameter bounds violated')
    if not np.array_equal(x[CONT_END:], np.rint(x[CONT_END:])):
        raise ValueError('Last 7 architectural parameters must be integers')


def repair(x, lo, hi):
    x = np.clip(x, lo, hi)
    x[CONT_END:] = np.rint(x[CONT_END:])
    for j in MASK_CENTRES:
        x[j] = max(float(x[j]), max(1e-16, float(lo[j])))
    return x


def scenarios(udp, x, chosen, cache):
    missing = [case for case in chosen if case not in cache]
    if missing:
        rover = Rover(x.tolist())
        for m, s in missing:
            a, b = udp.run_single_scenario(rover, m, s)
            cache[(m, s)] = float((1 + a) * b)
    return float(np.mean([cache[case] for case in chosen]))


def choose_scenarios(rng, score_by_case):
    """Same cases for all candidates in a round, with 6-map coverage."""
    quick = []
    balanced = []
    for m in range(6):
        ranked = sorted([(m,s) for s in range(5)],
                        key=lambda case: score_by_case[case], reverse=True)
        # hardest plus two rotating cases from every map
        rest = rng.choice(len(ranked)-1, size=2, replace=False)
        cases = [ranked[0], ranked[1+int(rest[0])], ranked[1+int(rest[1])]]
        quick.extend(cases)
        balanced.extend(cases)
        missing = [case for case in ranked if case not in cases]
        balanced.append(missing[int(rng.integers(len(missing)))])
    return quick, balanced  # 18/30, 24/30


def directions(op, parent, elites, step, rng):
    delta = np.zeros_like(parent)
    if op == 0:
        delta[int(rng.integers(MASK_END, BIAS_END))] = step
    elif op == 1:
        n = int(rng.choice([2,4,8]))
        ids = rng.choice(np.arange(MASK_END, BIAS_END), n, replace=False)
        delta[ids] = rng.normal(0, step, n)
    elif op == 2:
        delta[int(rng.integers(BIAS_END, CONT_END))] = step
    elif op == 3:
        n = int(rng.choice([4,8,16]))
        start = int(rng.integers(BIAS_END, CONT_END - n + 1))
        delta[start:start+n] = rng.normal(0, step, n)
    elif op == 4:
        ids = [i for i in range(MASK_END) if i not in MASK_CENTRES]
        delta[int(rng.choice(ids))] = step
    elif op == 5:
        mode = int(rng.integers(4))
        row, col = rng.integers(0, 9, size=2)
        patch = rng.normal(0, step, (3,3))
        start = mode*121
        mask = delta[start:start+121].reshape(11,11)
        mask[row:row+3, col:col+3] = patch
    else:
        donors = [e for e in elites if not np.array_equal(e, parent)]
        if donors:
            donor = donors[int(rng.integers(len(donors)))]
            diff = donor[MASK_END:BIAS_END] - parent[MASK_END:BIAS_END]
            nonzero = np.flatnonzero(np.abs(diff) > 1e-12) + MASK_END
            if len(nonzero):
                ids = rng.choice(nonzero, min(8, len(nonzero)), replace=False)
                delta[ids] = np.clip(donor[ids]-parent[ids], -step, step)
        if not np.any(delta):
            ids = rng.choice(np.arange(MASK_END, BIAS_END), 4, replace=False)
            delta[ids] = rng.normal(0, step, 4)
    return delta


def archive_elites(best, lo, hi):
    pool = [best.copy()]
    state = RESULTS / 'tamro_v5_state.npz'
    if state.exists():
        with np.load(state, allow_pickle=False) as z:
            members = z['elites'] if 'elites' in z else []
        for p in members:
            p = np.asarray(p, dtype=float)
            try:
                validate(p, lo, hi)
            except ValueError:
                continue
            if not any(np.array_equal(p,q) for q in pool):
                pool.append(p.copy())
    return pool


def save(best, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = STATE.with_suffix('.tmp')
    with p.open('wb') as f:
        np.savez_compressed(f, best=best, metadata=json.dumps(state))
    os.replace(p, STATE)
    p = BEST.with_suffix('.tmp')
    with p.open('wb') as f:
        np.save(f, best)
    os.replace(p, BEST)


def append_log(rows):
    first = not HISTORY.exists() or HISTORY.stat().st_size == 0
    with HISTORY.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if first:
            w.writeheader()
        w.writerows(rows)


def select(items, n, key):
    ranked = sorted(items, key=key)
    first = ranked[:min(2,n)]
    used = {p['trial'] for p in first}
    seen = {p['op'] for p in first}
    for p in ranked:
        if len(first) >= n:
            break
        if p['op'] not in seen and p['trial'] not in used:
            first.append(p)
            seen.add(p['op'])
            used.add(p['trial'])
    for p in ranked:
        if len(first) >= n:
            break
        if p['trial'] not in used:
            first.append(p)
            used.add(p['trial'])
    return first


def optimize(rounds, trials, seed, target, finalists):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(b, dtype=float) for b in udp.get_bounds())
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as z:
            best = z['best'].astype(float)
            state = json.loads(str(z['metadata'].item()))
        rng.bit_generator.state = state['rng_state']
        print(f"Resuming v5.4 after {state['rounds_done']} rounds", flush=True)
    else:
        sources = [RESULTS/'tamro_v5_best.npy',
                   ROOT/'solution'/'checkpoints'/'tamro_v5_best.npy']
        source = next((p for p in sources if p.exists()), None)
        if source is None:
            raise FileNotFoundError('Missing v5 best solution checkpoint')
        best = np.load(source).astype(float)
        state = {'rounds_done':0, 'steps':INITIAL_STEP.tolist(),
                 'attempts':[0]*len(OPS), 'successes':[0]*len(OPS),
                 'stagnation':0, 'seed':seed}
        print(f'Starting from {source}', flush=True)
    validate(best, lo, hi)
    best_full = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(best_full):
        raise ValueError('Non-finite starting fitness')
    print(f'Official starting fitness: {best_full:.9f}', flush=True)
    elites = archive_elites(best, lo, hi)
    profile = {}
    prof_fitness = scenarios(udp, best, ALL_CASES, profile)
    if abs(prof_fitness-best_full) > 1e-4:
        raise RuntimeError('Per-scenario scores disagree with official fitness')

    for _ in range(rounds):
        if best_full < target:
            print(f'TARGET ACHIEVED: < {target}', flush=True)
            break
        rid = state['rounds_done'] + 1
        quick, balanced = choose_scenarios(rng, profile)
        qbase = scenarios(udp, best, quick, profile)
        bbase = scenarios(udp, best, balanced, profile)
        print(f'\nROUND {rid} | best={best_full:.9f} | quick baseline={qbase:.5f} | balanced baseline={bbase:.5f}', flush=True)
        candidates, records = [], []
        seen = {best.tobytes()}
        # try positive and negative direction for the identical move
        pairs = 0
        while len(records) < trials and pairs < trials*8:
            if pairs < len(OPS):
                op = pairs
            else:
                wins = np.array(state['successes'], dtype=float)
                attempts = np.array(state['attempts'], dtype=float)
                probs = 0.5 + (wins+1)/(attempts+2)
                probs /= probs.sum()
                op = int(rng.choice(len(OPS), p=probs))
            pairs += 1
            step = state['steps'][op]*float(rng.choice([0.1, 0.25, 0.5, 1, 2]))
            delta = directions(op, best, elites, step, rng)
            for sign in (1, -1):
                if len(records) >= trials:
                    break
                x = repair(best + sign*delta, lo, hi)
                key = x.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                cache = {}
                quick_score = scenarios(udp, x, quick, cache)
                trial = len(records)+1
                records.append(dict(round=rid, trial=trial, operator=OPS[op],
                                    step=step, quick=quick_score, balanced='',
                                    official_full='', accepted=False, best_full=best_full))
                if np.isfinite(quick_score):
                    candidates.append(dict(trial=trial, op=op, x=x, cache=cache, quick=quick_score))
                print(f'Quick {trial}/{trials} {OPS[op]}: {quick_score:.6f} (delta {quick_score-qbase:+.6f})',flush=True)
        if not candidates:
            raise RuntimeError('No viable candidates')
        shortlist = select(candidates, min(14, len(candidates)), lambda c:c['quick'])
        for c in shortlist:
            bscore = scenarios(udp, c['x'], balanced, c['cache'])
            c['balanced'] = bscore
            records[c['trial']-1]['balanced'] = bscore
            print(f"Balanced {c['trial']} {OPS[c['op']]}: {bscore:.6f} (delta {bscore-bbase:+.6f})", flush=True)
        finalists_list = select([x for x in shortlist if np.isfinite(x['balanced'])],
                                min(finalists, len(shortlist)), lambda c:c['balanced'])
        won = False
        for c in finalists_list:
            official = float(udp.fitness(c['x'].tolist())[0])
            op = c['op']
            state['attempts'][op] += 1
            better = bool(np.isfinite(official) and official < best_full-1e-8)
            records[c['trial']-1]['official_full'] = official
            records[c['trial']-1]['accepted'] = better
            if better:
                best, best_full = c['x'].copy(), official
                state['successes'][op] += 1
                state['steps'][op] = min(0.5, state['steps'][op]*1.10)
                won = True
                state['rng_state'] = rng.bit_generator.state
                save(best, state)  # retain best even if interrupted
                print(f'*** NEW VERIFIED RECORD: {best_full:.9f} ***', flush=True)
            print(f"Full {c['trial']} {OPS[op]}: {official:.9f}", flush=True)
            if best_full < target:
                break
        if won:
            profile = {}
            score = scenarios(udp, best, ALL_CASES, profile)
            if abs(score-best_full) > 1e-4:
                raise RuntimeError('New best differs from scenario average')
            state['stagnation'] = 0
            elites.insert(0,best.copy())
            elites = elites[:10]
        else:
            state['stagnation'] += 1
            # Shrink all search regions on repeated failures. Never expand on failure.
            state['steps'] = [max(1e-6, v*0.70) for v in state['steps']]
        state['rounds_done'] = rid
        state['rng_state'] = rng.bit_generator.state
        for row in records:
            row['best_full'] = best_full
        save(best, state)
        append_log(records)
        print(f'ROUND {rid} DONE | best={best_full:.9f} | steps={np.round(state["steps"],6)}', flush=True)
    print(f'\nFINAL OFFICIAL FITNESS: {best_full:.9f}', flush=True)
    print(f'Checkpoint: {BEST}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--trials', type=int, default=24)
    p.add_argument('--seed', type=int, default=2037)
    p.add_argument('--target', type=float, default=4.0)
    p.add_argument('--finalists', type=int, default=8)
    args = p.parse_args()
    if args.rounds < 1 or args.trials < 14 or not (1 <= args.finalists <= args.trials) or not np.isfinite(args.target):
        p.error('rounds >= 1; trials >= 14; 1 <= finalists <= trials; finite target')
    optimize(args.rounds, args.trials, args.seed, args.target, args.finalists)
