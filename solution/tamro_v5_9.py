"""TAMRO v5.9: scenario-balanced island search for ESA SpOC-2 Morphing Rovers.

Uses only official full fitness for deciding improvements.  Quick/balanced
scores are candidate-selection proxies, not claims of official fitness.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from morphing_udp import Rover, morphing_rover_UDP

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'solution' / 'results'
CHECKPOINT = RESULTS / 'tamro_v5_9_state.npz'
BEST_NPY = RESULTS / 'tamro_v5_9_best.npy'
HISTORY = RESULTS / 'tamro_v5_9_history.csv'
CASES = [(m, s) for m in range(6) for s in range(5)]
CENTER = (60, 181, 302, 423)
MASK_END = 484
BIAS_END = 630
CONT_END = 19126
OPS = ('MASK_COORD', 'MASK_PATCH', 'MASK_MODE', 'BIAS_GROUP',
       'WEIGHT_BLOCK', 'WEIGHT_SPARSE', 'ELITE_DIRECTION', 'FUSION')
INITIAL_STEPS = [0.001, 0.022, 0.018, 0.065, 0.060, 0.045, 0.22, 0.045]
LOG_FIELDS = ('round', 'trial', 'operator', 'scale', 'quick', 'balanced',
              'full', 'improved', 'round_best')


def valid(x, lower, upper):
    return (x.shape == lower.shape and x.size == 19133 and
            np.isfinite(x).all() and (x >= lower).all() and
            (x <= upper).all() and
            np.array_equal(x[CONT_END:], np.rint(x[CONT_END:])))


def repair(x, lower, upper):
    y = np.clip(x, lower, upper)
    y[CONT_END:] = np.rint(y[CONT_END:])
    for i in CENTER:
        y[i] = max(float(y[i]), 1e-16)
    return y


def run_cases(udp, chromosome, subset, cache=None):
    cache = {} if cache is None else cache
    pending = [case for case in subset if case not in cache]
    if pending:
        rover = Rover(chromosome.tolist())
        for m, s in pending:
            value = udp.run_single_scenario(rover, m, s)
            cache[(m, s)] = float((1.0 + value[0]) * value[1])
    return float(np.mean([cache[case] for case in subset])), cache


def load_seed():
    choices = [
        RESULTS/'tamro_v5_6_best.npy',
        ROOT/'solution'/'checkpoints'/'tamro_v5_6_best.npy',
        RESULTS/'tamro_v5_8_best.npy',
        RESULTS/'tamro_v5_7_best.npy',
    ]
    for path in choices:
        if path.exists():
            return np.load(path).astype(np.float64), path
    raise FileNotFoundError('Cannot find verified TAMRO v5.6 checkpoint')


def load_elites(best):
    pool = [best.copy()]
    for version in ('v5_5', 'v5', 'v4_4'):
        for folder in (RESULTS, ROOT/'solution'/'checkpoints'):
            path = folder / f'tamro_{version}_best.npy'
            if path.exists():
                x = np.load(path).astype(np.float64)
                if x.size == best.size and not any(np.array_equal(x, v) for v in pool):
                    pool.append(x)
                break
    return pool[:6]


def make_direction(x, elites, op, rng, step):
    """Return a perturbation (same step is used for +/- paired search)."""
    delta = np.zeros_like(x)
    if op == 0:  # a coordinate in one of the 4 morphology masks
        idx = int(rng.integers(0, MASK_END))
        delta[idx] = step
    elif op == 1:  # local 3 x 3 terrain patch
        mode = int(rng.integers(0, 4))
        r, c = [int(v) for v in rng.integers(0, 9, size=2)]
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                delta[mode*121 + rr*11 + cc] = float(rng.normal(0, step))
    elif op == 2:  # sparse mode mask
        mode = int(rng.integers(0, 4))
        ids = mode*121 + rng.choice(121, 18, replace=False)
        delta[ids] = rng.normal(0, step, len(ids))
    elif op == 3:  # biases; use 1 / 2 / 4 / 8 nearby coordinates
        size = int(rng.choice([1, 2, 4, 8]))
        start = int(rng.integers(MASK_END, BIAS_END-size+1))
        delta[start:start+size] = rng.normal(0, step, size)
    elif op == 4:  # neural contiguous weight block
        size = int(rng.choice([16, 32, 64, 128]))
        start = int(rng.integers(BIAS_END, CONT_END-size+1))
        delta[start:start+size] = rng.normal(0, step, size)
    elif op == 5:  # neural sparse
        size = int(rng.choice([8, 16, 32]))
        ids = rng.choice(np.arange(BIAS_END, CONT_END), size, replace=False)
        delta[ids] = rng.normal(0, step, size)
    elif op == 6 and len(elites) > 1:
        donor = elites[int(rng.integers(1, len(elites)))]
        # select a coherent portion, not a blind 19k-dimensional crossover
        group = int(rng.integers(0, 3))
        lo, hi = [(0, MASK_END), (MASK_END, BIAS_END), (BIAS_END, CONT_END)][group]
        ids = rng.choice(np.arange(lo, hi), size=min(48, hi-lo), replace=False)
        delta[ids] = step * (donor[ids] - x[ids])
    else:  # fusion: two sparse changes in masks and biases
        a = int(rng.integers(0, MASK_END))
        b = int(rng.integers(MASK_END, BIAS_END))
        delta[a] = step * float(rng.choice([-1, 1]))
        delta[b] = step * float(rng.choice([-1, 1]))
    return delta


def choose_cases(rng, baseline):
    # Six hardest + one independent choice per map -> avoid single-terrain overfitting.
    hard = sorted(CASES, key=lambda c: baseline[c], reverse=True)
    quick = list(hard[:6])
    for m in range(6):
        options = [(m, s) for s in range(5) if (m, s) not in quick]
        if options:
            case = options[int(rng.integers(0, len(options)))]
            quick.append(case)
    quick = list(dict.fromkeys(quick))
    balanced = quick.copy()
    for case in hard[6:12]:
        if case not in balanced:
            balanced.append(case)
    remaining = [c for c in CASES if c not in balanced]
    rng.shuffle(remaining)
    balanced.extend(remaining[:max(0, 22-len(balanced))])
    return quick, balanced


def select_diverse(items, limit, key):
    # Reserve positions for alternative operators, and keep some pure-score leaders.
    items = sorted(items, key=key)
    selected = items[:max(2, limit//3)]
    seen_ops = {item['op'] for item in selected}
    seen_trials = {item['trial'] for item in selected}
    for item in items:
        if len(selected) >= limit:
            break
        if item['op'] not in seen_ops and item['trial'] not in seen_trials:
            selected.append(item)
            seen_ops.add(item['op'])
            seen_trials.add(item['trial'])
    for item in items:
        if len(selected) >= limit:
            break
        if item['trial'] not in seen_trials:
            selected.append(item)
            seen_trials.add(item['trial'])
    return selected


def save_state(best, elites, state):
    temp = CHECKPOINT.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, best=best, elites=np.vstack(elites),
                            state=np.array(json.dumps(state)))
    os.replace(temp, CHECKPOINT)
    temp = BEST_NPY.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, best)
    os.replace(temp, BEST_NPY)


def append_history(rows):
    fresh = not HISTORY.exists()
    with HISTORY.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def optimize(rounds, trials, seed, target, finalists):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lower, upper = [np.asarray(v, dtype=np.float64) for v in udp.get_bounds()]
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as data:
            best = data['best'].astype(float)
            elites = [row.copy() for row in data['elites']]
            state = json.loads(str(data['state'].item()))
        rng = np.random.default_rng()
        rng.bit_generator.state = state['rng_state']
        print(f"Resuming TAMRO v5.9 after round {state['rounds_done']}", flush=True)
    else:
        best, source = load_seed()
        elites = load_elites(best)
        rng = np.random.default_rng(seed)
        state = dict(rounds_done=0, seed=seed, wins=[0]*len(OPS),
                     attempts=[0]*len(OPS), steps=INITIAL_STEPS.copy(), stagnation=0)
        print(f'Starting from {source}', flush=True)
    if not valid(best, lower, upper):
        raise ValueError('Starting best violates official ESA bounds')
    elites = [x for x in elites if valid(x, lower, upper)]
    official_best = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(official_best):
        raise ValueError('Nonfinite initial fitness')
    print(f'Official starting fitness: {official_best:.9f}', flush=True)
    # Verify baseline per-scenario agreement with official objective.
    profile, baseline = run_cases(udp, best, CASES)
    if abs(profile - official_best) > 1e-4:
        raise ValueError('Per-scenario scores do not match official fitness')
    for _ in range(rounds):
        round_no = state['rounds_done'] + 1
        quick_cases, balanced_cases = choose_cases(rng, baseline)
        base_quick = float(np.mean([baseline[k] for k in quick_cases]))
        base_balanced = float(np.mean([baseline[k] for k in balanced_cases]))
        print(f'\nROUND {round_no} | full={official_best:.9f} | quick={base_quick:.5f} | balanced={base_balanced:.5f}', flush=True)
        for scenario in sorted(CASES, key=lambda k: baseline[k], reverse=True)[:3]:
            print(f'  hard M{scenario[0]+1}S{scenario[1]+1}: {baseline[scenario]:.4f}', flush=True)
        wins = np.asarray(state['wins'], dtype=float)
        attempts = np.asarray(state['attempts'], dtype=float)
        probs = 0.35+(wins+1)/(attempts+2)
        probs /= probs.sum()
        tested = set()
        candidates = []
        history = []
        for pair in range((trials+1)//2):
            op = pair if pair < len(OPS) else int(rng.choice(len(OPS), p=probs))
            factor = float(rng.choice([0.1, 0.35, 1.0, 3.0, 10.0]))
            step = state['steps'][op]*factor
            delta = make_direction(best, elites, op, rng, step)
            for sign in (1, -1):
                trial = len(history)+1
                if trial > trials:
                    break
                candidate = repair(best+sign*delta, lower, upper)
                if np.array_equal(candidate, best) or candidate.tobytes() in tested:
                    # Still count it to avoid silent duplicate full-score work.
                    history.append(dict(round=round_no, trial=trial, operator=OPS[op], scale=step,
                                        quick='', balanced='', full='', improved=False, round_best=official_best))
                    continue
                tested.add(candidate.tobytes())
                q, cache = run_cases(udp, candidate, quick_cases)
                history.append(dict(round=round_no, trial=trial, operator=OPS[op], scale=step,
                                    quick=q, balanced='', full='', improved=False, round_best=official_best))
                if np.isfinite(q):
                    candidates.append(dict(trial=trial, op=op, x=candidate, q=q, cache=cache))
                print(f'Quick {trial}/{trials} {OPS[op]}: {q:.6f} ({q-base_quick:+.6f})', flush=True)
        shortlist = select_diverse(candidates, min(12, len(candidates)), key=lambda c: c['q'])
        balanced = []
        for c in shortlist:
            b, cache = run_cases(udp, c['x'], balanced_cases, c['cache'])
            history[c['trial']-1]['balanced'] = b
            if np.isfinite(b):
                balanced.append({**c, 'b': b, 'cache': cache})
            print(f"Balanced {c['trial']} {OPS[c['op']]}: {b:.6f} ({b-base_balanced:+.6f})", flush=True)
        pick = select_diverse(balanced, min(finalists, len(balanced)), key=lambda c: c['b'])
        improved_round = False
        for c in pick:
            score = float(udp.fitness(c['x'].tolist())[0])
            op = c['op']
            state['attempts'][op] += 1
            improved = bool(np.isfinite(score) and score < official_best-1e-8)
            if improved:
                official_best = score
                best = c['x'].copy()
                state['wins'][op] += 1
                state['steps'][op] = min(1.0, state['steps'][op]*1.08)
                improved_round = True
                elites.insert(0, best.copy())
                elites = elites[:6]
                # Save the new verified chromosome immediately, not just at round end.
                state['rng_state'] = rng.bit_generator.state
                save_state(best, elites, state)
                print(f'*** NEW VERIFIED RECORD: {score:.9f} ***', flush=True)
            history[c['trial']-1]['full'] = score
            history[c['trial']-1]['improved'] = improved
            print(f"Full {c['trial']} {OPS[op]}: {score:.9f}", flush=True)
            if official_best < target:
                break
        if improved_round:
            profile, baseline = run_cases(udp, best, CASES)
            if abs(profile - official_best) > 1e-4:
                raise RuntimeError('Official and per-scenario fitness disagree')
            state['stagnation'] = 0
        else:
            state['stagnation'] += 1
            if state['stagnation'] % 3 == 0:
                # avoid shrinking to an unproductive microscopic step forever
                state['steps'] = [min(0.5, v*1.6) for v in state['steps']]
        state['rounds_done'] = round_no
        state['rng_state'] = rng.bit_generator.state
        for record in history:
            record['round_best'] = official_best
        save_state(best, elites, state)
        append_history(history)
        print(f'ROUND {round_no} DONE | best={official_best:.9f}', flush=True)
        if official_best < target:
            print('TARGET REACHED!', flush=True)
            break
    print(f'\nFINAL VERIFIED FITNESS: {official_best:.9f}', flush=True)
    print(f'Checkpoint: {BEST_NPY}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--trials', type=int, default=24)
    p.add_argument('--seed', type=int, default=2042)
    p.add_argument('--target', type=float, default=3.9)
    p.add_argument('--finalists', type=int, default=7)
    a = p.parse_args()
    if a.rounds < 1 or a.trials < 16 or a.finalists < 1:
        p.error('rounds>=1, trials>=16, finalists>=1')
    optimize(a.rounds, a.trials, a.seed, a.target, a.finalists)
