"""TAMRO v5.8: local mask sensitivity search around index 138.

Every new record is validated on all 30 ESA scenarios using udp.fitness.
No source code or external benchmark solution is copied.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from morphing_udp import Rover, morphing_rover_UDP

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "solution" / "results"
BEST = RESULTS / "tamro_v5_8_best.npy"
STATE = RESULTS / "tamro_v5_8_state.json"
HISTORY = RESULTS / "tamro_v5_8_history.csv"
FOCUS = 138
ALL_CASES = [(m, s) for m in range(6) for s in range(5)]
QUICK = [(2, 3), (2, 4), (3, 1), (1, 1), (0, 1), (4, 0), (5, 1)]
COLUMNS = ["round", "trial", "kind", "indices", "step", "quick", "balanced", "full", "accepted", "best_full"]


def mask_neighbors(index):
    base = (index // 121) * 121
    position = index % 121
    r, c = divmod(position, 11)
    return [base + rr * 11 + cc
            for rr in range(max(0, r-1), min(11, r+2))
            for cc in range(max(0, c-1), min(11, c+2))
            if base + rr*11 + cc != index]


def partial(udp, chromosome, cases, cache):
    missing = [case for case in cases if case not in cache]
    if missing:
        rover = Rover(chromosome.tolist())
        for m, s in missing:
            a, b = udp.run_single_scenario(rover, m, s)
            cache[(m, s)] = float((1.0 + a) * b)
    return float(np.mean([cache[case] for case in cases]))


def atomic_best(x):
    temp = BEST.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.save(handle, x)
    os.replace(temp, BEST)


def atomic_state(state):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    os.replace(temp, STATE)


def log_rows(rows):
    had_file = HISTORY.exists()
    with HISTORY.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not had_file:
            writer.writeheader()
        writer.writerows(rows)


def initial_best():
    for path in (BEST,
                 RESULTS / 'tamro_v5_6_best.npy',
                 ROOT / 'solution' / 'checkpoints' / 'tamro_v5_6_best.npy'):
        if path.exists():
            return np.load(path).astype(float), path
    raise FileNotFoundError('Missing v5.6 checkpoint (results/checkpoints)')


def propose(best, lo, hi, trial, rng, radius, neighbors):
    x = best.copy()
    if trial <= 12:
        amplitude = (1e-6, 5e-6, 2e-5, 8e-5, 3e-4, 1e-3)[(trial-1)//2] * radius
        sign = 1 if trial % 2 else -1
        indices = [FOCUS]
        steps = [sign * amplitude]
        kind = 'FOCUS_LINE'
    elif trial % 4 == 0:
        j = int(rng.choice(neighbors))
        indices = [FOCUS, j]
        amplitude = float(rng.choice([1e-5, 1e-4, 5e-4, 2e-3])) * radius
        steps = [float(rng.choice([-1, 1])) * amplitude,
                 float(rng.choice([-1, 1])) * amplitude]
        kind = 'FOCUS_NEIGHBOR'
    else:
        j = int(rng.choice(neighbors))
        indices = [j]
        amplitude = float(rng.choice([1e-6, 1e-5, 1e-4, 1e-3, 5e-3])) * radius
        steps = [float(rng.choice([-1, 1])) * amplitude]
        kind = 'NEIGHBOR_COORD'
    for index, step in zip(indices, steps):
        x[index] = float(np.clip(x[index] + step, lo[index], hi[index]))
    return x, kind, indices, steps


def optimize(rounds, trials, seed, target):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = [np.asarray(v, dtype=float) for v in udp.get_bounds()]
    best, source = initial_best()
    if len(best) != 19133 or not np.isfinite(best).all():
        raise ValueError('Invalid chromosome')
    if (best < lo).any() or (best > hi).any() or not np.array_equal(best[-7:], np.rint(best[-7:])):
        raise ValueError('Invalid chromosome bounds or discrete parameters')

    if STATE.exists():
        state = json.loads(STATE.read_text(encoding='utf-8'))
        rng = np.random.default_rng()
        rng.bit_generator.state = state['rng_state']
        print(f'Resuming v5.8 after round {state["rounds_done"]}', flush=True)
    else:
        rng = np.random.default_rng(seed)
        state = {'rounds_done': 0, 'radius': 1.0, 'stagnation': 0, 'seed': seed}
        print(f'Starting from: {source}', flush=True)

    best_full = float(udp.fitness(best.tolist())[0])
    print(f'Official starting fitness: {best_full:.9f}', flush=True)
    neighbors = mask_neighbors(FOCUS)
    print('Focus mask coordinate:', FOCUS, '| neighbor indices:', neighbors, flush=True)
    if best_full < target:
        print('Target already achieved.', flush=True)
        return
    atomic_best(best)

    for _ in range(rounds):
        rid = state['rounds_done'] + 1
        balanced = list(QUICK)
        while len(balanced) < 15:
            case = ALL_CASES[int(rng.integers(0, len(ALL_CASES)))]
            if case not in balanced:
                balanced.append(case)
        baseline_cache = {}
        quick_base = partial(udp, best, QUICK, baseline_cache)
        balanced_base = partial(udp, best, balanced, baseline_cache)
        print(f'\nROUND {rid} | best={best_full:.9f} | quick baseline={quick_base:.6f} | balanced baseline={balanced_base:.6f}', flush=True)
        candidates = []
        rows = {}
        seen = {best.tobytes()}
        for trial in range(1, trials+1):
            x, kind, indices, steps = propose(best, lo, hi, trial, rng, state['radius'], neighbors)
            if x.tobytes() in seen:
                print(f'Quick {trial}/{trials} {kind}: duplicate; skipped', flush=True)
                continue
            seen.add(x.tobytes())
            cache = {}
            q = partial(udp, x, QUICK, cache)
            row = {'round': rid, 'trial': trial, 'kind': kind,
                   'indices': ','.join(map(str, indices)),
                   'step': ','.join(f'{v:+.9g}' for v in steps),
                   'quick': q, 'balanced': '', 'full': '',
                   'accepted': False, 'best_full': best_full}
            rows[trial] = row
            if np.isfinite(q):
                candidates.append((q, trial, x, cache))
            print(f'Quick {trial}/{trials} {kind}: {q:.6f} ({q-quick_base:+.6f})', flush=True)

        # Include candidates with best quick means and a few focused on the toughest cases.
        candidates.sort(key=lambda item: item[0])
        shortlist = candidates[:min(9, len(candidates))]
        selected = {c[1] for c in shortlist}
        for scenario in [(2, 3), (3, 1), (2, 4)]:
            relevant = sorted(candidates, key=lambda item: item[3].get(scenario, float('inf')))
            for c in relevant[:2]:
                if c[1] not in selected:
                    shortlist.append(c)
                    selected.add(c[1])
                    break

        evaluated = []
        for q, trial, x, cache in shortlist:
            b = partial(udp, x, balanced, cache)
            rows[trial]['balanced'] = b
            if np.isfinite(b):
                evaluated.append((b, trial, x, cache))
            print(f'Balanced {trial}: {b:.6f} ({b-balanced_base:+.6f})', flush=True)
        evaluated.sort(key=lambda item: item[0])
        finalists = evaluated[:min(5, len(evaluated))]
        selected = {c[1] for c in finalists}
        for scenario in [(2, 3), (3, 1)]:
            for c in sorted(evaluated, key=lambda item: item[3].get(scenario, float('inf'))):
                if c[1] not in selected:
                    finalists.append(c)
                    selected.add(c[1])
                    break

        improved = False
        for _, trial, x, _ in finalists:
            full = float(udp.fitness(x.tolist())[0])
            accepted = bool(np.isfinite(full) and full < best_full - 1e-8)
            rows[trial]['full'] = full
            rows[trial]['accepted'] = accepted
            if accepted:
                best = x.copy()
                best_full = full
                improved = True
                atomic_best(best)
                print(f'*** NEW VERIFIED RECORD: {best_full:.9f} ***', flush=True)
            print(f'Full30 {trial}: {full:.9f}', flush=True)
            if best_full < target:
                break

        if improved:
            state['stagnation'] = 0
            state['radius'] = min(3.0, state['radius'] * 1.2)
        else:
            state['stagnation'] += 1
            state['radius'] = max(0.1, state['radius'] * 0.7)
        state['rounds_done'] = rid
        state['rng_state'] = rng.bit_generator.state
        for row in rows.values():
            row['best_full'] = best_full
        atomic_state(state)
        log_rows(list(rows.values()))
        print(f'ROUND {rid} DONE | best={best_full:.9f} | radius={state["radius"]:.5f}', flush=True)
        if best_full < target:
            print(f'TARGET ACHIEVED: {best_full:.9f} < {target}', flush=True)
            break
    print(f'\nFINAL OFFICIAL FITNESS: {best_full:.9f}', flush=True)
    print(f'Checkpoint: {BEST}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--trials', type=int, default=24)
    p.add_argument('--seed', type=int, default=2041)
    p.add_argument('--target', type=float, default=3.9)
    a = p.parse_args()
    if a.rounds < 1 or a.trials < 12:
        p.error('rounds >= 1 and trials >= 12')
    optimize(a.rounds, a.trials, a.seed, a.target)
