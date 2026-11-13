"""TAMRO v5.5: Active-Parameter Paired Search with Multi-fidelity Verification.

Original optimization code; requires the independent ESA Morphing Rovers
simulator and project-local tamro_v4_4 compatibility functions.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import morphing_rover_UDP
from tamro_v4_4 import ALL, BIAS_END, CONT, MASKS, RESULTS, ROOT, evaluate, repair, validate

STATE = RESULTS / 'tamro_v5_5_state.npz'
BEST = RESULTS / 'tamro_v5_5_best.npy'
HISTORY = RESULTS / 'tamro_v5_5_history.csv'
COLUMNS = ('round', 'trial', 'kind', 'index', 'direction', 'step',
           'quick', 'balanced', 'full', 'accepted', 'best_full')
INITIAL_STEPS = {'BIAS': 0.025, 'WEIGHT': 0.025, 'MASK': 0.015}


def atomic_save(best, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, best=best, metadata=json.dumps(state))
    os.replace(temp, STATE)
    temp = BEST.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, best)
    os.replace(temp, BEST)


def write_log(rows):
    exists = HISTORY.exists() and HISTORY.stat().st_size > 0
    with HISTORY.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def starting_checkpoint():
    sources = [
        RESULTS / 'tamro_v5_best.npy',
        ROOT / 'solution' / 'checkpoints' / 'tamro_v5_best.npy',
    ]
    for source in sources:
        if source.exists():
            return np.load(source).astype(float), source
    raise FileNotFoundError('TAMRO v5 best chromosome missing')


def prepare(seed, lo, hi):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as data:
            best = data['best'].astype(float)
            state = json.loads(str(data['metadata'].item()))
        rng.bit_generator.state = state['rng_state']
        print(f"Resuming after round {state['rounds_done']}", flush=True)
    else:
        best, source = starting_checkpoint()
        state = {'rounds_done': 0, 'stagnation': 0, 'steps': dict(INITIAL_STEPS),
                 'rng_state': rng.bit_generator.state, 'seed': seed}
        print(f'Starting from {source}', flush=True)
    validate(best, lo, hi)
    if len(lo) != 19133 or CONT != 19126:
        raise RuntimeError('Unexpected chromosome layout')
    return best, state, rng


def pick_scenarios(rng, baseline):
    # Balanced coverage (one case per map), plus the hardest individual cases.
    quick = list(sorted(ALL, key=lambda s: baseline[s], reverse=True)[:5])
    for m in range(6):
        item = (m, int(rng.integers(5)))
        if item not in quick:
            quick.append(item)
    while len(quick) < 12:
        case = ALL[int(rng.integers(len(ALL)))]
        if case not in quick:
            quick.append(case)
    balanced = list(quick)
    for case in sorted(ALL, key=lambda s: baseline[s], reverse=True):
        if len(balanced) >= 22:
            break
        if case not in balanced:
            balanced.append(case)
    while len(balanced) < 22:
        case = ALL[int(rng.integers(len(ALL)))]
        if case not in balanced:
            balanced.append(case)
    return quick, balanced


def probe_index(rng, counts):
    # At least one sample from every region, otherwise bias-heavy sampling.
    k = int(rng.choice(3, p=np.array([0.50, 0.34, 0.16])))
    if counts['BIAS'] == 0:
        k = 0
    elif counts['WEIGHT'] == 0:
        k = 1
    elif counts['MASK'] == 0:
        k = 2
    if k == 0:
        return 'BIAS', int(rng.integers(MASKS, BIAS_END))
    if k == 1:
        # Mix weights across network and final-output segment.
        if rng.random() < 0.30:
            return 'WEIGHT', int(rng.integers(CONT - 128, CONT))
        return 'WEIGHT', int(rng.integers(BIAS_END, CONT))
    return 'MASK', int(rng.integers(0, MASKS))


def shortlist(candidates, n):
    ranked = sorted(candidates, key=lambda c: c['balanced'])
    chosen = ranked[:min(2, n)]
    seen = {c['kind'] for c in chosen}
    used = {c['trial'] for c in chosen}
    for c in ranked[2:]:
        if len(chosen) >= n:
            break
        if c['kind'] not in seen:
            chosen.append(c)
            used.add(c['trial'])
            seen.add(c['kind'])
    for c in ranked[2:]:
        if len(chosen) >= n:
            break
        if c['trial'] not in used:
            chosen.append(c)
            used.add(c['trial'])
    return chosen


def optimize(rounds, trials, seed, target):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(a, dtype=float) for a in udp.get_bounds())
    best, state, rng = prepare(seed, lo, hi)
    best_score = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(best_score):
        raise RuntimeError('Invalid starting full score')
    print(f'Official starting fitness: {best_score:.9f}', flush=True)

    base = {}
    by_scenario = evaluate(udp, best, ALL, base)
    if abs(by_scenario - best_score) > 1e-4:
        raise RuntimeError('Official score differs from 30-scenario mean')

    for _ in range(rounds):
        if best_score < target:
            print(f'Target already reached (<{target})', flush=True)
            break
        rnd = state['rounds_done'] + 1
        quick, balanced = pick_scenarios(rng, base)
        qbase = float(np.mean([base[s] for s in quick]))
        bbase = float(np.mean([base[s] for s in balanced]))
        print(f'\nROUND {rnd} | best={best_score:.9f} | quick={qbase:.5f} | balanced={bbase:.5f}', flush=True)

        rows, candidates, scouts = [], [], []
        seen = {best.tobytes()}
        counts = {'BIAS': 0, 'WEIGHT': 0, 'MASK': 0}
        # Reserve 4 trials for combinations of sensitively selected directions.
        n_pairs = (trials - 4) // 2
        tried_coords = set()

        def add_candidate(kind, index, direction, step, x):
            x = repair(x, lo, hi)
            fingerprint = x.tobytes()
            if fingerprint in seen:
                return None
            seen.add(fingerprint)
            trial = len(rows) + 1
            cache = {}
            q = evaluate(udp, x, quick, cache)
            if not np.isfinite(q):
                return None
            rec = {'round': rnd, 'trial': trial, 'kind': kind,
                   'index': index, 'direction': direction, 'step': step,
                   'quick': q, 'balanced': '', 'full': '',
                   'accepted': False, 'best_full': best_score}
            rows.append(rec)
            cand = {'trial': trial, 'kind': kind, 'x': x, 'cache': cache, 'quick': q}
            candidates.append(cand)
            print(f'Quick {trial}/{trials}: {kind} {index} {direction:+} = {q:.6f} ({q-qbase:+.6f})', flush=True)
            return cand

        for p in range(n_pairs):
            kind, idx = probe_index(rng, counts)
            for _retry in range(10):
                if idx not in tried_coords:
                    break
                kind, idx = probe_index(rng, counts)
            tried_coords.add(idx)
            counts[kind] += 1
            scale = state['steps'][kind] * float(rng.choice([0.5, 1.0, 2.0, 4.0]))
            scale *= max(1., abs(best[idx]))
            outputs = []
            for sign in (+1, -1):
                x = best.copy()
                x[idx] += sign * scale
                out = add_candidate(kind, idx, sign, scale, x)
                if out is not None:
                    outputs.append((sign, out))
            if outputs:
                winner_sign, winner = min(outputs, key=lambda t: t[1]['quick'])
                scouts.append((winner['quick'] - qbase, kind, idx, winner_sign, scale))

        # Combine coordinates suggested by the probe phase.
        scouts.sort(key=lambda s: s[0])
        for fusion in range(4):
            if len(rows) >= trials or not scouts:
                break
            x = best.copy()
            n = min(2 + fusion, len(scouts))
            chosen = scouts[:n]
            multiplier = [0.25, 0.5, 1.0, 1.5][fusion]
            for _, kind, idx, sign, scale in chosen:
                x[idx] += multiplier * sign * scale
            add_candidate('FUSION', -1, +1, multiplier, x)

        if not candidates:
            print('No nonduplicate candidates this round', flush=True)
            state['stagnation'] += 1
        else:
            # Balanced stage: high ranks plus at least one candidate per region.
            top_q = sorted(candidates, key=lambda c: c['quick'])
            screened = top_q[:max(1, min(8, len(top_q)))]
            seen_types = {c['kind'] for c in screened}
            for c in top_q:
                if len(screened) >= min(12, len(top_q)):
                    break
                if c['kind'] not in seen_types:
                    screened.append(c)
                    seen_types.add(c['kind'])
            selected_trials = {c['trial'] for c in screened}
            for c in top_q:
                if len(screened) >= min(12, len(top_q)):
                    break
                if c['trial'] not in selected_trials:
                    screened.append(c)
                    selected_trials.add(c['trial'])

            for c in screened:
                b = evaluate(udp, c['x'], balanced, c['cache'])
                c['balanced'] = b
                rows[c['trial'] - 1]['balanced'] = b
                print(f"Balanced {c['trial']} {c['kind']}: {b:.6f} ({b-bbase:+.6f})", flush=True)

            eligible = [c for c in screened if np.isfinite(c['balanced'])]
            finalists = shortlist(eligible, min(6, len(eligible)))
            won = False
            for c in finalists:
                # Reuse per-candidate per-scenario cache; run only missing cases.
                full = evaluate(udp, c['x'], ALL, c['cache'])
                if not np.isfinite(full):
                    continue
                improved = full < best_score - 1e-8
                if improved:
                    # Independent official full evaluation before acceptance.
                    official = float(udp.fitness(c['x'].tolist())[0])
                    if abs(official - full) > 1e-4 or not np.isfinite(official):
                        raise RuntimeError('Official and scenario full scores disagree')
                    improved = official < best_score - 1e-8
                    full = official
                rows[c['trial'] - 1]['full'] = full
                rows[c['trial'] - 1]['accepted'] = improved
                if improved:
                    best = c['x'].copy()
                    best_score = full
                    base = dict(c['cache'])
                    won = True
                    kind = c['kind']
                    if kind in state['steps']:
                        state['steps'][kind] = min(0.2, state['steps'][kind] * 1.15)
                    state['rng_state'] = rng.bit_generator.state
                    atomic_save(best, state)
                    print(f'*** NEW VERIFIED RECORD: {best_score:.9f} ***', flush=True)
                print(f"Full {c['trial']} {c['kind']}: {full:.9f}", flush=True)
                if best_score < target:
                    break
            if won:
                state['stagnation'] = 0
            else:
                state['stagnation'] += 1

        if state['stagnation'] > 0:
            factor = 0.75 if state['stagnation'] % 3 else 1.7
            state['steps'] = {k: max(0.001, min(0.2, v * factor))
                              for k, v in state['steps'].items()}
        state['rounds_done'] = rnd
        state['rng_state'] = rng.bit_generator.state
        for r in rows:
            r['best_full'] = best_score
        atomic_save(best, state)
        write_log(rows)
        print(f"ROUND {rnd} DONE | best={best_score:.9f} | steps={state['steps']}", flush=True)

    print(f'\nFINAL OFFICIAL FITNESS: {best_score:.9f}', flush=True)
    print(f'Checkpoint: {BEST}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--seed', type=int, default=2038)
    parser.add_argument('--target', type=float, default=4.0)
    args = parser.parse_args()
    if args.rounds < 1 or args.trials < 16 or not np.isfinite(args.target):
        parser.error('rounds >= 1, trials >= 16, and finite target are required')
    optimize(args.rounds, args.trials, args.seed, args.target)
