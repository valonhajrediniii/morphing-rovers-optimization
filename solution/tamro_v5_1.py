"""TAMRO v5.1: paired local bias and parameter-block refinement.

Continues an ESA SpOC2 Morphing Rovers run from v5's verified checkpoint.
Only full 30-scenario official fitness can replace the incumbent.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import morphing_rover_UDP
from tamro_v4_4 import ALL, BIAS_END, CONT, MASKS, RESULTS, ROOT, evaluate, repair, validate

STATE = RESULTS / 'tamro_v5_1_state.npz'
BEST = RESULTS / 'tamro_v5_1_best.npy'
HISTORY = RESULTS / 'tamro_v5_1_history.csv'
OPS = ('BIAS_PAIR', 'BIAS_GROUP', 'BIAS_DIRECTION', 'WEIGHT_BLOCK', 'MASK_PATCH', 'ARCH_STEP')
COLS = ('round','trial','operator','sigma','quick','balanced','full','best_full','improved')


def atomic_npy(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, value)
    os.replace(tmp, path)


def save(best, direction, state, rng):
    state['rng_state'] = rng.bit_generator.state
    tmp = STATE.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, best=best, direction=direction, metadata=json.dumps(state))
    os.replace(tmp, STATE)
    atomic_npy(BEST, best)


def load(lo, hi, seed):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as data:
            best = data['best'].astype(float)
            direction = data['direction'].astype(float)
            state = json.loads(str(data['metadata'].item()))
        rng.bit_generator.state = state['rng_state']
        print(f"Resuming v5.1 after round {state['rounds_done']}", flush=True)
    else:
        sources = (RESULTS / 'tamro_v5_best.npy',
                   ROOT / 'solution' / 'checkpoints' / 'tamro_v5_best.npy')
        source = next((p for p in sources if p.exists()), None)
        if source is None:
            raise FileNotFoundError('Missing v5 best checkpoint (tamro_v5_best.npy)')
        best = np.load(source).astype(float)
        direction = np.zeros(BIAS_END - MASKS, dtype=float)
        state = dict(rounds_done=0, attempts=[0]*len(OPS), successes=[0]*len(OPS),
                     scales=[1.0]*len(OPS), stagnation=0, seed=seed)
        print(f'Starting from: {source}', flush=True)
    validate(best, lo, hi)
    if direction.shape != (BIAS_END - MASKS,):
        raise ValueError('Direction length does not match ESA biases')
    return best, direction, state, rng


def create_pair(best, op, sigma, direction, rng, lo, hi):
    delta = np.zeros_like(best)
    if op == 0:
        idx = rng.choice(np.arange(MASKS, BIAS_END), size=2, replace=False)
        delta[idx] = rng.normal(0, sigma, size=2)
    elif op == 1:
        n = int(rng.choice([6, 12, 24, 48]))
        idx = rng.choice(np.arange(MASKS, BIAS_END), size=n, replace=False)
        delta[idx] = rng.normal(0, sigma, size=n)
    elif op == 2:
        n = int(rng.choice([8, 16, 32]))
        ids = rng.choice(np.arange(BIAS_END - MASKS), size=n, replace=False)
        vec = direction[ids]
        if np.linalg.norm(vec) < 1e-10:
            vec = rng.standard_normal(n)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        delta[MASKS + ids] = sigma * np.sqrt(n) * vec
    elif op == 3:
        width = int(rng.choice([16, 32, 64, 128]))
        start = int(rng.integers(BIAS_END, CONT - width + 1))
        delta[start:start+width] = rng.normal(0, sigma, width)
    elif op == 4:
        mode = int(rng.integers(4))
        row = int(rng.integers(9))
        col = int(rng.integers(9))
        patch = np.zeros((11, 11), dtype=float)
        patch[row:row+3, col:col+3] = rng.normal(0, sigma, (3, 3))
        delta[mode*121:(mode+1)*121] = patch.reshape(-1)
    else:
        idx = int(rng.integers(CONT, CONT + 7))
        choices = [int(best[idx])+k for k in (-1, 1)]
        choices = [v for v in choices if lo[idx] <= v <= hi[idx]]
        if not choices:
            return []
        candidate = best.copy()
        candidate[idx] = int(rng.choice(choices))
        return [repair(candidate, lo, hi)]
    return [repair(best + delta, lo, hi), repair(best - delta, lo, hi)]


def select_cases(rng, baseline):
    # Two scenarios per map: one difficult, one random distinct scenario.
    quick = []
    for m in range(6):
        hard = max(range(5), key=lambda s: baseline[(m, s)])
        random_other = int(rng.choice([s for s in range(5) if s != hard]))
        quick += [(m, hard), (m, random_other)]
    balanced = list(quick)
    for c in sorted(ALL, key=lambda c: baseline[c], reverse=True):
        if c not in balanced:
            balanced.append(c)
        if len(balanced) >= 22:
            break
    return quick, balanced


def shortlist(ranked, limit):
    # Two strongest candidates, then operator diversity.
    selected = list(ranked[:min(2, limit)])
    seen_trials = {x['trial'] for x in selected}
    seen_ops = {x['op'] for x in selected}
    for item in ranked:
        if len(selected) >= limit:
            break
        if item['op'] not in seen_ops and item['trial'] not in seen_trials:
            selected.append(item)
            seen_trials.add(item['trial'])
            seen_ops.add(item['op'])
    for item in ranked:
        if len(selected) >= limit:
            break
        if item['trial'] not in seen_trials:
            selected.append(item)
            seen_trials.add(item['trial'])
    return selected


def log_round(records):
    exists = HISTORY.exists() and HISTORY.stat().st_size > 0
    with HISTORY.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            writer.writeheader()
        writer.writerows(records)


def optimize(rounds, trials, seed, target, finalists):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(v, dtype=float) for v in udp.get_bounds())
    if len(lo) != CONT+7:
        raise ValueError('Unexpected ESA chromosome size')
    best, direction, state, rng = load(lo, hi, seed)
    best_score = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(best_score):
        raise ValueError('Initial fitness is not finite')
    base_cache = {}
    baseline = evaluate(udp, best, ALL, base_cache)
    if not np.isfinite(baseline) or abs(baseline - best_score) > 1e-4:
        raise RuntimeError('Per-scenario score disagrees with ESA official fitness')
    print(f'Official starting fitness: {best_score:.9f}', flush=True)

    for _ in range(rounds):
        if best_score < target:
            print(f'Target already achieved: < {target}', flush=True)
            break
        r = state['rounds_done'] + 1
        quick, balanced = select_cases(rng, base_cache)
        qbase = float(np.mean([base_cache[c] for c in quick]))
        bbase = float(np.mean([base_cache[c] for c in balanced]))
        attempts = np.asarray(state['attempts'], dtype=float)
        successes = np.asarray(state['successes'], dtype=float)
        weights = 0.15 + (successes + 1) / (attempts + 2)
        weights[:3] *= 1.4
        weights /= weights.sum()
        seen = {best.tobytes()}
        candidates, records = [], {}
        print(f'\nROUND {r} | best={best_score:.9f} | quick baseline={qbase:.5f}', flush=True)
        draws = 0
        while len(records) < trials and draws < trials * 15:
            pair_id = draws
            draws += 1
            op = pair_id if pair_id < len(OPS) else int(rng.choice(len(OPS), p=weights))
            sigma = float(rng.choice([0.001, 0.003, 0.01, 0.03, 0.10, 0.3]))
            sigma *= float(state['scales'][op])
            if state['stagnation'] >= 4:
                sigma *= 1.5
            for x in create_pair(best, op, sigma, direction, rng, lo, hi):
                if len(records) >= trials:
                    break
                fingerprint = x.tobytes()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                trial = len(records) + 1
                cache = {}
                quick_score = evaluate(udp, x, quick, cache)
                records[trial] = dict(round=r, trial=trial, operator=OPS[op], sigma=sigma,
                                      quick=quick_score, balanced='', full='',
                                      best_full=best_score, improved=False)
                if np.isfinite(quick_score):
                    candidates.append(dict(trial=trial, op=op, x=x, cache=cache, quick=quick_score))
                print(f'Quick {trial}/{trials} {OPS[op]}: {quick_score:.6f} ({quick_score-qbase:+.6f})', flush=True)
        candidates.sort(key=lambda a: a['quick'])
        stage2 = shortlist(candidates, min(12, len(candidates)))
        for item in stage2:
            value = evaluate(udp, item['x'], balanced, item['cache'])
            item['balanced'] = value
            records[item['trial']]['balanced'] = value
            print(f'Balanced {item["trial"]} {OPS[item["op"]]}: {value:.6f} ({value-bbase:+.6f})', flush=True)
        stage2 = [a for a in stage2 if np.isfinite(a['balanced'])]
        stage2.sort(key=lambda a: a['balanced'])
        final = shortlist(stage2, min(finalists, len(stage2)))
        improved_round = False
        for item in final:
            score = evaluate(udp, item['x'], ALL, item['cache'])
            op = item['op']
            state['attempts'][op] += 1
            improved = bool(np.isfinite(score) and score < best_score - 1e-8)
            if improved:
                official = float(udp.fitness(item['x'].tolist())[0])
                if not np.isfinite(official) or abs(official-score) > 1e-4:
                    raise RuntimeError('ESA official score mismatch')
                improved = official < best_score - 1e-8
                score = official
            records[item['trial']]['full'] = score
            records[item['trial']]['improved'] = improved
            if improved:
                prev = best.copy()
                best = item['x'].copy()
                best_score = score
                state['successes'][op] += 1
                state['scales'][op] = min(3.0, state['scales'][op] * 1.08)
                direction = 0.85 * direction + 0.15 * (best[MASKS:BIAS_END] - prev[MASKS:BIAS_END])
                base_cache = dict(item['cache'])
                improved_round = True
                save(best, direction, state, rng)
                print(f'*** NEW OFFICIAL RECORD: {best_score:.9f} ***', flush=True)
            else:
                state['scales'][op] = max(0.15, state['scales'][op] * 0.97)
            print(f'Full {item["trial"]} {OPS[op]}: {score:.9f}', flush=True)
            if best_score < target:
                break
        state['stagnation'] = 0 if improved_round else state['stagnation'] + 1
        state['rounds_done'] = r
        for record in records.values():
            record['best_full'] = best_score
        save(best, direction, state, rng)
        log_round(list(records.values()))
        print(f'ROUND {r} DONE | best={best_score:.9f}', flush=True)
        if best_score < target:
            print(f'TARGET ACHIEVED: fitness < {target}', flush=True)
            break
    print(f'\nFINAL OFFICIAL FITNESS: {best_score:.9f}', flush=True)
    print(f'Checkpoint: {BEST}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--trials', type=int, default=24)
    p.add_argument('--seed', type=int, default=2034)
    p.add_argument('--target', type=float, default=4.0)
    p.add_argument('--finalists', type=int, default=6)
    args = p.parse_args()
    if args.rounds < 1 or args.trials < 12 or args.finalists < 1 or not np.isfinite(args.target):
        p.error('Require rounds >= 1, trials >= 12, finalists >= 1 and finite target')
    optimize(args.rounds, args.trials, args.seed, args.target, args.finalists)
