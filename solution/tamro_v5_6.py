"""TAMRO v5.6: momentum-line search + paired active-coordinate fusion.

An original follow-up to v5.5. Requires the ESA SpOC2 simulator and the
project's tamro_v4_4 compatibility helpers. The official 30-scenario
fitness function is the sole criterion for accepting a new best.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from morphing_udp import morphing_rover_UDP
from tamro_v4_4 import ALL, MASKS, BIAS_END, CONT, ROOT, RESULTS, evaluate, repair, validate

STATE = RESULTS / 'tamro_v5_6_state.npz'
BEST = RESULTS / 'tamro_v5_6_best.npy'
HISTORY = RESULTS / 'tamro_v5_6_history.csv'
COLUMNS = ('round', 'trial', 'kind', 'coordinate', 'multiplier', 'quick',
           'balanced', 'full', 'accepted', 'best_full')


def atomic_save(best, anchor, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, best=best, anchor=anchor,
                            metadata=json.dumps(state))
    os.replace(tmp, STATE)
    tmp = BEST.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, best)
    os.replace(tmp, BEST)


def write_history(rows):
    exists = HISTORY.exists() and HISTORY.stat().st_size > 0
    with HISTORY.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def load_chromosome(paths):
    for p in paths:
        if p.exists():
            return np.load(p).astype(float), p
    raise FileNotFoundError('Cannot find checkpoint among: ' + ', '.join(map(str, paths)))


def initialize(seed, lo, hi):
    rng = np.random.default_rng(seed)
    if STATE.exists():
        with np.load(STATE, allow_pickle=False) as z:
            best = z['best'].astype(float)
            anchor = z['anchor'].astype(float)
            state = json.loads(str(z['metadata'].item()))
        rng.bit_generator.state = state['rng_state']
        print(f"Resuming TAMRO v5.6 after round {state['rounds_done']}", flush=True)
    else:
        best, source = load_chromosome([
            RESULTS / 'tamro_v5_5_best.npy',
            ROOT / 'solution' / 'checkpoints' / 'tamro_v5_5_best.npy',
        ])
        anchor, anchor_source = load_chromosome([
            RESULTS / 'tamro_v5_best.npy',
            ROOT / 'solution' / 'checkpoints' / 'tamro_v5_best.npy',
        ])
        state = {'rounds_done': 0, 'stagnation': 0,
                 'steps': {'BIAS': .008, 'WEIGHT': .009, 'MASK': .006},
                 'rng_state': rng.bit_generator.state, 'seed': seed}
        print(f'Starting from: {source}', flush=True)
        print(f'Previous anchor: {anchor_source}', flush=True)
    validate(best, lo, hi)
    validate(anchor, lo, hi)
    if len(lo) != 19133 or CONT != 19126:
        raise RuntimeError('Unexpected ESA chromosome layout')
    return best, anchor, state, rng


def scenario_subset(rng, baseline):
    """Include high-loss cases and random per-map coverage; then 22/30."""
    ranked = sorted(ALL, key=lambda k: baseline[k], reverse=True)
    quick = list(ranked[:5])
    for m in range(6):
        case = (m, int(rng.integers(5)))
        if case not in quick:
            quick.append(case)
    while len(quick) < 12:
        case = ALL[int(rng.integers(30))]
        if case not in quick:
            quick.append(case)
    balanced = list(quick)
    for case in ranked:
        if len(balanced) >= 22:
            break
        if case not in balanced:
            balanced.append(case)
    while len(balanced) < 22:
        case = ALL[int(rng.integers(30))]
        if case not in balanced:
            balanced.append(case)
    return quick, balanced


def choose_coordinate(rng, changed, seen):
    """Prefer recently useful dimensions, retain independent exploration."""
    for _ in range(50):
        category = str(rng.choice(['BIAS', 'WEIGHT', 'MASK'], p=[.46, .36, .18]))
        if len(changed) and rng.random() < .60:
            restricted = changed[(changed >= (MASKS if category == 'BIAS' else
                                               BIAS_END if category == 'WEIGHT' else 0)) &
                                 (changed < (BIAS_END if category == 'BIAS' else
                                             CONT if category == 'WEIGHT' else MASKS))]
        else:
            restricted = np.array([], dtype=int)
        if len(restricted):
            idx = int(rng.choice(restricted))
        elif category == 'BIAS':
            idx = int(rng.integers(MASKS, BIAS_END))
        elif category == 'WEIGHT':
            idx = int(rng.integers(BIAS_END, CONT))
        else:
            idx = int(rng.integers(0, MASKS))
        if idx not in seen:
            seen.add(idx)
            return category, idx
    return category, idx


def select_diverse(ranked, wanted, key):
    ranked = sorted(ranked, key=lambda c: c[key])
    output = list(ranked[:min(3, wanted)])
    used = {c['trial'] for c in output}
    kinds = {c['kind'] for c in output}
    for c in ranked:
        if len(output) >= wanted:
            break
        if c['kind'] not in kinds and c['trial'] not in used:
            output.append(c)
            used.add(c['trial'])
            kinds.add(c['kind'])
    for c in ranked:
        if len(output) >= wanted:
            break
        if c['trial'] not in used:
            output.append(c)
            used.add(c['trial'])
    return output


def optimize(rounds, trials, seed, target):
    RESULTS.mkdir(parents=True, exist_ok=True)
    udp = morphing_rover_UDP()
    lo, hi = (np.asarray(b, dtype=float) for b in udp.get_bounds())
    best, anchor, state, rng = initialize(seed, lo, hi)
    official_best = float(udp.fitness(best.tolist())[0])
    if not np.isfinite(official_best):
        raise RuntimeError('Non-finite official fitness')
    baseline = {}
    by_case = evaluate(udp, best, ALL, baseline)
    if abs(by_case - official_best) > 1e-4:
        raise RuntimeError('ESA official fitness differs from scenario mean')
    print(f'Official starting fitness: {official_best:.9f}', flush=True)
    print(f'Previous winning direction coordinates: '
          f'{np.count_nonzero(np.abs(best[:CONT]-anchor[:CONT]) > 1e-12)}',
          flush=True)

    for _ in range(rounds):
        if official_best < target:
            print(f'Target already achieved (< {target})', flush=True)
            break
        rid = state['rounds_done'] + 1
        quick, balanced = scenario_subset(rng, baseline)
        qbase = float(np.mean([baseline[c] for c in quick]))
        bbase = float(np.mean([baseline[c] for c in balanced]))
        print(f'\nROUND {rid} | official={official_best:.9f} '
              f'| quick={qbase:.5f} balanced={bbase:.5f}', flush=True)

        rows, candidates, scouts = [], [], []
        signatures = {best.tobytes()}
        delta_winner = best - anchor
        delta_winner[CONT:] = 0.0  # never interpolate discrete choices
        changed = np.flatnonzero(np.abs(delta_winner[:CONT]) > 1e-12)

        def add(kind, coordinate, multiplier, raw):
            x = repair(raw, lo, hi)
            key = x.tobytes()
            if key in signatures or len(rows) >= trials:
                return None
            signatures.add(key)
            cache = {}
            quick_score = evaluate(udp, x, quick, cache)
            if not np.isfinite(quick_score):
                return None
            n = len(rows) + 1
            record = {'round': rid, 'trial': n, 'kind': kind,
                      'coordinate': coordinate, 'multiplier': multiplier,
                      'quick': quick_score, 'balanced': '', 'full': '',
                      'accepted': False, 'best_full': official_best}
            rows.append(record)
            cand = {'trial': n, 'kind': kind, 'x': x, 'cache': cache,
                    'quick': quick_score}
            candidates.append(cand)
            print(f'Quick {n}/{trials} {kind}: {quick_score:.6f} '
                  f'(delta {quick_score-qbase:+.6f})', flush=True)
            return cand

        # Phase A: direction discovered in v5.5's successful FUSION.
        line_budget = min(8, max(4, trials // 3))
        if len(changed):
            gains = [.05, .10, .20, .35, .60, 1.0, -.10, -.30]
            if state['stagnation'] >= 3:
                gains = [.025, .05, .10, .20, -.025, -.05, .60, 1.2]
            for gain in gains[:line_budget]:
                add('WINNER_LINE', -1, gain,
                    best + gain * delta_winner)

        # Phase B: probing positive and negative steps on the same coordinate.
        budget_fusions = min(4, max(2, trials // 6))
        probes = max(0, (trials - len(rows) - budget_fusions) // 2)
        visited = set()
        for _probe in range(probes):
            kind, index = choose_coordinate(rng, changed, visited)
            step = state['steps'][kind] * float(rng.choice([.25, .5, 1, 2, 4]))
            step *= max(1., abs(float(best[index])))
            for sign in (1, -1):
                x = best.copy()
                x[index] += sign * step
                result = add(kind, index, sign * step, x)
                if result is not None:
                    scouts.append((result['quick'] - qbase,
                                   index, sign * step))

        # Phase C: recombine top coordinate directions, locally or with momentum.
        scouts.sort(key=lambda a: a[0])
        deduped = []
        used_indices = set()
        for gain, index, shift in scouts:
            if index not in used_indices:
                deduped.append((gain, index, shift))
                used_indices.add(index)
        for k in range(budget_fusions):
            if not deduped:
                break
            x = best.copy()
            scale = [.20, .40, .70, 1.0][k]
            for _, index, shift in deduped[:min(k + 2, len(deduped))]:
                x[index] += scale * shift
            if len(changed) and k % 2 == 1:
                x += (.05 if k == 1 else .15) * delta_winner
            add('FUSION', -1, scale, x)

        # Screen candidates on the same scenarios as the baseline.
        screened = select_diverse(candidates, min(12, len(candidates)), 'quick')
        for c in screened:
            b = evaluate(udp, c['x'], balanced, c['cache'])
            c['balanced'] = b
            rows[c['trial'] - 1]['balanced'] = b
            print(f'Balanced {c["trial"]} {c["kind"]}: {b:.6f} '
                  f'(delta {b-bbase:+.6f})', flush=True)
        eligible = [c for c in screened if np.isfinite(c['balanced'])]
        finalists = select_diverse(eligible, min(6, len(eligible)), 'balanced')

        improved_round = False
        for c in finalists:
            # Reuse cache to finish all 30 scenario scores, then recheck official.
            score = evaluate(udp, c['x'], ALL, c['cache'])
            won = bool(np.isfinite(score) and score < official_best - 1e-8)
            if won:
                official = float(udp.fitness(c['x'].tolist())[0])
                if not np.isfinite(official) or abs(official - score) > 1e-4:
                    raise RuntimeError('Official fitness recheck mismatch')
                won = official < official_best - 1e-8
                score = official
            rec = rows[c['trial'] - 1]
            rec['full'] = score
            rec['accepted'] = won
            print(f'Full {c["trial"]} {c["kind"]}: {score:.9f}', flush=True)
            if won:
                anchor = best.copy()
                best = c['x'].copy()
                official_best = score
                baseline = dict(c['cache'])
                state['stagnation'] = 0
                state['rng_state'] = rng.bit_generator.state
                atomic_save(best, anchor, state)
                improved_round = True
                print(f'*** NEW VERIFIED RECORD: {official_best:.9f} ***', flush=True)
            if official_best < target:
                break

        if not improved_round:
            state['stagnation'] += 1
            # Conservative shrinking, occasional bounded broadening.
            factor = .75 if state['stagnation'] % 4 else 1.5
            state['steps'] = {
                k: float(np.clip(v * factor, .0002, .08))
                for k, v in state['steps'].items()
            }
        state['rounds_done'] = rid
        state['rng_state'] = rng.bit_generator.state
        for rec in rows:
            rec['best_full'] = official_best
        atomic_save(best, anchor, state)
        write_history(rows)
        print(f'ROUND {rid} DONE | best={official_best:.9f} '
              f'| steps={state["steps"]}', flush=True)

    print(f'\nFINAL OFFICIAL FITNESS: {official_best:.9f}', flush=True)
    print(f'Checkpoint: {BEST}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--seed', type=int, default=2039)
    parser.add_argument('--target', type=float, default=4.0)
    args = parser.parse_args()
    if args.rounds < 1 or args.trials < 16:
        parser.error('Require --rounds >= 1 and --trials >= 16')
    optimize(args.rounds, args.trials, args.seed, args.target)
