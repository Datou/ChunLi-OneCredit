"""Evaluate full natural one-credit campaigns, with incremental results and load auditing."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chunppo import evaluate

# Verified by reading Runtime.enemy_id() after loading each of the 12 stage states.
NATIVE_IDENTITY = {5: 'ChunLi', 6: 'Zangief', 7: 'Dhalsim', 0: 'Ryu', 1: 'EHonda',
                   2: 'Blanka', 3: 'Guile', 4: 'Ken', 10: 'Balrog', 11: 'Vega',
                   9: 'Sagat', 8: 'Bison'}
ROSTER = ('ChunLi', 'Zangief', 'Dhalsim', 'Ryu', 'EHonda', 'Blanka', 'Guile', 'Ken',
          'Balrog', 'Vega', 'Sagat', 'Bison')


def one_credit(job):
    from chunppo.env import Runtime
    original_load, original_set = Runtime.load_stage, Runtime.set_state
    audit = {'stage_load_calls': 0, 'state_load_calls': 0, 'initial_stage': None}

    def checked_load(runtime, stage, offset=0, rng=None):
        audit['stage_load_calls'] += 1
        audit['initial_stage'] = stage
        if audit['stage_load_calls'] != 1 or stage != 1:
            raise RuntimeError('Campaign attempted to load a stage after its initial stage-1 start')
        return original_load(runtime, stage, offset, rng)

    def checked_set(runtime, blob):
        audit['state_load_calls'] += 1
        if audit['state_load_calls'] != 1:
            raise RuntimeError('Campaign attempted to restore an emulator state mid-run')
        return original_set(runtime, blob)

    Runtime.load_stage, Runtime.set_state = checked_load, checked_set
    try:
        result = evaluate.campaign(job)
    finally:
        Runtime.load_stage, Runtime.set_state = original_load, original_set
    if audit != {'stage_load_calls': 1, 'state_load_calls': 1, 'initial_stage': 1}:
        raise RuntimeError('Campaign did not use exactly one initial stage-1 state')
    ids = [match['enemy'] for match in result['matches']]
    if len(ids) != len(set(ids)) or any(enemy not in NATIVE_IDENTITY for enemy in ids):
        raise RuntimeError('Duplicate or unknown opponent in campaign')
    if result['cleared'] and (len(ids) != 12 or not all(m['win'] for m in result['matches'])):
        raise RuntimeError('Clear does not contain twelve distinct won matches')
    result['one_credit_audit'] = audit
    for match in result['matches']:
        match['opponent'] = NATIVE_IDENTITY[match['enemy']]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--campaigns', type=int, default=20)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--sync-round-observation', action='store_true',
                        help='Opt-in: commit the current post-transition screen at the next decision')
    parser.add_argument('--resume', action='store_true', help='Resume a matching incomplete evaluation')
    args = parser.parse_args()
    if args.campaigns < 1 or not 1 <= args.workers <= 6:
        raise ValueError('Campaign count must be positive, with one to six workers')
    model, out = Path(args.model), Path(args.out)
    keys = ('offset', 'jitter', 'seed', 'deterministic')
    prior, source_sha = None, None
    if args.reference:
        reference = Path(args.reference)
        source = (reference / 'results.jsonl').read_bytes()
        prior = [json.loads(line) for line in source.decode('utf-8').splitlines() if line.strip()]
        prior = [row for row in prior if 'cleared' in row]
        jobs = [dict({key: row[key] for key in keys}, case_index=i) for i, row in enumerate(prior)]
        source_sha = hashlib.sha256(source).hexdigest()
    else:
        rng, jobs = random.Random(args.seed), []
        for jitter in (0.0, 0.05):
            for _ in range(args.campaigns):
                jobs.append(dict(offset=rng.randrange(31, 91), jitter=jitter,
                                 seed=rng.randrange(2**30), deterministic=True, case_index=len(jobs)))
    if (collections.Counter(job['jitter'] for job in jobs) !=
            collections.Counter({0.0: args.campaigns, 0.05: args.campaigns}) or
            not all(job['deterministic'] for job in jobs)):
        raise ValueError('Unexpected campaign reference cases')
    with zipfile.ZipFile(model) as archive:
        stored = int(json.loads(archive.read('data'))['num_timesteps'])
    existed = out.exists()
    out.mkdir(parents=True, exist_ok=args.resume)
    protocol = dict(model=str(model.resolve()), model_stored_timesteps=stored,
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    out=str(out.resolve()), reference=args.reference, source_results_sha256=source_sha,
                    fixtures=0, campaigns=args.campaigns, jitters='0,0.05', stochastic=False,
                    seed=args.seed, workers=args.workers, hist_len=24, delay=0, jobs=len(jobs),
                    initial_stage=1, mid_campaign_loads_allowed=0, continue_after_lost_match=False,
                    native_enemy_identity=NATIVE_IDENTITY,
                    campaign_source_sha256=hashlib.sha256((ROOT/'chunppo/evaluate.py').read_bytes()).hexdigest(),
                    sync_round_observation=args.sync_round_observation,
                    created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), cases=jobs)
    results, started = [], time.time()
    if existed:
        old = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        for key in ('model_sha256', 'cases'):
            if old[key] != protocol[key]:
                raise ValueError('Resume protocol mismatch: %s' % key)
        if old.get('sync_round_observation', False) != args.sync_round_observation:
            raise ValueError('Resume observation timing mismatch')
        result_file = out/'results.jsonl'
        if result_file.exists():
            results = [json.loads(line) for line in result_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        expected = {job['case_index']: job for job in jobs}
        if len({r['case_index'] for r in results}) != len(results):
            raise ValueError('Duplicate existing result')
        for row in results:
            if row['case_index'] not in expected or any(row[k] != expected[row['case_index']][k] for k in keys):
                raise ValueError('Existing result does not match resume job')
        if (out/'DONE').exists() and (out/'summary.json').exists():
            print('Already completed', flush=True)
            return
    else:
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    completed = {row['case_index'] for row in results}
    remaining_jobs = [job for job in jobs if job['case_index'] not in completed]
    with (out/'results.jsonl').open('a' if existed else 'x', encoding='utf-8') as stream:
        with mp.get_context('spawn').Pool(args.workers, initializer=evaluate._init,
                                         initargs=(str(model), dict(hist_len=24, delay=0,
                                             sync_round_observation=args.sync_round_observation))) as pool:
            for result in pool.imap_unordered(one_credit, remaining_jobs, chunksize=1):
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                last = result['matches'][-1]['opponent'] if result['matches'] else 'none'
                print('completed %d/%d jitter=%s wins=%d/12 reason=%s last=%s elapsed=%.1fs' %
                      (len(results), len(jobs), result['jitter'], result['wins'],
                       result['reason'], last, time.time()-started), flush=True)
    if len(results) != len(jobs) or len({row['case_index'] for row in results}) != len(jobs):
        raise RuntimeError('Missing or duplicate campaigns')
    summary = dict(model=str(model.resolve()), model_stored_timesteps=stored,
                   seconds=round(time.time()-started, 1), jobs=len(results), by_jitter={})
    for jitter in (0.0, 0.05):
        rows = [row for row in results if row['jitter']==jitter]
        clears = sum(row['cleared'] for row in rows)
        opponents = {}
        for name in ROSTER:
            matches = [match for row in rows for match in row['matches'] if match['opponent']==name]
            n, wins = len(matches), sum(match['win'] for match in matches)
            opponents[name] = dict(reached=n, wins=wins, losses=n-wins,
                                   win_rate=round(wins/n, 4) if n else None)
        group = dict(campaign_clears='%d/%d' % (clears, len(rows)),
                     campaign_clear_rate=round(clears/len(rows), 4),
                     campaign_ci95=evaluate.wilson(clears, len(rows)),
                     campaign_mean_wins=round(sum(row['wins'] for row in rows)/len(rows), 2),
                     campaign_max_wins=max(row['wins'] for row in rows), per_opponent=opponents,
                     campaign_stop_reasons=dict(collections.Counter(row['reason'] for row in rows)),
                     losses_by_opponent=dict(collections.Counter(
                         row['matches'][-1]['opponent'] for row in rows if row['reason']=='lost_match')))
        if prior is not None:
            group['paired_improved_clears'] = sum(row['cleared'] and not prior[row['case_index']]['cleared'] for row in rows)
            group['paired_regressed_clears'] = sum(not row['cleared'] and prior[row['case_index']]['cleared'] for row in rows)
        summary['by_jitter'][str(jitter)] = group
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'DONE').write_text(str(len(results)), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
