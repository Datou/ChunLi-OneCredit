"""Collect new-seed natural campaigns for fine tuning, never evaluation cases."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

from .common import ROSTER


def collect(job):
    from . import evaluate
    from .env import Runtime, Observer, MatchTracker
    from .statebank import capture, encode
    rt = Runtime(job['seed'], job['jitter'], 0)
    out = Path(job['out']) / ('case%03d' % job['index'])
    out.mkdir(exist_ok=False)
    entries, matches, beaten = [], [], []
    native_to_stage = {5: 1, 6: 2, 7: 3, 0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 10: 9, 11: 10, 9: 11, 8: 12}

    def save(blob, stage, kind, decision):
        path = out / ('stage%02d_%s_%05d.statepack' % (stage, kind, decision))
        path.write_bytes(blob)
        entries.append(dict(path=str(path.resolve()), stage=stage, kind=kind, decision=decision,
                            source_seed=job['seed'], source_jitter=job['jitter'],
                            sha256=hashlib.sha256(blob).hexdigest()))

    try:
        rt.load_stage(1, job['offset'], random.Random(job['seed']))
        observer = Observer(24)
        reason = None
        while reason is None:
            enemy = rt.enemy_id()
            stage = native_to_stage[enemy]
            observer.reset(rt.screen)
            tr = MatchTracker(rt)
            save(encode(capture(rt, observer, tr)), stage, 'entry', 0)
            tail = collections.deque(maxlen=4)
            recovery = collections.deque(maxlen=6)
            last_recovery = -24
            decision = 0
            while True:
                a = evaluate._act(observer.obs(), True)
                for _ in rt.macro(a):
                    pass
                previous_taken = tr.taken
                tr.update()
                observer.push(rt.screen, a)
                decision += 1
                if tr.live and rt.controllable() and decision % 16 == 0:
                    tail.append((decision, encode(capture(rt, observer, tr))))
                if (stage in (7, 8, 11, 12) and tr.live and rt.controllable()
                        and tr.taken > previous_taken and decision-last_recovery >= 24
                        and rt.hp()[0] >= 24):
                    recovery.append((decision, encode(capture(rt, observer, tr))))
                    last_recovery = decision
                if not tr.live:
                    _, over, trunc = tr.settle()
                    if trunc:
                        reason = 'settle_timeout'
                    if over or trunc:
                        break
                if rt.total_frames > 400000:
                    reason = 'frame_limit'
                    break
            if reason:
                break
            win = tr.score[0] >= 2
            for d, blob in tail:
                save(blob, stage, 'tail_win' if win else 'pre_loss', d)
            for d, blob in recovery:
                save(blob, stage, 'recovery', d)
            matches.append(dict(stage=stage, opponent=ROSTER[stage-1], win=win,
                                score=list(tr.score), decisions=decision, dealt=tr.dealt, taken=tr.taken))
            if not win:
                reason = 'lost_match'
                break
            beaten.append(enemy)
            if len(beaten) == 12:
                reason = 'cleared'
                break
            seen_lock = False
            for _ in range(5000):
                rt.neutral(4)
                e = rt.enemy_id()
                if not rt.controllable():
                    seen_lock = True
                elif (seen_lock and e < 12 and e not in beaten and rt.round_state() == 5
                      and min(rt.hp()) > 100 and rt.score() == (0, 0) and rt.bonus() == 0):
                    break
            else:
                reason = 'next_match_timeout'
        result = dict(job, reason=reason, wins=len(beaten), matches=matches,
                      entries=entries, frames=rt.total_frames, initial_loads=1, mid_campaign_loads=0)
        (out/'result.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        rt.game.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--campaigns', type=int, default=24)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=6051105)
    a = p.parse_args()
    from .evaluate import _init
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(a.seed)
    jobs = [dict(index=i, offset=rng.randint(0, 90), jitter=(0.0 if i%2 == 0 else 0.05),
                 seed=rng.randrange(2**30), out=str(out)) for i in range(a.campaigns)]
    protocol = dict(model=str(Path(a.model).resolve()),
                    model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                    seed=a.seed, training_only=True, source='new natural campaigns', jobs=jobs)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'),
                                               initializer=_init, initargs=(a.model, dict(hist_len=24, delay=0))) as pool:
        for result in pool.map(collect, jobs):
            results.append(result)
            print(json.dumps(dict(completed=len(results), total=len(jobs), wins=result['wins'],
                                  jitter=result['jitter'], seconds=round(time.time()-start, 1))), flush=True)
    entries = [e for r in results for e in r['entries']]
    manifest = dict(protocol=protocol, campaign_count=len(results), entries=entries,
                    by_stage=dict(collections.Counter(e['stage'] for e in entries)),
                    by_kind=dict(collections.Counter(e['kind'] for e in entries)),
                    results=[{k: v for k, v in r.items() if k != 'entries'} for r in results])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'DONE').write_text(str(len(entries)))
    print(json.dumps(dict(entries=len(entries), by_stage=manifest['by_stage'], seconds=time.time()-start)), flush=True)


if __name__ == '__main__':
    main()
