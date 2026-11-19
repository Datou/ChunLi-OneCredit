"""Evaluation: (a) per-stage fixture matrix on HELD-OUT start offsets x jitter levels,
(b) natural one-credit campaigns from stage 1 (no reloads, first lost match ends it).

  .venv\\Scripts\\python.exe -m chunppo.evaluate --model runs/chunppo/E3/ckpt_xxx.zip --out runs/chunppo/E3/eval_xxx
      [--fixtures 8] [--jitters 0,0.05] [--campaigns 20] [--workers 12]

Training offsets are drawn from [0, max_offset=30] with the training RNG; held-out
offsets are 31..90 with an independent walk RNG, so a policy that memorised
training trajectories cannot score here. Memorisation gap = train-curve win rate
minus held-out fixture win rate at the same jitter.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np

_MODEL = None
_ARGS = None


def _init(model_path, args):
    global _MODEL, _ARGS
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import cv2
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from stable_baselines3 import PPO
    from chunppo.policy import POLICY_KWARGS
    _MODEL = PPO.load(model_path, device='cpu', custom_objects={'policy_kwargs': POLICY_KWARGS})
    _ARGS = args
    if args.get('sync_round_observation', False):
        from .roundsync import install
        install()


def _act(obs, deterministic):
    action, _ = _MODEL.predict(obs, deterministic=deterministic)
    return int(action)


def fixture(job):
    from chunppo.env import ChunEnv
    env = ChunEnv(seed=job['seed'], stages=(job['stage'],), hist_len=_ARGS['hist_len'], delay=_ARGS['delay'])
    try:
        env.set_jitter(job['jitter'])
        env.force_next(job['stage'], job['offset'])
        obs, done = env.reset(), False
        while not done:
            obs, _, done, info = env.step(_act(obs, job['deterministic']))
        return dict(job, **info['match'])
    finally:
        env.close()


def campaign(job, limit=400_000, act=None, hook=None, on_match_start=None):
    """Run a continuous campaign. A capture callback requests frame-by-frame intros."""
    act = act or _act
    from chunppo.env import MatchTracker, Observer, Runtime
    rt = Runtime(job['seed'], job['jitter'], (_ARGS or {}).get('delay', 0))
    hist_len = (_ARGS or {}).get('hist_len', 24)
    try:
        rt.load_stage(1, job['offset'], random.Random(job['seed']))
        observer = Observer(hist_len)
        enemy, beaten, matches, reason = rt.enemy_id(), [], [], None
        while reason is None:
            observer.reset(rt.screen)
            tr = MatchTracker(rt)
            while True:
                a = act(observer.obs(), job['deterministic'])
                for _ in rt.macro(a):
                    pass
                if hook:
                    hook(rt)
                tr.update()
                observer.push(rt.screen, a)
                if not tr.live:
                    _, over, trunc = tr.settle()
                    if trunc:
                        reason = 'settle_timeout'
                    if over or trunc:
                        break
                if rt.total_frames > limit:
                    reason = 'frame_limit'
                    break
            if reason:
                break
            win = tr.score[0] >= 2
            matches.append(dict(enemy=enemy, score=list(tr.score), dealt=tr.dealt, taken=tr.taken, win=win))
            if not win:
                reason = 'lost_match'
                break
            beaten.append(enemy)
            if len(beaten) == 12:
                reason = 'cleared'
                break
            seen_lock = False
            step = 1 if on_match_start is not None else 4
            for _ in range(20_000 // step):
                rt.neutral(step)
                e = rt.enemy_id()
                if not rt.controllable():
                    seen_lock = True
                elif (seen_lock and e < 12 and e not in beaten and rt.round_state() == 5
                        and min(rt.hp()) > 100 and rt.score() == (0, 0) and rt.bonus() == 0):
                    if on_match_start is not None and not on_match_start(rt, e):
                        continue
                    enemy = e
                    break
            else:
                reason = 'next_match_timeout'
        return dict(job, cleared=reason == 'cleared', reason=reason, wins=len(beaten),
                    matches=matches, frames=rt.total_frames)
    finally:
        rt.game.close()


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def run(kind, job):
    return fixture(job) if kind == 'fixture' else campaign(job)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fixtures', type=int, default=8, help='held-out offsets per stage')
    p.add_argument('--jitters', default='0,0.05')
    p.add_argument('--campaigns', type=int, default=20, help='per jitter level')
    p.add_argument('--stochastic', action='store_true', help='sample actions instead of argmax')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--hist-len', type=int, default=24)
    p.add_argument('--delay', type=int, default=0, help='fixed capture->execution delay (must match training)')
    p.add_argument('--seed', type=int, default=777)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    jitters = [float(j) for j in args.jitters.split(',')]
    det = not args.stochastic
    jobs = []
    for j in jitters:
        for stage in range(1, 13):
            for off in rng.sample(range(31, 91), args.fixtures):
                jobs.append(('fixture', dict(stage=stage, offset=off, jitter=j, seed=rng.randrange(2**30),
                                             deterministic=det)))
        for c in range(args.campaigns):
            jobs.append(('campaign', dict(offset=rng.randrange(31, 91), jitter=j, seed=rng.randrange(2**30),
                                          deterministic=det)))
    (out / 'protocol.json').write_text(json.dumps(dict(vars(args), jobs=len(jobs)), indent=2), encoding='utf8')
    t0, results = time.time(), []
    with mp.get_context('spawn').Pool(args.workers, initializer=_init,
                                      initargs=(args.model, dict(hist_len=args.hist_len, delay=args.delay))) as pool:
        for r in pool.starmap(run, jobs):
            results.append(r)
    with open(out / 'results.jsonl', 'w', encoding='utf8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    summary = dict(model=args.model, seconds=round(time.time() - t0, 1), by_jitter={})
    for j in jitters:
        fx = [r for r in results if 'stage' in r and r['jitter'] == j]
        cp = [r for r in results if 'cleared' in r and r['jitter'] == j]
        stages = {s: [r['win'] for r in fx if r['stage'] == s] for s in range(1, 13)}
        k = sum(r['cleared'] for r in cp)
        summary['by_jitter'][str(j)] = dict(
            fixture_win=round(float(np.mean([r['win'] for r in fx])), 4),
            per_stage={s: '%d/%d' % (sum(v), len(v)) for s, v in stages.items()},
            campaign_clears='%d/%d' % (k, len(cp)), campaign_ci95=wilson(k, len(cp)),
            campaign_mean_wins=round(float(np.mean([r['wins'] for r in cp])), 2) if cp else None,
            campaign_stop_reasons={x: sum(r['reason'] == x for r in cp) for x in {r['reason'] for r in cp}})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
