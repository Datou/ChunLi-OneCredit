"""Frozen-checkpoint fine tuning with complete natural reset states and mixed jitter."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .env import ChunEnv, MatchTracker
from .statebank import capture, decode, encode, restore


JITTERS = (0.0, 0.01, 0.03, 0.05, 0.08)
JITTER_WEIGHTS = (0.25, 0.2, 0.2, 0.25, 0.1)
WEAK_STAGES = (7, 8, 11, 12)


class FineTuneEnv(ChunEnv):
    def __init__(self, bank, seed=0, stage_copies=None, natural_entry_fraction=0.70,
                 hard_mid_probability=0.0, loss_mid_probability=0.0, **kwargs):
        super().__init__(seed=seed, pool_prob=0.0, **kwargs)
        self.stage_copies = tuple(stage_copies) if stage_copies is not None else tuple(
            3 if s in WEAK_STAGES else 0 for s in self.stages)
        if len(self.stage_copies) != len(self.stages) or min(self.stage_copies) < 0:
            raise ValueError('stage_copies needs one nonnegative count per stage')
        self.natural_entry_fraction = float(natural_entry_fraction)
        self.hard_mid_probability = float(hard_mid_probability)
        self.loss_mid_probability = float(loss_mid_probability)
        if not 0 <= self.natural_entry_fraction <= 1 or not 0 <= self.hard_mid_probability <= 1:
            raise ValueError('Sampling probabilities must be in [0,1]')
        if not 0 <= self.loss_mid_probability <= 1:
            raise ValueError('Loss snapshot probability must be in [0,1]')
        manifest = json.loads((Path(bank)/'manifest.json').read_text())
        self.bank = {stage: collections.defaultdict(list) for stage in self.stages}
        for item in manifest['entries']:
            self.bank[item['stage']]['entry' if item['kind'] == 'entry' else 'mid'].append(item['path'])
            if item['kind'] in ('pre_loss', 'recovery'):
                self.bank[item['stage']]['hard_mid'].append(item['path'])
            if item.get('source_ken_win') is False and item['kind'] != 'entry':
                self.bank[item['stage']]['loss_mid'].append(item['path'])
        missing = [s for s in self.stages if not self.bank[s]['entry']]
        if missing:
            raise ValueError('No natural entry states for stages %s' % missing)
        self._cache = {}
        self.source = None

    def _next_stage(self):
        if not self._cycle:
            self._cycle = list(self.stages) + [s for s, n in zip(self.stages, self.stage_copies)
                                              for _ in range(n)]
            self.rng.shuffle(self._cycle)
        return self._cycle.pop()

    def reset(self):
        self.stage = self._next_stage()
        jitter = self.rng.choices(JITTERS, JITTER_WEIGHTS)[0]
        self.rt.fifo.p = jitter
        choice = self.rng.random()
        pool = self.pool[self.stage]
        self.from_pool = False
        if choice < 0.60:
            items = self.bank[self.stage]
            kind = 'entry' if self.rng.random() < self.natural_entry_fraction or not items['mid'] else 'mid'
            if (kind == 'mid' and items['loss_mid'] and self.loss_mid_probability > 0
                    and self.rng.random() < self.loss_mid_probability):
                kind = 'loss_mid'
            if (kind == 'mid' and items['hard_mid'] and self.hard_mid_probability > 0
                    and self.rng.random() < self.hard_mid_probability):
                kind = 'hard_mid'
            path = self.rng.choice(items[kind])
            if path not in self._cache:
                self._cache[path] = Path(path).read_bytes()
            self.tracker = restore(self.rt, self.observer, decode(self._cache[path]),
                                   jitter=jitter, restore_rng=False)
            self.source, self.offset = 'natural_' + kind, -2
            self.from_pool = True
        elif choice < 0.75 and pool:
            self.tracker = restore(self.rt, self.observer, decode(self.rng.choice(pool)),
                                   jitter=jitter, restore_rng=False)
            self.source, self.offset, self.from_pool = 'online_complete', -1, True
        else:
            self.offset = self.rng.randint(0, self.max_offset)
            self.rt.load_stage(self.stage, self.offset, self.rng)
            self.tracker = MatchTracker(self.rt)
            self.observer.reset(self.rt.screen)
            self.source = 'fresh_fixture'
        self.decisions = 0
        return self.observer.obs()

    def step(self, action):
        action = int(action)
        tr = self.tracker
        for _ in self.rt.macro(action):
            pass
        raw = tr.update()
        done = truncated = False
        save_snapshot = tr.live and self.rng.random() < self.snapshot_p
        if not tr.live:
            r2, done, truncated = tr.settle()
            raw += r2
        self.decisions += 1
        if self.decisions >= self.max_decisions and not done:
            truncated = True
        self.observer.push(self.rt.screen, action)
        if save_snapshot and not done and not truncated and self.rt.controllable():
            self.pool[self.stage].append(encode(capture(self.rt, self.observer, tr)))
        info = {}
        if done or truncated:
            info['TimeLimit.truncated'] = truncated and not done
            info['match'] = dict(stage=self.stage, win=bool(tr.score[0] >= 2), score=list(tr.score),
                                 dealt=tr.dealt, taken=tr.taken, from_pool=self.from_pool,
                                 source=self.source, offset=self.offset, jitter=self.rt.fifo.p,
                                 decisions=self.decisions, truncated=truncated)
        return self.observer.obs(), 0.01 * raw - self.time_penalty, bool(done or truncated), info


def make_env(rank, args):
    def init():
        import cv2
        import torch
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        return FineTuneEnv(args.bank, seed=args.seed*1000+rank, hist_len=24, delay=0,
                           stage_copies=args.stage_copies,
                           natural_entry_fraction=args.natural_entry_fraction,
                           hard_mid_probability=args.hard_mid_probability,
                           loss_mid_probability=args.loss_mid_probability)
    return init


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--total', type=int, required=True, help='Absolute lifetime target, not additional steps')
    p.add_argument('--start-timesteps', type=int, default=200001536)
    p.add_argument('--envs', type=int, default=14)
    p.add_argument('--n-steps', type=int, default=512)
    p.add_argument('--batch', type=int, default=1792)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--ent', type=float, default=0.005)
    p.add_argument('--checkpoint', type=int, default=5000000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=605)
    p.add_argument('--stage-copies', help='12 comma-separated extra stage counts per uniform cycle')
    p.add_argument('--natural-entry-fraction', type=float, default=0.70)
    p.add_argument('--hard-mid-probability', type=float, default=0.0)
    p.add_argument('--loss-mid-probability', type=float, default=0.0)
    a = p.parse_args()
    a.stage_copies = tuple(map(int, a.stage_copies.split(','))) if a.stage_copies else tuple(
        3 if s in WEAK_STAGES else 0 for s in range(1, 13))
    if len(a.stage_copies) != 12 or min(a.stage_copies) < 0:
        p.error('--stage-copies requires 12 nonnegative integers')
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from .fastppo import AsyncPPO
    from .policy import POLICY_KWARGS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert a.n_steps*a.envs % a.batch == 0
    torch.set_num_threads(2)
    venv = SubprocVecEnv([make_env(i, a) for i in range(a.envs)], start_method='spawn')
    model = AsyncPPO.load(a.resume, env=venv, device=a.device,
                          custom_objects={'policy_kwargs': POLICY_KWARGS},
                          learning_rate=a.lr, ent_coef=a.ent, target_kl=0.01,
                          clip_range=0.1, n_steps=a.n_steps, batch_size=a.batch, n_epochs=a.epochs,
                          tensorboard_log=str(out/'tb'), seed=a.seed)
    model.coalesce = 0.002
    config = dict(vars(a), jitters=JITTERS, jitter_weights=JITTER_WEIGHTS,
                  weak_stages=WEAK_STAGES, natural_probability=0.60,
                  online_pool_probability=0.15,
                  effective_stage_counts=tuple(1+n for n in a.stage_copies),
                  reward_changed=False, complete_snapshot_restore=True,
                  actual_lr=model.lr_schedule(1), actual_entropy=model.ent_coef,
                  actual_clip=model.clip_range(1), loaded_timesteps=model.num_timesteps,
                  resume_sha256=hashlib.sha256(Path(a.resume).read_bytes()).hexdigest(),
                  bank_manifest_sha256=hashlib.sha256((Path(a.bank)/'manifest.json').read_bytes()).hexdigest())
    (out/('config_%010d.json' % model.num_timesteps)).write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.log = (out/'matches.jsonl').open('a')
            self.start, self.t0 = model.num_timesteps, time.time()
            self.next_checkpoint = ((self.start-a.start_timesteps)//a.checkpoint+1)*a.checkpoint+a.start_timesteps
            self.recent = collections.defaultdict(lambda: collections.deque(maxlen=100))

        def _on_step(self):
            for info in self.locals['infos']:
                if 'match' in info:
                    m = dict(info['match'], t=self.num_timesteps)
                    self.log.write(json.dumps(m)+'\n')
                    self.recent[m['stage']].append(m['win'])
            if self.num_timesteps >= self.next_checkpoint:
                self.model.save(out/('ckpt_%010d' % self.num_timesteps))
                self.next_checkpoint += a.checkpoint
            return True

        def _on_rollout_end(self):
            self.log.flush()
            progress = dict(timesteps=self.num_timesteps, added=self.num_timesteps-a.start_timesteps,
                            target=a.total, seconds=time.time()-self.t0,
                            decisions_per_second=(self.num_timesteps-self.start)/max(1, time.time()-self.t0),
                            training_win_rates={s: float(np.mean(v)) for s, v in self.recent.items()},
                            pid=os.getpid(), updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            tmp = out/'progress.tmp'
            tmp.write_text(json.dumps(progress, indent=2))
            tmp.replace(out/'progress.json')
            for s, value in progress['training_win_rates'].items():
                self.logger.record('win/stage%02d' % s, value)
            self.logger.record('time/added_decisions', progress['added'])

    callback = Progress()
    try:
        remaining = a.total-model.num_timesteps
        if remaining > 0:
            model.learn(remaining, callback=callback, reset_num_timesteps=False)
    finally:
        model.save(out/'final')
        callback.log.close()
        venv.close()
    # A completion marker must never expose the initial or partially written final.zip.
    (out/'DONE').write_text(str(model.num_timesteps))


if __name__ == '__main__':
    main()
