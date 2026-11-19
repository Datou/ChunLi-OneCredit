"""PPO trainer (SB3 1.8 / gym 0.21) for the Chun-Li environment.

Example (full run, after the gomoku job frees the machine):
  .venv\\Scripts\\python.exe -m chunppo.train --out runs/chunppo/E3_main --envs 14 --total 200000000

Smoke:
  .venv\\Scripts\\python.exe -m chunppo.train --out runs/chunppo/smoke --envs 2 --n-steps 64 --batch 64 --total 512 --device cpu
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def make_env(rank, args):
    def _init():
        import cv2
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        if args.worker_priority == 'below' and os.name == 'nt':
            # with async collection the workers never idle; keep them from starving the
            # learner/inference process (which every worker is waiting on)
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
        from chunppo.env import ChunEnv
        stages = tuple(int(s) for s in args.stages.split(','))
        return ChunEnv(seed=args.seed * 1000 + rank, stages=stages, max_offset=args.max_offset,
                       pool_prob=0.0, snapshot_every=args.snapshot_every, jitter=0.0,
                       hist_len=args.hist_len, time_penalty=args.time_penalty, delay=args.delay)
    return _init


def schedule(frac, start, end, v0, v1):
    """Linear ramp of a value from v0 (before `start`) to v1 (after `end`), frac in [0,1]."""
    if frac <= start:
        return v0
    if frac >= end:
        return v1
    return v0 + (v1 - v0) * (frac - start) / (end - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--envs', type=int, default=14)
    p.add_argument('--stages', default='1,2,3,4,5,6,7,8,9,10,11,12')
    p.add_argument('--total', type=int, default=200_000_000, help='decisions (SB3 timesteps)')
    p.add_argument('--n-steps', type=int, default=512)
    p.add_argument('--batch', type=int, default=1792)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--ent', type=float, default=0.01)
    p.add_argument('--target-kl', type=float, default=0.02)
    p.add_argument('--hist-len', type=int, default=24)
    p.add_argument('--delay', type=int, default=0, help='fixed capture->execution delay in frames (FIFO base)')
    p.add_argument('--time-penalty', type=float, default=0.0005)
    p.add_argument('--max-offset', type=int, default=30)
    p.add_argument('--snapshot-every', type=int, default=300)
    # anti-memorisation / robustness curricula, as fractions of --total
    p.add_argument('--pool-prob', type=float, default=0.3, help='final snapshot-pool reset probability')
    p.add_argument('--pool-start', type=float, default=0.05)
    p.add_argument('--jitter', type=float, default=0.08, help='final private FIFO probability (target 0.05 + margin)')
    p.add_argument('--jitter-start', type=float, default=0.4)
    p.add_argument('--jitter-end', type=float, default=0.8)
    p.add_argument('--checkpoint', type=int, default=5_000_000)
    p.add_argument('--collector', choices=('async', 'sync'), default='async',
                   help='async = per-env rollout collection + uint8 GPU buffer (chunppo/fastppo.py); sync = plain SB3')
    p.add_argument('--worker-priority', choices=('below', 'normal'), default='below')
    p.add_argument('--coalesce', type=float, default=0.002,
                   help='async: wait up to this many seconds for half the in-flight envs before a forward')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume')
    args = p.parse_args()

    from stable_baselines3 import PPO
    from chunppo.fastppo import AsyncPPO
    from stable_baselines3.common.callbacks import BaseCallback
    from chunppo.policy import POLICY_KWARGS
    from stable_baselines3.common.vec_env import SubprocVecEnv

    class Curriculum(BaseCallback):
        """Sets pool/jitter knobs per rollout and logs per-stage match outcomes."""

        def __init__(self, out: Path):
            super().__init__()
            self.out = out
            self.log = open(out / 'matches.jsonl', 'a', encoding='utf8')
            self.recent = collections.defaultdict(lambda: collections.deque(maxlen=50))
            self.t0 = time.time()
            self.last_ckpt = self.t_start = None

        def _on_rollout_start(self):
            if self.t_start is None:                      # first rollout (also after --resume)
                self.t_start = self.last_ckpt = self.num_timesteps = self.model.num_timesteps
            frac = self.num_timesteps / args.total
            jit = schedule(frac, args.jitter_start, args.jitter_end, 0.0, args.jitter)
            pool = args.pool_prob if frac >= args.pool_start else 0.0
            self.training_env.env_method('set_jitter', jit)
            self.training_env.env_method('set_pool_prob', pool)
            self.logger.record('curriculum/jitter', jit)
            self.logger.record('curriculum/pool_prob', pool)

        def _on_step(self):
            for info in self.locals['infos']:
                m = info.get('match')
                if m is None:
                    continue
                m['t'] = self.num_timesteps
                self.log.write(json.dumps(m) + '\n')
                if not m['from_pool']:
                    self.recent[m['stage']].append(m['win'])
            if self.num_timesteps - self.last_ckpt >= args.checkpoint:
                self.last_ckpt = self.num_timesteps
                self.model.save(self.out / ('ckpt_%010d' % self.num_timesteps))
            return True

        def _on_rollout_end(self):
            self.log.flush()
            rates = {s: float(np.mean(v)) for s, v in sorted(self.recent.items()) if v}
            for s, r in rates.items():
                self.logger.record('win/stage%02d' % s, r)
            if rates:
                self.logger.record('win/mean_fresh', float(np.mean(list(rates.values()))))
            self.logger.record('time/decisions_per_s',
                               (self.num_timesteps - self.t_start) / max(1e-9, time.time() - self.t0))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    cfg = 'config.json' if not args.resume else 'config_resume_%s.json' % time.strftime('%Y%m%d_%H%M%S')
    (out / cfg).write_text(json.dumps(vars(args), indent=2), encoding='utf8')
    torch.set_num_threads(2)
    venv = SubprocVecEnv([make_env(i, args) for i in range(args.envs)], start_method='spawn')
    rollout = args.n_steps * args.envs
    assert rollout % args.batch == 0, 'batch must divide n_steps*envs (%d)' % rollout
    algo = AsyncPPO if args.collector == 'async' else PPO
    if args.resume:
        model = algo.load(args.resume, env=venv, device=args.device,
                         custom_objects={'policy_kwargs': POLICY_KWARGS})
    else:
        model = algo('MultiInputPolicy', venv, n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs,
                    learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=0.2,
                    ent_coef=args.ent, vf_coef=0.5, max_grad_norm=0.5, target_kl=args.target_kl,
                    policy_kwargs=POLICY_KWARGS,
                    tensorboard_log=str(out / 'tb'), device=args.device, seed=args.seed, verbose=1)
    if isinstance(model, AsyncPPO):
        model.coalesce = args.coalesce
    # SB3 adds `total_timesteps` on top of the loaded counter when not resetting, so pass the remainder
    remaining = args.total - model.num_timesteps if args.resume else args.total
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=Curriculum(out), reset_num_timesteps=not args.resume)
        # completion marker for supervise.py / evaluators: final.zip alone is also written on a crash
        (out / 'DONE').write_text(str(model.num_timesteps), encoding='utf8')
    finally:
        model.save(out / 'final')
        venv.close()


if __name__ == '__main__':
    main()
