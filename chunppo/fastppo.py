"""PPO with a per-env asynchronous rollout collector and a uint8 GPU-resident rollout buffer.

Why (measured 2026-11-01, 14 envs, see PLAN.md section 2): with SB3's lock-step SubprocVecEnv
every vector step waits for the slowest worker. A between-round transition (KO -> score ->
next-round intro lock) is ~744 frames / ~314 ms in ONE worker while the other 13 idle; together
with the per-step GPU forward (workers idle) and the float32 rollout buffer (607 MB copied per
minibatch) the workers were idle ~2/3 of the wall time.

What changes:
  * `AsyncPPO.collect_rollouts`: each env advances independently -- as soon as a worker returns,
    its next action is computed (batched with whichever other workers are ready) and sent.
    Every env still fills its own buffer column with exactly `n_steps` consecutive transitions
    of its own trajectory, all under the same (frozen) policy, so GAE and the PPO objective are
    unchanged; the only wait is at the end of the rollout for the env with the slowest 512 steps.
  * `Uint8DictRolloutBuffer`: observations are stored in their space dtype (uint8 image) and the
    whole rollout is moved to the GPU once per update; minibatches are indexed on the GPU.
"""
from __future__ import annotations

import time
from multiprocessing.connection import wait

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv


class Uint8DictRolloutBuffer(DictRolloutBuffer):
    """DictRolloutBuffer that keeps observations in their native dtype and samples on-device."""

    def reset(self) -> None:
        self.observations = {k: np.zeros((self.buffer_size, self.n_envs, *shape), dtype=self.observation_space[k].dtype)
                             for k, shape in self.obs_shape.items()}
        for name in ('rewards', 'returns', 'episode_starts', 'values', 'log_probs', 'advantages'):
            setattr(self, name, np.zeros((self.buffer_size, self.n_envs), dtype=np.float32))
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.generator_ready = False
        self._dev = None
        self.pos, self.full = 0, False

    def get(self, batch_size=None):
        assert self.full, ''
        n = self.buffer_size * self.n_envs
        if not self.generator_ready:
            # same (env-major) flattening as SB3; then one host->device copy per rollout
            dev = {'obs': {k: th.as_tensor(self.swap_and_flatten(v), device=self.device)
                           for k, v in self.observations.items()}}
            for name in ('actions', 'values', 'log_probs', 'advantages', 'returns'):
                dev[name] = th.as_tensor(self.swap_and_flatten(getattr(self, name)), device=self.device)
            self._dev = dev
            self.generator_ready = True
        if batch_size is None:
            batch_size = n
        idx = th.randperm(n, device=self.device)
        for s in range(0, n, batch_size):
            yield self._get_samples(idx[s:s + batch_size])

    def _get_samples(self, idx, env=None):
        d = self._dev
        return DictRolloutBufferSamples(
            observations={k: v[idx] for k, v in d['obs'].items()},
            actions=d['actions'][idx],
            old_values=d['values'][idx].flatten(),
            old_log_prob=d['log_probs'][idx].flatten(),
            advantages=d['advantages'][idx].flatten(),
            returns=d['returns'][idx].flatten(),
        )


def _stack(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _row(obs, i):
    return {k: v[i] for k, v in obs.items()}


class AsyncPPO(PPO):
    """PPO whose rollouts are collected per env, without a lock-step barrier (see module doc).

    The main process is the scarce resource once workers never idle (each forward has a ~1.5-5 ms
    fixed cost), so `_act` is a lean version of `policy.forward` (one device sync instead of three,
    no torch.distributions), and `coalesce` > 0 waits up to that many seconds for at least half of
    the in-flight envs before running a forward."""

    rollout_deterministic = False          # tests only: argmax actions
    coalesce = 0.0

    def _setup_model(self) -> None:
        super()._setup_model()
        if isinstance(self.observation_space, spaces.Dict):
            self.rollout_buffer = Uint8DictRolloutBuffer(self.n_steps, self.observation_space, self.action_space,
                                                         device=self.device, gamma=self.gamma,
                                                         gae_lambda=self.gae_lambda, n_envs=self.n_envs)

    def _act(self, obs_list):
        """Batched forward on a list of per-env observations -> numpy actions, values, log_probs.
        Same computation as ActorCriticPolicy.forward with a Categorical head (SB3 1.8)."""
        pol = self.policy
        with th.inference_mode():
            obs = {k: th.from_numpy(v).to(self.device, non_blocking=True) for k, v in _stack(obs_list).items()}
            latent_pi, latent_vf = pol.mlp_extractor(pol.extract_features(obs))
            logp_all = th.log_softmax(pol.action_net(latent_pi), dim=-1)
            if self.rollout_deterministic:
                a = logp_all.argmax(dim=-1, keepdim=True)
            else:
                a = th.multinomial(logp_all.exp(), 1)
            out = th.cat([a.to(logp_all.dtype), pol.value_net(latent_vf), logp_all.gather(1, a)], dim=1).cpu().numpy()
        return out[:, 0].astype(np.int64), out[:, 1], out[:, 2]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        if not isinstance(env, SubprocVecEnv) or self.use_sde:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        assert self._last_obs is not None, 'No previous observation was provided'
        assert isinstance(self.action_space, spaces.Discrete)
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()          # curriculum env_method calls happen while all envs are idle

        n_envs, buf = env.num_envs, rollout_buffer
        remotes = env.remotes
        last_obs = [_row(self._last_obs, e) for e in range(n_envs)]
        last_start = np.array(self._last_episode_starts, dtype=np.float32).copy()
        count = np.zeros(n_envs, dtype=np.int64)
        inflight = {}                        # env -> (action, value, log_prob) of the step in progress
        by_conn = {id(r): e for e, r in enumerate(remotes)}

        def launch(envs):
            acts, vals, logps = self._act([last_obs[e] for e in envs])
            for i, e in enumerate(envs):
                inflight[e] = (acts[i], vals[i], logps[i])
                remotes[e].send(('step', acts[i]))

        launch(list(range(n_envs)))
        while inflight:
            pending = [remotes[e] for e in inflight]
            got = wait(pending)
            if self.coalesce > 0 and 2 * len(got) < len(pending):
                deadline = time.perf_counter() + self.coalesce
                while 2 * len(got) < len(pending):
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        break
                    done_ids = {id(c) for c in got}
                    got += wait([c for c in pending if id(c) not in done_ids], left)
            ready = [by_conn[id(c)] for c in got]
            results = [remotes[e].recv() for e in ready]
            infos = [r[3] for r in results]
            rewards = np.array([r[1] for r in results], dtype=np.float64)   # bootstrap in float64, as SB3
            dones = np.array([r[2] for r in results], dtype=bool)
            # timeout bootstrap, as in SB3 (GitHub issue #633)
            trunc = [i for i, (d, inf) in enumerate(zip(dones, infos))
                     if d and inf.get('terminal_observation') is not None and inf.get('TimeLimit.truncated', False)]
            if trunc:
                with th.no_grad():
                    tv = self.policy.predict_values(
                        obs_as_tensor(_stack([infos[i]['terminal_observation'] for i in trunc]), self.device))
                for j, i in enumerate(trunc):
                    rewards[i] += float(self.gamma * tv[j])           # float32 product, as SB3
            for i, e in enumerate(ready):
                a, v, lp = inflight.pop(e)
                t = count[e]
                for k, x in last_obs[e].items():
                    buf.observations[k][t, e] = x
                buf.actions[t, e] = a
                buf.rewards[t, e] = rewards[i]
                buf.episode_starts[t, e] = last_start[e]
                buf.values[t, e] = v
                buf.log_probs[t, e] = lp
                count[e] = t + 1
                last_obs[e] = results[i][0]
                last_start[e] = float(dones[i])

            self.num_timesteps += len(ready)
            callback.update_locals({'infos': infos, 'dones': dones, 'rewards': rewards, 'env_ids': ready})
            if callback.on_step() is False:
                for e in list(inflight):     # drain so the workers are idle before returning
                    remotes[e].recv()
                return False
            self._update_info_buffer(infos)

            nxt = [e for e in ready if count[e] < n_rollout_steps]
            if nxt:
                launch(nxt)

        buf.pos, buf.full = buf.buffer_size, True
        self._last_obs = _stack(last_obs)
        self._last_episode_starts = last_start.astype(bool)
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
        buf.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)
        callback.on_rollout_end()
        return True
