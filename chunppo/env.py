"""Chun-Li PPO environment: Ultimate recipe + start-state diversity + jitter schedule.

Anti-memorisation without corrupting credit assignment (PLAN.md section 1):
  * balanced stage cycling over all 12 opponents;
  * random start offset after the intro, filled with a random walk (shifts the
    game's RNG phase *and* the starting geometry);
  * optional snapshot pool: mid-fight emulator states harvested from the
    worker's own rollouts become extra reset points (low HP, corners, round 2/3);
  * execution jitter (private FIFO) is a separate knob set by the trainer's
    curriculum via `set_jitter`, never observed by the policy.
"""
from __future__ import annotations

import collections
import random
import zlib

import cv2
import gym
import numpy as np

from .common import (ACTIONS, BONUS, CONTROL_WAIT_LIMIT, ENEMY_ID, FIFO, FULL_HP, LOCK, N_ACTIONS, NEUTRAL,
                     P1_HP, P1_X, P2_HP, P2_X, ROUND_STATE, byte, make_retro, resolve, stage_state, word)

SETTLE_LIMIT = 4000          # neutral frames allowed between rounds before truncation
WALK = (0, 1, 2, 4)          # neutral/back/fwd/down used for the random-walk start


class Observer:
    """Ultimate's motion-foreground 84x84 observation (R=t-16, G=t-8, B=t at 4-frame
    decisions) plus a one-hot history of the agent's own issued actions, which is
    what makes charge duration observable."""

    def __init__(self, hist_len: int = 24, size: int = 84):
        self.hist_len, self.size = hist_len, size

    def _process(self, frame):
        h, w = frame.shape[:2]
        cw, ch = min(256, w), min(200, h)
        left, top = (w - cw) // 2, (h - ch) // 2
        frame = frame[top:top + ch, left:left + cw]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.motion.append(gray)
        if len(self.motion) >= 5:
            first = cv2.absdiff(self.motion[-5], self.motion[-3])
            second = cv2.absdiff(self.motion[-3], self.motion[-1])
            motion = cv2.GaussianBlur(np.maximum(first, second), (5, 5), 0)
            _, mask = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            frame = np.where(mask[..., None] > 0, frame, frame * 0.35).astype(np.uint8)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

    def reset(self, screen):
        self.motion = collections.deque(maxlen=5)
        first = self._process(screen)
        self.stack = collections.deque([first] * 5, maxlen=5)
        self.hist = collections.deque([-1] * self.hist_len, maxlen=self.hist_len)

    def push(self, screen, action):
        self.stack.append(self._process(screen))
        self.hist.append(action)

    def obs(self):
        s = self.stack
        image = np.stack([s[0][..., 0], s[2][..., 1], s[4][..., 2]], axis=0)
        hist = np.zeros((self.hist_len, N_ACTIONS), np.float32)
        for i, a in enumerate(self.hist):
            if a >= 0:
                hist[i, a] = 1.0
        return {'image': image, 'hist': hist.reshape(-1)}


class Runtime:
    """Thin wrapper over one gym-retro instance; every physical frame passes the FIFO."""

    def __init__(self, seed: int, jitter: float = 0.0, delay: int = 0):
        self.game = make_retro()
        self.game.reset()
        self.fifo = FIFO(seed, jitter, base=delay)
        self._stage_cache = {}
        self.screen = self.ram = self.info = None
        self.total_frames = 0

    def frame(self, request, full: bool = True):
        """Advance one physical frame. `full=False` skips screen/RAM/info refresh
        (pure emulation; ~2.5x cheaper) -- used for all but the last frame of a
        decision, whose state is all the policy and reward ever see."""
        applied = self.fifo(request)
        self.total_frames += 1
        if full:
            self.screen, _, _, self.info = self.game.step(applied)
            self.ram = self.game.get_ram()
        else:
            self.game.em.set_button_mask(applied.astype(np.uint8), 0)
            self.game.em.step()
        return self.screen

    def neutral(self, n=1):
        z = np.zeros(12, np.int8)
        for i in range(n):
            self.frame(z, full=(i == n - 1))

    def hp(self):
        return word(self.ram, P1_HP), word(self.ram, P2_HP)

    def score(self):
        return int(self.info['matches_won']), int(self.info['enemy_matches_won'])

    def opponent_right(self):
        return word(self.ram, P2_X) >= word(self.ram, P1_X)

    def round_state(self):
        return byte(self.ram, ROUND_STATE)

    def enemy_id(self):
        return byte(self.ram, ENEMY_ID)

    def bonus(self):
        return byte(self.ram, BONUS)

    def controllable(self):
        return byte(self.ram, LOCK) == 0

    def wait_control(self):
        """Run neutral frames through a round intro: the lock byte is 0 before the
        intro starts, 255 during it and 0 again once inputs register."""
        seen = False
        for _ in range(CONTROL_WAIT_LIMIT // 2):
            if not self.controllable():
                seen = True
            elif seen:
                return True
            self.neutral(2)
        return False

    def macro(self, action, every_frame: bool = False):
        """Execute one 4-frame action (facing fixed at macro start). Yields after
        each frame; RAM/screen are fresh only on the last frame unless every_frame."""
        right = self.opponent_right()
        frames = ACTIONS[action]
        for i, keys in enumerate(frames):
            yield self.frame(resolve(keys, right), full=every_frame or i == len(frames) - 1)

    def set_state(self, blob: bytes):
        self.game.em.set_state(blob)
        self.game.data.reset()
        self.game.data.update_ram()
        self.fifo.reset()

    def load_stage(self, stage: int, offset: int = 0, rng: random.Random | None = None):
        """Load the stage state re-saved at the first controllable frame
        (generated_states/controllable, see make_states.py), then the random walk."""
        if stage not in self._stage_cache:
            self._stage_cache[stage] = stage_state(stage, controllable=True)
        self.set_state(self._stage_cache[stage])
        self.neutral(1)
        if not self.controllable():
            raise RuntimeError('stage %d: controllable state is not controllable' % stage)
        # random walk during the offset: shifts RNG phase and starting spacing
        a, right = NEUTRAL, self.opponent_right()
        for t in range(offset):
            if t % 4 == 0:
                right = self.opponent_right()
                if rng is not None:
                    a = rng.choice(WALK)
            self.frame(resolve(ACTIONS[a][t % 4], right), full=(t % 4 == 3 or t == offset - 1))


class MatchTracker:
    """Ultimate reward: 0.01*(3*dealt - taken) per decision, round end bonus
    +(176+own_hp) on win / -(176+enemy_hp) on loss, small time penalty."""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self.score = rt.score()
        self.prev_hp = rt.hp()
        self.live = min(self.prev_hp) >= 0
        self.ko_hp = self.prev_hp
        self.dealt = self.taken = 0

    def update(self):
        """Call once per decision (after the macro's last, fully-read frame)."""
        hp, score = self.rt.hp(), self.rt.score()
        raw = 0.0
        if self.live:
            d = max(0, max(self.prev_hp[1], 0) - max(hp[1], 0))
            t = max(0, max(self.prev_hp[0], 0) - max(hp[0], 0))
            self.dealt += d
            self.taken += t
            raw += 3.0 * d - t
            # KO is HP < 0 in this ROM (HP == 0 is still alive; verified by RAM poke)
            if hp[0] < 0 or hp[1] < 0 or score != self.score:
                self.live = False
                self.ko_hp = hp
        self.prev_hp = hp
        return raw

    def settle(self):
        """Called after a decision ends a round: run neutral frames until the score
        updates and either the match is over or the next round is live.
        Returns (raw_reward, match_over, truncated)."""
        raw, rt = 0.0, self.rt
        start, seen_lock = self.score, False
        for _ in range(SETTLE_LIMIT // 4):
            score = rt.score()
            if score != start:
                if max(score) >= 2:
                    break
                if not rt.controllable():
                    seen_lock = True
                elif seen_lock and min(rt.hp()) > 100:
                    break
            rt.neutral(4)
        else:
            return raw, False, True
        score = rt.score()
        if score[0] > start[0]:
            raw += FULL_HP + max(self.ko_hp[0], 0)
        if score[1] > start[1]:
            raw -= FULL_HP + max(self.ko_hp[1], 0)
        self.score = score
        self.prev_hp = rt.hp()
        self.live = True
        return raw, max(score) >= 2, False


class ChunEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, seed: int = 0, stages=tuple(range(1, 13)), max_offset: int = 30,
                 pool_prob: float = 0.0, snapshot_every: int = 300, pool_size: int = 16,
                 jitter: float = 0.0, hist_len: int = 24, time_penalty: float = 0.0005,
                 max_decisions: int = 8000, delay: int = 0):
        super().__init__()
        self.rng = random.Random(seed)
        self.rt = Runtime(seed * 7919 + 1, jitter, delay)
        self.observer = Observer(hist_len)
        self.stages = tuple(stages)
        self.max_offset, self.pool_prob = max_offset, pool_prob
        self.snapshot_p, self.time_penalty = 1.0 / snapshot_every, time_penalty
        self.max_decisions = max_decisions
        self.pool = {s: collections.deque(maxlen=pool_size) for s in self.stages}
        self._cycle = []
        self._forced = None
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (3, 84, 84), np.uint8),
            'hist': gym.spaces.Box(0.0, 1.0, (hist_len * N_ACTIONS,), np.float32)})
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    # ---- knobs used by trainer / evaluator
    def set_jitter(self, p: float):
        self.rt.fifo.p = float(p)

    def set_pool_prob(self, p: float):
        self.pool_prob = float(p)

    def force_next(self, stage: int, offset: int):
        """Evaluation: next reset uses this exact fixture (no pool)."""
        self._forced = (stage, offset)

    def _next_stage(self):
        if not self._cycle:
            self._cycle = list(self.stages)
            self.rng.shuffle(self._cycle)
        return self._cycle.pop()

    def reset(self):
        self.from_pool = False
        if self._forced is not None:
            (self.stage, self.offset), self._forced = self._forced, None
            self.rt.load_stage(self.stage, self.offset, random.Random(self.offset * 1000003 + self.stage))
        else:
            self.stage = self._next_stage()
            pool = self.pool[self.stage]
            if pool and self.rng.random() < self.pool_prob:
                self.from_pool, self.offset = True, -1
                self.rt.set_state(zlib.decompress(self.rng.choice(pool)))
                self.rt.neutral(1)
            else:
                self.offset = self.rng.randint(0, self.max_offset)
                self.rt.load_stage(self.stage, self.offset, self.rng)
        self.tracker = MatchTracker(self.rt)
        self.observer.reset(self.rt.screen)
        self.decisions = 0
        return self.observer.obs()

    def step(self, action):
        action = int(action)
        raw = 0.0
        tr = self.tracker
        for _ in self.rt.macro(action):
            pass
        raw += tr.update()
        done = truncated = False
        if not tr.live:
            r2, done, truncated = tr.settle()
            raw += r2
        elif self.pool_prob > 0 and not self.from_pool and self.rng.random() < self.snapshot_p:
            self.pool[self.stage].append(zlib.compress(self.rt.game.em.get_state(), 1))
        self.decisions += 1
        if self.decisions >= self.max_decisions and not done:
            truncated = True
        self.observer.push(self.rt.screen, action)
        reward = 0.01 * raw - self.time_penalty
        info = {}
        if done or truncated:
            info['TimeLimit.truncated'] = truncated and not done
            info['match'] = dict(stage=self.stage, win=bool(tr.score[0] >= 2), score=list(tr.score),
                                 dealt=tr.dealt, taken=tr.taken, from_pool=self.from_pool,
                                 offset=self.offset, jitter=self.rt.fifo.p,
                                 decisions=self.decisions, truncated=truncated)
        return self.observer.obs(), reward, bool(done or truncated), info

    def close(self):
        self.rt.game.close()
