"""Shared constants, RAM helpers, Chun-Li action table and private input FIFO.

Design goals (see PLAN.md):
  * native emulator speed: no semantic extraction, no per-frame hashing/logging;
  * Ultimate-style 4-frame decisions with a Chun-Li specific action table in
    which charge moves are reachable (hold down-back across decisions, release);
  * actions are facing-relative (FWD/BACK resolved from RAM x at macro start).
"""
from __future__ import annotations

import collections
import gzip
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAME = 'StreetFighterIISpecialChampionEdition-Genesis'
STATE_DIR = ROOT / 'generated_states'
CONTROLLABLE_DIR = STATE_DIR / 'controllable'   # re-saved at first controllable frame (make_states.py)
ROSTER = ('ChunLi', 'Zangief', 'Dhalsim', 'Ryu', 'EHonda', 'Blanka', 'Guile', 'Ken',
          'Balrog', 'Vega', 'Sagat', 'Bison')
ENEMY_IDS = (5, 6, 7, 0, 1, 2, 3, 4, 10, 11, 9, 8)
FULL_HP = 176
CONTROL_WAIT_LIMIT = 600   # max neutral frames to wait for the round intro to end

# Genesis 12-button order used by gym-retro.
BUTTONS = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')
IDX = {b: i for i, b in enumerate(BUTTONS)}

# RAM offsets in the 64 KiB work RAM (68k big-endian; gym-retro exposes it byte-swapped).
P1_X, P2_X = 0x8006, 0x8286
P1_Y, P2_Y = 0x800A, 0x828A
P1_HP, P2_HP = 0x8042, 0x82C2
ROUND_STATE = 0x81DB       # 5 == fight live (asserted at stage start)
ENEMY_ID = 0x845B
BONUS = 0x9727             # non-zero during bonus stages
LOCK = 0x96B2              # 255 during each round intro (inputs ignored), 0 once the player has control.
                           # Found 2026-11-01: flips exactly at first effective input on stages 1/4/8/11
                           # (stage 1: 52 frames after the old 96-frame intro, others ~89); also 255
                           # for ~180 frames at the start of later rounds.


def byte(raw: np.ndarray, off: int) -> int:
    return int(raw[off ^ 1])


def word(raw: np.ndarray, off: int) -> int:
    n = int(raw[off ^ 1]) * 256 + int(raw[(off + 1) ^ 1])
    return n - 65536 if n >= 32768 else n


def stage_state(stage: int, controllable: bool = False) -> bytes:
    """Stage is 1-based (1=Chun-Li mirror ... 4=Ryu, 8=Ken, 12=Bison).
    controllable=True loads the re-saved state whose first frame already accepts input."""
    folder = CONTROLLABLE_DIR if controllable else STATE_DIR
    return gzip.open(folder / ('Stage%02d.%s.state' % (stage, ROSTER[stage - 1])), 'rb').read()


def make_retro():
    import retro
    retro.data.add_custom_integration(str(ROOT / 'retro_data'))
    return retro.make(game=GAME, state=retro.State.NONE, record=False,
                      use_restricted_actions=retro.Actions.ALL)


# --------------------------------------------------------------------------- actions
# Each action = 4 physical frames; each frame = tuple of logical keys.
# FWD/BACK are resolved against the opponent side at the start of the macro.
# Attack buttons are pressed 3 frames and released on the 4th so that choosing
# the same attack twice in a row produces two presses (Ultimate held all 4,
# which makes repeated presses / Lightning Legs mashing impossible).
PUNCH = ('X', 'Y', 'Z')    # light, medium, heavy punch
KICK = ('A', 'B', 'C')     # light, medium, heavy kick


def _hold(*keys):
    return [tuple(keys)] * 4


def _press(*keys, held=()):
    return [tuple(keys)] * 3 + [tuple(held)]


def _seq(direction, button):
    return [(direction,), (direction, button), (direction, button), (direction,)]


def build_actions():
    acts, names = [], []

    def add(name, frames):
        names.append(name)
        acts.append(frames)

    add('neutral', _hold())
    for name, keys in (('back', ('BACK',)), ('fwd', ('FWD',)), ('up', ('UP',)), ('down', ('DOWN',)),
                       ('down_back', ('DOWN', 'BACK')), ('down_fwd', ('DOWN', 'FWD')),
                       ('up_back', ('UP', 'BACK')), ('up_fwd', ('UP', 'FWD'))):
        add(name, _hold(*keys))
    for b in PUNCH + KICK:
        add('st_' + b, _press(b))
    # crouching attacks keep DOWN held on the release frame (keeps down-charge alive)
    for b in PUNCH + KICK:
        add('cr_' + b, _press('DOWN', b, held=('DOWN',)))
    # Releases are "direction first, then direction+button" (E2 probe 2026-11-01:
    # pressing UP and the kick on the same frame cancels the jump and gives a
    # standing kick; UP one frame earlier works).
    # Spinning Bird Kick VERIFIED: >=75 frames down(-back) charge (~19 decisions,
    # 60 was not enough) then up_B/up_C. Uncharged, up_B is a jump + early kick.
    # Chun-Li has no projectile in this version (no Kikoken); fwd_* are forward
    # attacks / throws (FWD+Z or FWD+C at close range).
    add('fwd_Y', _seq('FWD', 'Y'))
    add('fwd_Z', _seq('FWD', 'Z'))
    add('fwd_C', _seq('FWD', 'C'))
    add('up_B', _seq('UP', 'B'))
    add('up_C', _seq('UP', 'C'))
    # Lightning Legs: two distinct kick presses inside one decision
    add('mash_B', [('B',), (), ('B',), ()])
    return acts, names


ACTIONS, ACTION_NAMES = build_actions()
N_ACTIONS = len(ACTIONS)
NEUTRAL = 0


def resolve(frame_keys, opponent_right: bool) -> np.ndarray:
    out = np.zeros(12, np.int8)
    for k in frame_keys:
        if k == 'FWD':
            k = 'RIGHT' if opponent_right else 'LEFT'
        elif k == 'BACK':
            k = 'LEFT' if opponent_right else 'RIGHT'
        out[IDX[k]] = 1
    return out


class FIFO:
    """Private ordered input delay: each physical frame, with prob `p` the
    packet is delayed 1..5 frames; later packets never overtake earlier ones.
    Identical semantics to the project's historical private FIFO.

    `base` adds a fixed capture->execution delay of that many frames to every
    packet (non-blocking deployment latency, see PLAN.md / realtime.py): the
    request issued on frame t takes effect on frame t+base; until the first
    release the previous input (neutral after reset) keeps being applied."""

    def __init__(self, seed: int, p: float = 0.0, base: int = 0):
        self.rng = np.random.RandomState(seed & 0x7FFFFFFF)
        self.p = float(p)
        self.base = int(base)
        self.reset()

    def reset(self):
        self.pending = collections.deque()
        self.release = 0
        self.current = np.zeros(12, np.int8)
        self.tick = 0

    def __call__(self, request: np.ndarray) -> np.ndarray:
        delay = int(self.rng.randint(1, 6)) if (self.p > 0 and self.rng.random_sample() < self.p) else 0
        self.release = max(self.tick + self.base + delay, self.release)
        self.pending.append((self.release, request))
        while self.pending and self.pending[0][0] <= self.tick:
            _, self.current = self.pending.popleft()
        self.tick += 1
        return self.current
