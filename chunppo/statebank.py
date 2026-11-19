"""Complete training reset states: emulator, observation, tracker and private FIFO."""
from __future__ import annotations

import copy
import pickle
import zlib
from pathlib import Path

from .env import MatchTracker


def capture(rt, observer, tracker):
    fifo = rt.fifo
    return dict(emulator=rt.game.em.get_state(), screen=rt.screen.copy(), ram=rt.ram.copy(),
                info=copy.deepcopy(rt.info), total_frames=rt.total_frames,
                observer=copy.deepcopy(observer.__dict__),
                tracker={k: copy.deepcopy(v) for k, v in tracker.__dict__.items() if k != 'rt'},
                fifo=dict(pending=copy.deepcopy(fifo.pending), release=fifo.release,
                          current=fifo.current.copy(), tick=fifo.tick,
                          rng=fifo.rng.get_state(), base=fifo.base, p=fifo.p))


def restore(rt, observer, state, jitter=None, restore_rng=True):
    rt.set_state(state['emulator'])
    rt.screen, rt.ram = state['screen'].copy(), state['ram'].copy()
    rt.info, rt.total_frames = copy.deepcopy(state['info']), state['total_frames']
    observer.__dict__.clear()
    observer.__dict__.update(copy.deepcopy(state['observer']))
    f = state['fifo']
    rt.fifo.pending, rt.fifo.release = copy.deepcopy(f['pending']), f['release']
    rt.fifo.current, rt.fifo.tick = f['current'].copy(), f['tick']
    rt.fifo.base, rt.fifo.p = f['base'], f['p'] if jitter is None else float(jitter)
    if restore_rng:
        rt.fifo.rng.set_state(f['rng'])
    tracker = MatchTracker(rt)
    tracker.__dict__.update(copy.deepcopy(state['tracker']))
    tracker.rt = rt
    return tracker


def encode(state):
    return zlib.compress(pickle.dumps(state, protocol=4), 1)


def decode(blob):
    # Only locally generated training artifacts are accepted by this workflow.
    return pickle.loads(zlib.decompress(blob))


def load(path):
    return decode(Path(path).read_bytes())


def align_latest_observation(state):
    """Replace a stale pre-settle last image with the current screen, retaining history."""
    import numpy as np
    from .env import Observer
    observer = Observer()
    observer.__dict__.update(copy.deepcopy(state['observer']))
    if observer.__dict__.get('pending') is not None:
        raise ValueError('Deferred observations must be committed before snapshot normalization')
    # Recompute the last push with the same four preceding motion frames.
    observer.motion.pop()
    current = observer._process(state['screen'])
    if np.array_equal(current, state['observer']['stack'][-1]):
        return state, False
    observer.stack[-1] = current
    normalized = dict(state, observer=observer.__dict__)
    return normalized, True
