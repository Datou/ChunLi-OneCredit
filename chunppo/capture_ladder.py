"""Capture first-actionable openings while a policy plays the natural ladder."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from . import evaluate
from .common import CONTROLLABLE_DIR, ENEMY_IDS, FULL_HP, stage_state
from .env import Runtime
from .make_states import save_openings, verify_opening

_verifier = None


def init_verifier():
    # Gym Retro supports one emulator per process.
    global _verifier
    _verifier = Runtime(1)


def is_opening(state, stage):
    try:
        verify_opening(_verifier, state, stage)
        return True
    except RuntimeError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, default=CONTROLLABLE_DIR)
    parser.add_argument('--seed', type=int, default=140223910)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--jitter', type=float, default=0.0)
    args = parser.parse_args()
    if args.offset < 0:
        parser.error('--offset must be nonnegative')

    if not 0 <= args.jitter <= 1:
        parser.error('--jitter must be between 0 and 1')

    states = {1: stage_state(1, controllable=True)}
    evaluate._init(args.model, dict(hist_len=24, delay=0, sync_round_observation=True))
    with ProcessPoolExecutor(1, mp_context=mp.get_context('spawn'), initializer=init_verifier) as pool:
        def capture(rt, enemy):
            stage = ENEMY_IDS.index(enemy) + 1
            if rt.hp() != (FULL_HP, FULL_HP):
                return False
            state = rt.game.em.get_state()
            if not pool.submit(is_opening, state, stage).result():
                return False
            states[stage] = state
            print(f'Captured stage {stage:02d}', flush=True)
            return True

        result = evaluate.campaign(dict(seed=args.seed, jitter=args.jitter, offset=args.offset, deterministic=True),
                                   on_match_start=capture)
    save_openings(states, args.out)
    if len(states) != len(ENEMY_IDS):
        parser.exit(1, f"Captured {len(states)}/12 openings ({result['reason']}). "
                       "Try another opening offset or capture missing opponents manually.\n")


if __name__ == '__main__':
    main()
