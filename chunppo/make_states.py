"""Re-save the 12 stage states at the first frame the player can act (lock byte 0x96B2
goes 255 -> 0 after the round intro). Originals in generated_states/ are untouched;
new files go to generated_states/controllable/Stage%02d.<Name>.state (gzip).

  .venv\Scripts\python.exe -m chunppo.make_states
"""
import argparse
import gzip
import json
from pathlib import Path

from chunppo.common import CONTROLLABLE_DIR, ENEMY_IDS, FULL_HP, P1_X, ROSTER, stage_state, word
from chunppo.env import Runtime


def first_control(rt, blob):
    """Advance frame by frame (full reads) to the first controllable frame."""
    rt.set_state(blob)
    rt.ram = rt.game.get_ram()
    seen = not rt.controllable()
    for t in range(1, 601):
        rt.neutral(1)
        if not rt.controllable():
            seen = True
        elif seen:
            return t
    raise RuntimeError('intro did not end')


def verify_opening(rt, state, stage):
    rt.set_state(state)
    rt.ram = rt.game.get_ram()
    hp, enemy, x0 = rt.hp(), rt.enemy_id(), word(rt.ram, P1_X)
    controllable = rt.controllable()
    for _ in rt.macro(2):
        pass
    score = rt.score()
    moved = word(rt.ram, P1_X) != x0
    row = dict(stage=stage, opponent=ROSTER[stage - 1], hp=hp, score=score,
               controllable=controllable, forward_moves=moved, enemy_id=enemy, bytes=len(state))
    if not (controllable and moved and hp == (FULL_HP, FULL_HP) and score == (0, 0)
            and enemy == ENEMY_IDS[stage - 1]):
        raise RuntimeError(f"Invalid opening: {row}")
    return row


def save_openings(states, out):
    rt = Runtime(1)
    try:
        rows = {stage: verify_opening(rt, state, stage) for stage, state in states.items()}
    finally:
        rt.game.close()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    existing = json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else []
    manifest = {row['stage']: row for row in existing}
    for stage, state in states.items():
        path = out / ('Stage%02d.%s.state' % (stage, ROSTER[stage - 1]))
        path.write_bytes(gzip.compress(state, mtime=0))
        manifest[stage] = rows[stage]
        print(json.dumps(rows[stage]), flush=True)
    ordered = [manifest[stage] for stage in sorted(manifest)]
    manifest_path.write_text(json.dumps(ordered, indent=1), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', type=int, nargs='+', choices=range(1, 13), default=range(1, 13))
    parser.add_argument('--out', type=Path, default=CONTROLLABLE_DIR)
    args = parser.parse_args()
    rt, states = Runtime(1), {}
    try:
        for stage in args.stages:
            frames = first_control(rt, stage_state(stage))
            states[stage] = rt.game.em.get_state()
            print(f'Stage {stage:02d}: unlocked after {frames} frames', flush=True)
    finally:
        rt.game.close()
    save_openings(states, args.out)


if __name__ == '__main__':
    main()
