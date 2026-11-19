"""Real-time 60 Hz evaluator: the emulator runs on a wall clock, the policy in a separate process, non-blocking.

  .venv\\Scripts\\python.exe -m chunppo.realtime --model runs/chunppo/E3_ryu_ken/final.zip --out <dir>
      [--stages 4,8] [--fixtures 4] [--delay 0] [--jitter 0] [--pairs 4] [--device cpu] [--blocking]

Game process (the clock): frame f is emulated at wall time t0 + f/60 and never waits for the policy. On every
capture frame -- exactly the frames where ChunEnv returns an observation (every 4 frames in a fight, the first
live frame after a round transition) -- it sends the screen to the agent process and keeps running. The reply's
4 per-frame packets are put on a delay line at frames c+1+D .. c+4+D (D = --delay, the fixed capture->execution
delay of training's FIFO `base`). A packet that arrives after its frame is dropped and the previous input is
held -- the same semantics as the training FIFO. With every reply on time the game is identical to lockstep
ChunEnv(delay=D); `--blocking` waits for every reply and must reproduce it bit for bit (test_realtime.py).

Reset/round transitions/match end reuse Runtime.load_stage and MatchTracker; only Runtime.frame is replaced.

Per decision it records capture->reply latency against the frame deadlines, so one run answers "which D
would every decision have met" (summary: on_time_by_delay), plus the clock's own pacing error.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np

FPS = 60.0

if os.name == 'nt':
    # time.perf_counter() has a per-process origin on Windows/py3.8 (measured ~0.5 s apart between the game
    # and agent processes); QueryPerformanceCounter itself is system-wide, so cross-process stamps use it.
    import ctypes as _ct
    _QPF = _ct.c_int64()
    _ct.windll.kernel32.QueryPerformanceFrequency(_ct.byref(_QPF))
    _QPC = _ct.windll.kernel32.QueryPerformanceCounter

    def now():
        v = _ct.c_int64()
        _QPC(_ct.byref(v))
        return v.value / _QPF.value
else:
    now = time.perf_counter


def _realtime_process(priority=0x00008000):                 # ABOVE_NORMAL_PRIORITY_CLASS
    if os.name == 'nt':
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)              # 1 ms sleep granularity
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), priority)


def load_policy(model_path, device='cpu', deterministic=True):
    """Lean single-observation forward (same computation as AsyncPPO._act) -> callable obs -> int."""
    import torch as th
    from stable_baselines3 import PPO
    from chunppo.policy import POLICY_KWARGS
    th.set_num_threads(1)
    pol = PPO.load(model_path, device=device, custom_objects={'policy_kwargs': POLICY_KWARGS}).policy
    pol.set_training_mode(False)
    dev = th.device(device)

    def act(obs):
        with th.inference_mode():
            x = {k: th.from_numpy(np.ascontiguousarray(v)[None]).to(dev) for k, v in obs.items()}
            latent_pi, _ = pol.mlp_extractor(pol.extract_features(x))
            logits = pol.action_net(latent_pi)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            return int(th.multinomial(th.softmax(logits, dim=-1), 1).item())
    return act


def agent_main(conn, model_path, device, deterministic, hist_len):
    """Policy process: Observer + forward per capture; replies (id, action, t_recv, t_done)."""
    _realtime_process()
    from chunppo.env import Observer
    act = load_policy(model_path, device, deterministic)
    observer, last = Observer(hist_len), None
    # warm-up: the first forward pays lazy kernel/primitive init (~40 ms on CPU), keep it off the clock
    observer.reset(np.zeros((224, 320, 3), np.uint8))
    for _ in range(5):
        act(observer.obs())
    conn.send('ready')
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind, k, screen = msg
        t_recv = now()
        if kind == 'reset':
            observer.reset(screen)
        else:
            observer.push(screen, last)
        last = act(observer.obs())
        conn.send((k, last, t_recv, now()))


class Clock:
    """Wall-clock pacing: frame `tick` starts at t0 + (tick - tick0)/60 (sleep, then spin the last 1.5 ms)."""

    def __init__(self):
        self.late = []

    def start(self, tick0):
        self.t0, self.tick0 = now(), tick0

    def slot(self, tick):
        return self.t0 + (tick - self.tick0) / FPS

    def wait(self, tick):
        target = self.slot(tick)
        while True:
            left = target - now()
            if left <= 0:
                break
            if left > 0.002:
                time.sleep(left - 0.0015)
        self.late.append(-left)


class AgentLink:
    """Game-side end of the agent pipe: sends captures, schedules replies on the runtime's delay line."""

    def __init__(self, conn, blocking):
        self.conn, self.blocking = conn, blocking
        self.pending, self.next_id, self.records = {}, 0, []

    def capture(self, rt, kind):
        k, self.next_id = self.next_id, self.next_id + 1
        t_send = now()
        self.pending[k] = (rt.tick, rt.opponent_right(), t_send)
        self.conn.send((kind, k, rt.screen))
        if self.blocking:
            self._schedule(rt, self.conn.recv())

    def poll(self, rt):
        while self.conn.poll():
            self._schedule(rt, self.conn.recv())

    def drain(self):
        """Replies still in flight when the match ended: never executed, not recorded."""
        while self.pending:
            self.pending.pop(self.conn.recv()[0])

    def _schedule(self, rt, msg):
        from chunppo.common import ACTIONS, resolve
        k, action, t_recv, t_done = msg
        req0, right, t_send = self.pending.pop(k)
        dropped = 0
        for i, keys in enumerate(ACTIONS[action]):
            target = req0 + i + rt.D
            if target >= rt.tick:
                rt.sched[target] = resolve(keys, right)
            else:
                dropped += 1
        self.actions.append(action)
        clock = rt.clock
        self.records.append((req0, rt.tick, t_send, t_recv, t_done, now(), dropped,
                             clock.slot(req0) if clock else np.nan))

    def new_match(self):
        self.actions = []


def make_runtime(seed, jitter, delay, link):
    from chunppo.env import Runtime

    class RTRuntime(Runtime):
        """Runtime whose frame() is paced by `clock`, polls the agent, and applies an explicit D-frame delay
        line (FIFO base 0 + jitter on top), so a reply only has to arrive before its own frame."""

        def __init__(self):
            super().__init__(seed, jitter, 0)
            self.D, self.link, self.clock = int(delay), link, None
            self._reset_line()

        def _reset_line(self):
            self.tick, self.sched, self.held = 0, {}, np.zeros(12, np.int8)

        def set_state(self, blob):
            super().set_state(blob)
            self._reset_line()

        def frame(self, request, full=True):
            if self.clock is not None:
                self.clock.wait(self.tick)
                if not self.link.blocking:
                    self.link.poll(self)
            if request is not None:                       # issued on this tick (walk, settle neutral)
                self.sched[self.tick + self.D] = request
            pkt = self.sched.pop(self.tick, None)
            if pkt is not None:
                self.held = pkt
            applied = self.fifo(self.held)
            self.tick += 1
            self.total_frames += 1
            if full:
                self.screen, _, _, self.info = self.game.step(applied)
                self.ram = self.game.get_ram()
            else:
                self.game.em.set_button_mask(applied.astype(np.uint8), 0)
                self.game.em.step()
            return self.screen

    return RTRuntime()


def play_match(job, link, delay, jitter, blocking, max_decisions=8000):
    """One held-out fixture under real time; control flow mirrors ChunEnv.reset/step (pool/jitter off)."""
    from chunppo.env import MatchTracker
    clock = None if blocking else Clock()
    rt = make_runtime(job['seed'] * 7919 + 1, jitter, delay, link)
    try:
        rt.load_stage(job['stage'], job['offset'], random.Random(job['offset'] * 1000003 + job['stage']))
        tr = MatchTracker(rt)
        link.new_match()
        if clock is not None:
            clock.start(rt.tick - 1)                      # the capture frame occupies the current slot
            rt.clock = clock
        else:
            rt.clock = None
        link.capture(rt, 'reset')
        decisions, done, truncated = 0, False, False
        while True:
            for i in range(4):
                rt.frame(None, full=(i == 3))
            tr.update()
            if not tr.live:
                _, done, truncated = tr.settle()
            decisions += 1
            if decisions >= max_decisions and not done:
                truncated = True
            if done or truncated:
                break
            link.capture(rt, 'step')
        link.drain()
        ram_crc = zlib.crc32(rt.ram.tobytes())
        return dict(job, win=bool(tr.score[0] >= 2), score=list(tr.score), dealt=tr.dealt, taken=tr.taken,
                    decisions=decisions, truncated=truncated, frames=rt.tick, actions=list(link.actions),
                    ram_sum=ram_crc), (clock.late if clock else [])
    finally:
        rt.game.close()


def pair_main(jobs, args, out_q):
    """One game process + its own agent process; runs its share of the fixtures sequentially."""
    _realtime_process()
    import cv2
    cv2.setNumThreads(1)
    ctx = mp.get_context('spawn')
    a_conn, g_conn = ctx.Pipe()
    agent = ctx.Process(target=agent_main, args=(a_conn, args['model'], args['device'], not args['stochastic'],
                                                 args['hist_len']), daemon=True)
    agent.start()
    assert g_conn.recv() == 'ready'
    link = AgentLink(g_conn, args['blocking'])
    try:
        for job in jobs:
            link.records = []
            res, late = play_match(job, link, args['delay'], job['jitter'], args['blocking'])
            out_q.put((res, link.records, late))
    finally:
        g_conn.send(None)
        agent.join(10)


def make_jobs(stages, fixtures, jitters, seed):
    """Held-out fixtures with the same scheme as evaluate.py (offsets 31..90, independent seeds)."""
    rng, jobs = random.Random(seed), []
    for j in jitters:
        for stage in stages:
            for off in rng.sample(range(31, 91), fixtures):
                jobs.append(dict(stage=stage, offset=off, jitter=j, seed=rng.randrange(2 ** 30)))
    return jobs


def run(args, jobs):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    shares = [jobs[i::args['pairs']] for i in range(args['pairs'])]
    procs = [ctx.Process(target=pair_main, args=(s, args, q)) for s in shares if s]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(len(jobs))]
    for p in procs:
        p.join()
    return results


def summarize(results, delay):
    from chunppo.evaluate import wilson
    recs = np.array([r for _, rs, _ in results for r in rs], dtype=np.float64)
    late = np.array([x for _, _, l in results for x in l], dtype=np.float64)
    out = dict(matches=len(results), decisions=int(len(recs)))
    if len(recs) and np.isfinite(recs[:, 7]).any():
        req0, arr_tick, t_send, t_recv, t_done, t_arr, dropped, slot0 = recs.T
        ms = lambda x: {q: round(float(np.percentile(x, p)) * 1e3, 2)
                        for q, p in (('p50', 50), ('p90', 90), ('p99', 99), ('max', 100))}
        frames = (t_done - (slot0 - 1 / FPS)) * FPS            # reply time, in frames after the capture slot
        out['ms'] = dict(to_agent=ms(t_recv - t_send), agent_compute=ms(t_done - t_recv),
                         capture_to_reply=ms(t_done - t_send))
        out['reply_frames_after_capture'] = {q: round(float(np.percentile(frames, p)), 3)
                                             for q, p in (('p50', 50), ('p99', 99), ('max', 100))}
        # a reply is usable for delay D' iff it is in the pipe before frame req0+D' starts
        out['on_time_by_delay'] = {str(d): round(float(np.mean(t_done < slot0 + d / FPS)), 5) for d in range(5)}
        out['this_run'] = dict(delay=delay, decisions_with_drop=int((dropped > 0).sum()),
                               dropped_packets=int(dropped.sum()))
    if len(late):
        out['clock_late_ms'] = dict(p50=round(float(np.percentile(late, 50)) * 1e3, 3),
                                    p99=round(float(np.percentile(late, 99)) * 1e3, 3),
                                    max=round(float(late.max()) * 1e3, 2),
                                    frames_over_4ms=int((late > 0.004).sum()), frames=int(len(late)))
    by = {}
    for res, _, _ in results:
        g = by.setdefault('stage%02d_j%s' % (res['stage'], res['jitter']), [0, 0, 0, 0])
        g[0] += 1; g[1] += res['win']; g[2] += res['dealt']; g[3] += res['taken']
    out['outcomes'] = {k: dict(wins='%d/%d' % (g[1], g[0]), ci95=wilson(g[1], g[0]),
                               dealt=round(g[2] / g[0], 1), taken=round(g[3] / g[0], 1)) for k, g in sorted(by.items())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stages', default='4,8')
    p.add_argument('--fixtures', type=int, default=4, help='held-out offsets per stage and jitter level')
    p.add_argument('--jitters', default='0')
    p.add_argument('--delay', type=int, default=0)
    p.add_argument('--pairs', type=int, default=4, help='concurrent game+agent process pairs')
    p.add_argument('--device', default='cpu')
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--blocking', action='store_true', help='wait for every reply (lockstep reference)')
    p.add_argument('--lockstep-reference', action='store_true',
                   help='also replay the same fixtures in blocking mode and report both')
    p.add_argument('--hist-len', type=int, default=24)
    p.add_argument('--seed', type=int, default=4242)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs([int(s) for s in a.stages.split(',')], a.fixtures, [float(j) for j in a.jitters.split(',')],
                     a.seed)
    base = dict(model=a.model, device=a.device, stochastic=a.stochastic, hist_len=a.hist_len,
                delay=a.delay, pairs=a.pairs)
    (out / 'protocol.json').write_text(json.dumps(dict(vars(a), jobs=len(jobs)), indent=2), encoding='utf8')
    summary = {}
    modes = [('blocking' if a.blocking else 'realtime', a.blocking)]
    if a.lockstep_reference and not a.blocking:
        modes.append(('lockstep_reference', True))
    for name, blocking in modes:
        t0 = time.time()
        results = run(dict(base, blocking=blocking), jobs)
        with open(out / ('%s.jsonl' % name), 'w', encoding='utf8') as f:
            for res, _, _ in results:
                f.write(json.dumps({k: v for k, v in res.items() if k != 'actions'}) + '\n')
        if not blocking:
            np.save(out / 'decisions.npy', np.array([r for _, rs, _ in results for r in rs], dtype=np.float64))
        summary[name] = dict(summarize(results, a.delay), seconds=round(time.time() - t0, 1))
    if len(modes) == 2:
        rt_res = {(r['stage'], r['offset'], r['jitter']): r for r in map(json.loads, open(out / 'realtime.jsonl'))}
        ls_res = {(r['stage'], r['offset'], r['jitter']): r for r in map(json.loads, open(out / 'lockstep_reference.jsonl'))}
        same = sum(rt_res[k]['ram_sum'] == ls_res[k]['ram_sum'] and rt_res[k]['decisions'] == ls_res[k]['decisions']
                   for k in rt_res)
        summary['realtime_vs_lockstep'] = dict(identical_matches='%d/%d' % (same, len(rt_res)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
