"""Native-rate natural campaign recording; gameplay never waits for policy replies or encoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import wave

import numpy as np

from . import realtime as real


class AsyncLink(real.AgentLink):
    """Queue captures off the game thread; late replies keep the existing input."""
    def __init__(self, conn):
        super().__init__(conn, False)
        self.queue = queue.Queue(maxsize=2)
        self.skipped_captures = 0
        self.error = None
        self.new_match()
        self.sender = threading.Thread(target=self._send, daemon=True)
        self.sender.start()

    def _send(self):
        try:
            while True:
                message = self.queue.get()
                self.conn.send(message)
                if message is None:
                    return
        except BaseException as exc:
            self.error = exc

    def capture(self, rt, kind):
        if self.error:
            raise RuntimeError('Agent sender failed') from self.error
        k, self.next_id = self.next_id, self.next_id+1
        self.pending[k] = (rt.tick, rt.opponent_right(), real.now())
        try:
            self.queue.put_nowait((kind, k, rt.screen.copy()))
        except queue.Full:
            del self.pending[k]
            self.skipped_captures += 1

    def finish(self):
        # Only called after all gameplay and ending frames have been recorded.
        self.queue.put(None)
        self.sender.join(timeout=10)
        if self.error:
            raise RuntimeError('Agent sender failed') from self.error


class Movie:
    def __init__(self, out, fps, audio_rate):
        self.out, self.fps, self.rate = out, fps, audio_rate
        self.frames = self.samples = self.nonzero = 0
        self.queue = queue.Queue(maxsize=512)
        self.error = None
        self.peak_queue = 0
        self.wav = wave.open(str(out/'native_audio.wav'), 'wb')
        self.wav.setnchannels(2)
        self.wav.setsampwidth(2)
        self.wav.setframerate(audio_rate)
        self.stderr = (out/'ffmpeg.log').open('wb')
        self.encoder = subprocess.Popen([
            'C:/ffmpeg/bin/ffmpeg.exe','-hide_banner','-loglevel','error','-nostdin',
            '-f','rawvideo','-pixel_format','rgb24','-video_size','320x224',
            '-framerate',str(fps),'-i','pipe:0','-an',
            '-vf','crop=256:200:0:16,scale=iw*3:ih*3:flags=neighbor',
            '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
            '-threads','2',str(out/'video_only.mp4')], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,stderr=self.stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def _write(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                rgb, audio = item
                self.encoder.stdin.write(rgb)
                self.wav.writeframesraw(audio)
            self.encoder.stdin.close()
            if self.encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')
            self.wav.close()
            self.stderr.close()
        except BaseException as exc:
            self.error = exc

    def append(self, game):
        if self.error:
            raise RuntimeError('Movie writer failed') from self.error
        rgb = game.em.get_screen()
        pcm = np.asarray(game.em.get_audio(), dtype='<i2')
        # A full recorder queue aborts the recording instead of delaying the game.
        self.queue.put_nowait((rgb.tobytes(),pcm.tobytes()))
        self.peak_queue = max(self.peak_queue,self.queue.qsize())
        self.frames += 1
        self.samples += len(pcm)
        self.nonzero += int(np.count_nonzero(pcm))

    def finish(self):
        self.queue.put(None)
        self.worker.join(timeout=120)
        if self.worker.is_alive() or self.error:
            raise RuntimeError('Movie writer did not finish') from self.error
        assert self.nonzero > 0
        assert abs(self.samples/self.rate-self.frames/self.fps)<.05
        movie = self.out/'e7_realtime_jitter005_no_borders.mp4'
        subprocess.run(['C:/ffmpeg/bin/ffmpeg.exe','-hide_banner','-loglevel','error','-nostdin',
            '-i',str(self.out/'video_only.mp4'),'-i',str(self.out/'native_audio.wav'),
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
            '-movflags','+faststart',str(movie)],check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        probe = json.loads(subprocess.check_output(['C:/ffmpeg/bin/ffprobe.exe','-v','error',
            '-show_streams','-show_format','-of','json',str(movie)]))
        (self.out/'ffprobe.json').write_text(json.dumps(probe,indent=2))
        (self.out/'video_only.mp4').unlink()
        (self.out/'native_audio.wav').unlink()
        return movie


def run(model, reference, out, blocking=False, no_video=False, ending_frames=1800, max_matches=12):
    import random
    from PIL import Image
    from .env import MatchTracker
    from .common import NEUTRAL
    job = {k:reference[k] for k in ('offset','jitter','seed','deterministic')}
    out.mkdir(parents=True,exist_ok=False)
    ctx = mp.get_context('spawn')
    agent_conn, game_conn = ctx.Pipe()
    agent = ctx.Process(target=real.agent_main,args=(agent_conn,str(model),'cpu',True,24))
    agent.start()
    assert game_conn.recv() == 'ready'  # Loading and warm-up are before game time begins.
    link = real.AgentLink(game_conn,True) if blocking else AsyncLink(game_conn)
    link.new_match()
    rt = real.make_runtime(job['seed'],job['jitter'],0,link)
    fps = float(rt.game.em.get_screen_rate())
    real.FPS = fps
    clock = None if blocking else real.Clock()
    movie = None if no_video else Movie(out,fps,int(rt.game.em.get_audio_rate()))
    original_frame, original_set = rt.frame, rt.set_state
    loads = 0
    start = None
    last_progress = time.time()
    matches, beaten, reason = [], [], None

    def set_state(blob):
        nonlocal loads,start
        loads += 1
        assert loads == 1, 'No mid-campaign loading is allowed'
        original_set(blob)
        if clock is not None:
            rt.clock = clock
            clock.start(rt.tick)
        start = real.now()

    def frame(request,full=True):
        nonlocal last_progress
        screen = original_frame(request,full)
        if movie:
            movie.append(rt.game)
        if time.time()-last_progress >= 10:
            progress = dict(frames=rt.total_frames,game_seconds=rt.total_frames/fps,
                wall_seconds=real.now()-start,matches_won=len(beaten),current_enemy=rt.enemy_id(),
                decisions=len(link.records),dropped_packets=sum(int(r[6]) for r in link.records),
                blocking=blocking)
            tmp = out/'progress.tmp'
            tmp.write_text(json.dumps(progress,indent=2))
            tmp.replace(out/'progress.json')
            print(json.dumps(progress),flush=True)
            last_progress = time.time()
        return screen

    rt.set_state,rt.frame = set_state,frame
    try:
        rt.load_stage(1,job['offset'],random.Random(job['seed']))
        enemy = rt.enemy_id()
        while reason is None:
            tr = MatchTracker(rt)
            match_start = rt.total_frames
            link.capture(rt,'reset')
            while True:
                for i in range(4):
                    rt.frame(None,full=i==3)
                tr.update()
                over = truncated = False
                if not tr.live:
                    _,over,truncated = tr.settle()
                if over or truncated:
                    if truncated:
                        reason = 'settle_timeout'
                    break
                if rt.total_frames > 400000:
                    reason = 'frame_limit'
                    break
                link.capture(rt,'step')
            if reason:
                break
            won = tr.score[0] >= 2
            match = dict(enemy=enemy,stage=len(matches)+1,jitter=job['jitter'],win=won,score=list(tr.score),
                         dealt=tr.dealt,taken=tr.taken,frames=rt.total_frames-match_start)
            matches.append(match)
            (out/'matches.json').write_text(json.dumps(matches,indent=2))
            print('match',json.dumps(match),flush=True)
            if not won:
                reason = 'lost_match'
                break
            beaten.append(enemy)
            if len(beaten) == max_matches:
                reason = 'cleared' if max_matches==12 else 'test_prefix_complete'
                break
            seen_lock = False
            for _ in range(5000):
                rt.neutral(4)
                e = rt.enemy_id()
                if not rt.controllable():
                    seen_lock = True
                elif (seen_lock and e < 12 and e not in beaten and rt.round_state()==5
                      and min(rt.hp())>100 and rt.score()==(0,0) and rt.bonus()==0):
                    enemy = e
                    break
            else:
                reason = 'next_match_timeout'
        campaign_frames = rt.total_frames
        campaign_wall = real.now()-start
        Image.fromarray(rt.game.em.get_screen()[16:216,:256]).save(out/'terminal.png')
        if reason=='cleared':
            rt.neutral(ending_frames)
            Image.fromarray(rt.game.em.get_screen()[16:216,:256]).save(out/'ending.png')
        game_wall = real.now()-start
        # All synchronization below happens after the game has ended.
        if not blocking:
            link.poll(rt)
        link.drain()
        if blocking:
            game_conn.send(None)
        else:
            link.finish()
        agent.join(timeout=10)
        recs = np.asarray(link.records,dtype=np.float64)
        np.save(out/'decisions.npy',recs)
        timing = real.summarize([(m,[],[]) for m in matches],0)
        # Supply the full recording once, while retaining individual match outcomes.
        if matches:
            timing = real.summarize([(matches[0],link.records,clock.late if clock else [])]+
                                   [(m,[],[]) for m in matches[1:]],0)
        scores_match = [m['score'] for m in matches] == [m['score'] for m in reference['matches'][:len(matches)]]
        damage_match = all(m['dealt']==r['dealt'] and m['taken']==r['taken']
                           for m,r in zip(matches,reference['matches']))
        result = dict(model=str(model.resolve()),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            job=job,cleared=reason=='cleared',reason=reason,wins=len(beaten),matches=matches,
            native_fps=fps,blocking=blocking,policy_waits_during_gameplay=blocking,
            wall_clock_paced=not blocking,encoder_on_separate_thread=True,
            first_stage_loads=loads,mid_campaign_loads=0,continues=0,
            campaign_frames=campaign_frames,campaign_game_seconds=campaign_frames/fps,
            campaign_wall_seconds=campaign_wall,all_game_wall_seconds=game_wall,
            reference_scores_match=scores_match,reference_damage_match=damage_match,
            reference_frames_match=campaign_frames==reference['frames'],
            skipped_captures=getattr(link,'skipped_captures',0),timing=timing,
            policy_inputs=['motion-preprocessed 84x84 temporal visual observation','24 issued-action history'],
            wrapper_ram_uses=['relative forward/back resolution','reward and round/match progression'],
            selected_existing_successful_case=True)
        if blocking:
            assert scores_match and damage_match
            if max_matches==12:
                assert result['reference_frames_match'] and result['cleared']
        if movie:
            path = movie.finish()
            result.update(movie=str(path.resolve()),video_frames=movie.frames,audio_samples=movie.samples,
                          audio_rate=movie.rate,audio_nonzero=movie.nonzero,
                          recorder_peak_queue=movie.peak_queue,native_crop=[256,200,0,16],output_size=[768,600])
        (out/'manifest.json').write_text(json.dumps(result,indent=2))
        (out/'DONE').write_text(reason)
        print(json.dumps(result),flush=True)
        return result
    finally:
        rt.game.close()
        if agent.is_alive():
            agent.terminate()
            agent.join()
        game_conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--test-blocking',action='store_true')
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--max-matches',type=int,default=12)
    p.add_argument('--ending-frames',type=int,default=1800)
    p.add_argument('--success-index',type=int,default=0,help='Index among existing jitter .05 successful cases')
    a = p.parse_args()
    real._realtime_process()
    rows = [json.loads(x) for x in Path(a.reference).read_text().splitlines()]
    successful = [r for r in rows if r['cleared'] and r['jitter']==.05]
    reference = successful[a.success_index]
    run(Path(a.model),reference,Path(a.out),blocking=a.test_blocking,no_video=a.no_video,
        max_matches=a.max_matches,ending_frames=a.ending_frames)


if __name__=='__main__':
    main()
