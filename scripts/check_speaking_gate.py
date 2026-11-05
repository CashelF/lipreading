#!/usr/bin/env python3
"""Validate continuation and trial start confirmation on saved local examples.

Offline mode replays already computed scores, including coalesced updates.
Live mode sends original JPEGs only to this machine's loopback inference server.
Captions are checked in memory and never printed or saved in these reports.
This is a small, personal validation set, not a word-accuracy benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.speech_gate import (CHECKPOINT, CONTEXT_FRAMES, MEAN_FRAMES, SCORE_THRESHOLD,
                                 START_SCORE_THRESHOLD, START_CONFIRM_FRAMES, SpeechGateState)
from backend.stream import sample_times
from scripts.evaluate_speaking_examples import replay_windows

ORIGIN = 'http://127.0.0.1:8000'
EXAMPLES = ROOT / 'data/speaking-examples'
REPORTS = ROOT / 'models/visual_speech'
LABELS = ('quiet_still', 'quiet_motion', 'speech_normal', 'speech_subtle')


def check_frozen_rule():
    frozen = json.loads((REPORTS / 'frozen-gate-selection.json').read_text())
    actual = {'checkpoint': CHECKPOINT, 'threshold': SCORE_THRESHOLD,
              'smoothing_frames': MEAN_FRAMES, 'context_frames_each_side': CONTEXT_FRAMES}
    if any(frozen.get(key) != value for key, value in actual.items()):
        raise ValueError('The continuation rule differs from the original frozen selection.')


def load_records():
    result = {}
    for path in EXAMPLES.glob('*/manifest.json'):
        if re.fullmatch(r'[0-9a-f]{32}', path.parent.name):
            manifest = json.loads(path.read_text())
            if manifest['id'] != path.parent.name:
                raise ValueError('Sample identifier does not match its directory.')
            result[manifest['id']] = (path.parent, manifest)
    return result


def schedules(count: int) -> dict[str, list[int]]:
    if count < 1:
        raise ValueError('There are no evaluated windows.')
    return {
        'sequential': list(range(count)),
        'coalesced_two_seconds': sorted({index for index in (0, 2, 4, count - 1) if index < count}),
        'coalesced_three_seconds': sorted({index for index in (0, 3, count - 1) if index < count}),
    }


def offline_check() -> dict:
    records = load_records()
    results = []
    for take in (1, 2):
        report = json.loads((REPORTS / f'take{take}-report.json').read_text())
        rows = report['recordings']
        if {row['label'] for row in rows} != set(LABELS):
            raise ValueError(f'Take {take} does not contain all four conditions.')
        for row in rows:
            directory, manifest = records[row['id']]
            if manifest['take'] != take or row['take'] != take or manifest['label'] != row['label']:
                raise ValueError('Report and recording labels disagree.')
            clips = list(replay_windows(directory, manifest))
            if len(clips) != len(row['windows']):
                raise ValueError('The saved report no longer matches live window replay.')
            runs = []
            quiet = row['label'].startswith('quiet_')
            for name, indices in schedules(len(clips)).items():
                state = SpeechGateState()
                updates = []
                for index in indices:
                    window = row['windows'][index]
                    if 'visual_error' in window:
                        raise ValueError('The saved report contains a visual preprocessing error.')
                    scores = window['candidates'][CHECKPOINT]['visual_scores']
                    times = sample_times(clips[index])
                    activity = state.update(scores, times)
                    updates.append({'index': index, 'state': activity['state'],
                                    'suppressed': activity['suppress'],
                                    'metrics': activity['metrics']})
                admitted = sum(not update['suppressed'] for update in updates)
                confirmed = [update for update in updates if update['metrics'].get('speech_start_confirmed')]
                gaps = sum(bool(update['metrics'].get('speech_gap')) for update in updates)
                invalid = sum(bool(update['metrics'].get('speech_invalid')) for update in updates)
                start_passed = not confirmed if quiet else bool(confirmed) and confirmed[0]['index'] == indices[0]
                passed = (admitted == (0 if quiet else len(updates))
                          and not gaps and not invalid and start_passed)
                runs.append({'schedule': name, 'updates': updates, 'admitted': admitted,
                             'start_confirmed_updates': len(confirmed),
                             'first_confirmation_ms': confirmed[0]['metrics']['speech_start_confirmed_ms'] if confirmed else None,
                             'gap_warnings': gaps, 'invalid_warnings': invalid, 'passed': passed})
            result = {'id': row['id'], 'label': row['label'], 'take': take,
                      'current_gate_admitted': sum(w['current_gate'].get('suppress') is False for w in row['windows']),
                      'schedules': runs, 'passed': all(run['passed'] for run in runs)}
            results.append(result)
            print(json.dumps({'mode': 'offline', 'label': row['label'], 'take': take,
                              'passed': result['passed'],
                              'admitted_by_schedule': {run['schedule']: run['admitted'] for run in runs}}), flush=True)
    return {'passed': all(row['passed'] for row in results), 'recordings': results,
            'limitations': ['Overlapping updates are correlated; recordings are the independent units.',
                            'Both takes informed start confirmation; neither is held out for the new rule.',
                            'Only gate decisions are checked here, not captions or frame-level speech labels.']}


def packets_for_record(directory: Path, manifest: dict, offset_ms: float = 0.):
    names, times = manifest['frame_files'], manifest['timestamps_ms']
    if not names or len(names) != len(times) or len(names) != manifest['frames']:
        raise ValueError('Malformed recording manifest.')
    packets = []
    for name, timestamp in zip(names, times):
        if not re.fullmatch(r'\d{6}\.jpg', name):
            raise ValueError('Invalid recording frame name.')
        relative_time = timestamp - times[0] + offset_ms
        packets.append((relative_time, struct.pack('<d', relative_time) + (directory / name).read_bytes()))
    return packets


async def replay_live(label: str, packets: list, *, mixed: bool = False,
                      expected_utterances: int = 1, speech_gate: bool = True,
                      max_utterance_seconds: int | None = None) -> dict:
    from websockets.asyncio.client import connect

    with urllib.request.urlopen(ORIGIN + '/api/status', timeout=5) as response:
        status = json.load(response)
    if not status.get('ready') or status.get('busy'):
        raise RuntimeError('The inference server must be ready and idle before live validation.')
    if status.get('speech_start_confirmation') != {'threshold': START_SCORE_THRESHOLD,
                                                  'consecutive_means': START_CONFIRM_FRAMES}:
        raise RuntimeError('The inference server does not have the new start-confirmation rule.')
    messages = []
    stopped = False
    async with connect('ws://127.0.0.1:8000/api/stream', origin=ORIGIN, max_size=2 ** 20) as ws:
        options = {} if max_utterance_seconds is None else {'max_utterance_seconds': max_utterance_seconds}
        await ws.send(json.dumps({'type': 'start', 'speech_gate': speech_gate, 'verify_visual': True, **options}))
        first = json.loads(await ws.recv())
        if first.get('state') != 'capturing':
            raise RuntimeError('The server did not admit the live validation session.')

        async def receive():
            nonlocal stopped
            async for raw in ws:
                message = json.loads(raw)
                if message['type'] == 'error':
                    raise RuntimeError('Live validation received an inference/transport error.')
                messages.append(message)
                if message['type'] == 'stopped':
                    stopped = True
                    return

        receiver = asyncio.create_task(receive())
        began = time.perf_counter()
        try:
            for timestamp, packet in packets:
                await asyncio.sleep(max(0., began + timestamp / 1000 - time.perf_counter()))
                if receiver.done():
                    await receiver
                    raise RuntimeError('Inference closed before all validation frames were sent.')
                await ws.send(packet)
            await ws.send(json.dumps({'type': 'stop'}))
            await asyncio.wait_for(receiver, timeout=90)
        finally:
            if not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    snapshots = [m for m in messages if m['type'] == 'transcript']
    activities = [m for m in messages if m['type'] == 'activity']
    if not stopped or not snapshots or not snapshots[-1].get('final'):
        raise AssertionError('Live validation did not produce a finalized transcript snapshot.')
    if speech_gate and (not activities or any(m.get('metrics', {}).get('speech_score_threshold') != SCORE_THRESHOLD for m in activities)):
        raise AssertionError('The deployed server did not use the frozen trained speech gate.')
    if speech_gate and any(m.get('metrics', {}).get('speech_start_threshold') != START_SCORE_THRESHOLD for m in activities):
        raise AssertionError('The deployed server did not report the expected start-confirmation rule.')
    if speech_gate and not label.startswith('quiet_') and not any(m.get('metrics', {}).get('speech_start_confirmed') for m in activities):
        raise AssertionError('The speaking recording never confirmed a start.')
    actual_decodes = [m for m in snapshots if m.get('timings_ms', {}).get('inference', 0.) > 0]
    decoded_intervals = [interval for m in snapshots for interval in m.get('decoded_intervals', [])]
    refinement_passes = sum(m.get('timings_ms', {}).get('refinement_passes', 0) for m in snapshots)
    if status.get('pause_regions_version') == 1:
        if bool(actual_decodes) != bool(decoded_intervals):
            raise AssertionError('The deployed server did not report its recognition intervals.')
        # Fully active evidence is 480 ms behind the final sample, and capture
        # timestamps can extend less than one sample beyond it. A known quiet
        # tail instead uses the existing 500 ms context. Never add whole seconds
        # of silence merely because Stop or a later rolling window arrived.
        if any(interval['end_ms'] - interval['last_active_ms'] > 540.001
               for interval in decoded_intervals):
            raise AssertionError('Recognition included excessive trailing context after speech.')
        if sum(bool(interval['final']) for interval in decoded_intervals) != refinement_passes:
            raise AssertionError('Recognition intervals disagree with final refinement counts.')
    if status.get('utterance_refinement'):
        expected_refinements = 0 if label.startswith('quiet_') and speech_gate else expected_utterances
        if refinement_passes != expected_refinements:
            raise AssertionError('Expected one final refinement per detected test utterance, and none for quiet video.')
    if label.startswith('quiet_'):
        if any(m['text'] for m in snapshots) or (speech_gate and actual_decodes):
            raise AssertionError('A quiet recording displayed captions or bypassed the speaking gate.')
        if not speech_gate:
            withheld = sum(m.get('timings_ms', {}).get('grounding_withheld', 0) for m in snapshots)
            if not actual_decodes or withheld != len(decoded_intervals):
                raise AssertionError('The visual verifier did not withhold every forced quiet-clip prediction.')
    elif not actual_decodes or not snapshots[-1]['text']:
        raise AssertionError('The speaking recording failed to reach decoding and display a caption.')
    stable = ''
    for snapshot in snapshots:
        updated = snapshot.get('stable_text', '')
        if stable and updated != stable and not updated.startswith(stable + ' '):
            raise AssertionError('A later update changed previously completed words.')
        stable = updated
    tail_checks = None
    if mixed:
        # The quiet portion lasts eight seconds. Check its final four seconds:
        # some of those six-second windows still contain earlier speech, which
        # must not be reused as fresh evidence to start another draft. One
        # final refinement is expected after a pause; it may arrive here when
        # processing coalesces windows. After that, every caption must stay put.
        final_time = packets[-1][0]
        tail = [m for m in snapshots if m.get('window_end_ms', 0.) >= final_time - 4000]
        if not tail or any(m.get('timings_ms', {}).get('inference', 0.) > 0
                           and not m.get('timings_ms', {}).get('refinement_passes', 0)
                           for m in tail):
            raise AssertionError('Old speech triggered another decode during the quiet tail.')
        settled = tail
        if status.get('utterance_refinement'):
            final_index = max(i for i, m in enumerate(snapshots)
                              if m.get('timings_ms', {}).get('refinement_passes', 0))
            if snapshots[final_index].get('final') or final_index == len(snapshots) - 1:
                raise AssertionError('The quiet pause did not finalize speech before Stop.')
            settled = snapshots[final_index:]
            if any(m.get('timings_ms', {}).get('inference', 0.) > 0 for m in settled[1:]):
                raise AssertionError('Decoding continued after the utterance was finalized.')
        if not settled[0]['text'] or any(m['text'] != settled[0]['text'] for m in settled):
            raise AssertionError('Existing captions changed or disappeared during the quiet tail.')
        tail_checks = {'updates': len(tail),
                       'refinement_passes': sum(m.get('timings_ms', {}).get('refinement_passes', 0) for m in tail),
                       'draft_decode_calls': 0, 'caption_preserved_after_refinement': True}
    # Never put recognized text, image pixels, or packets in a report/exception.
    result = {'label': label, 'passed': True, 'frames': len(packets),
              'duration_seconds': round(packets[-1][0] / 1000, 3),
              'updates': len(snapshots),
              'actual_decode_calls': len(decoded_intervals) if status.get('pause_regions_version') == 1 else len(actual_decodes),
              'decoded_intervals': decoded_intervals,
              'refinement_passes': refinement_passes,
              'expected_utterances': 0 if label.startswith('quiet_') and speech_gate else expected_utterances,
              'speech_gate': speech_gate,
              'grounding_withheld': sum(m.get('timings_ms', {}).get('grounding_withheld', 0) for m in snapshots),
              'caption_present': bool(snapshots[-1]['text']),
              'activities': [{key: activity.get(key) for key in ('segment_id', 'state', 'suppress', 'metrics', 'speech_regions')}
                             for activity in activities],
              'timings_ms': [m.get('timings_ms', {}) for m in snapshots],
              'quiet_tail': tail_checks}
    print(json.dumps({key: result[key] for key in ('label', 'passed', 'frames', 'updates', 'actual_decode_calls', 'caption_present')}), flush=True)
    return result


async def live_check(include_mixed: bool, include_grounding: bool = False) -> dict:
    records = load_records()
    saved_take = {}
    for directory, manifest in records.values():
        if manifest['take'] == 2:
            if manifest['label'] in saved_take:
                raise ValueError('Keep one recording per condition and take for an unambiguous live check.')
            saved_take[manifest['label']] = (directory, manifest)
    if set(saved_take) != set(LABELS):
        raise ValueError('Take 2 must contain all four recording conditions.')
    results = []
    for label in LABELS:
        directory, manifest = saved_take[label]
        results.append(await replay_live(label, packets_for_record(directory, manifest)))
    if include_grounding:
        for label in ('quiet_still', 'quiet_motion'):
            directory, manifest = saved_take[label]
            results.append(await replay_live(label + '_forced_decoder',
                           packets_for_record(directory, manifest), speech_gate=False))
    if include_mixed:
        directory, manifest = saved_take['speech_subtle']
        speaking = packets_for_record(directory, manifest)
        directory, manifest = saved_take['quiet_still']
        quiet = packets_for_record(directory, manifest, offset_ms=speaking[-1][0] + 40.)
        results.append(await replay_live('mixed_subtle_then_quiet', speaking + quiet, mixed=True))
        # A genuine second phrase must independently confirm after the pause,
        # retain the first phrase, and settle before the final quiet tail ends.
        pause = [packet for packet in quiet if packet[0] < speaking[-1][0] + 4040.]
        directory, manifest = saved_take['speech_normal']
        next_speech = packets_for_record(directory, manifest, offset_ms=pause[-1][0] + 40.)
        directory, manifest = saved_take['quiet_still']
        final_quiet = packets_for_record(directory, manifest, offset_ms=next_speech[-1][0] + 40.)
        results.append(await replay_live('mixed_two_phrases_with_pause',
                       speaking + pause + next_speech + final_quiet,
                       mixed=True, expected_utterances=2))
    return {'passed': all(row['passed'] for row in results), 'recordings': results,
            'limitations': ['A displayed caption is not evidence that the recognized words are correct.',
                            'The saved clips informed the new start rule and are development checks, not held-out evidence.',
                            'Mixed replay joins two recordings artificially; it does not validate natural onset/offset accuracy.']}


def write_private(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Replay saved scores (default if no mode is specified).')
    parser.add_argument('--live', action='store_true', help='Replay saved video to the local running server.')
    parser.add_argument('--mixed', action='store_true', help='Also test artificial speech/pause transitions and two separated phrases; requires --live.')
    parser.add_argument('--grounding', action='store_true', help='Force quiet clips through FLARE to test visual verification; requires --live.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mixed and not args.live:
        parser.error('--mixed requires --live')
    if args.grounding and not args.live:
        parser.error('--grounding requires --live')
    if not args.live:
        args.offline = True
    output = args.output or REPORTS / ('gate-live-validation.json' if args.live else 'gate-validation.json')
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'checkpoint': CHECKPOINT, 'threshold': SCORE_THRESHOLD,
              'start_threshold': START_SCORE_THRESHOLD, 'start_confirm_frames': START_CONFIRM_FRAMES,
              'caption_text_saved': False, 'passed': False}
    try:
        check_frozen_rule()
        if args.offline:
            result['offline'] = offline_check()
        if args.live:
            result['live'] = asyncio.run(live_check(args.mixed, args.grounding))
        result['passed'] = all(value['passed'] for key, value in result.items() if key in ('offline', 'live'))
    except (AssertionError, RuntimeError, ValueError, OSError) as exc:
        result['failure'] = str(exc)
        print(f'Validation failed: {exc}', file=sys.stderr)
    write_private(output, result)
    print(f'Local report: {output}', flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
