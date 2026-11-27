#!/usr/bin/env python3
"""Live page of training curves, read from data/train/runs/*/train.jsonl. See docs/TRAINING.md.

Standard library only. The logs hold numbers only, and so does the page.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time

RUNS = Path(__file__).resolve().parents[1] / 'data/train/runs'
PAGE = Path(__file__).with_suffix('.html')


def load(runs: Path = RUNS) -> dict:
    """Every run's curves: steps as [hours, ctc, attention, lr, speed, teacher], passes as [pass, hours, wer, loss]."""
    result = []
    for log in sorted(runs.glob('*/train.jsonl')):
        steps, passes, start, last = [], [], {}, {}
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # A line still being written.
            last = row
            event = row.get('event')
            if event is None:
                steps.append([row['hours_seen'], row['ctc'], row['attention'], row['lr'], row['video_hours_per_hour'],
                              row['teacher']])
            elif event == 'epoch':
                passes.append([row['epoch'] + 1, row['hours_seen'], row['dev_wer'], row['dev_loss']])
            elif event == 'start':
                start = {'time': row.get('time'), **start, 'total_hours': row.get('total_hours')}
                # A run that resumed from an older checkpoint logs those hours again.
                steps = [step for step in steps if step[0] <= row['hours_seen']]
        age = time.time() - log.stat().st_mtime
        status = last.get('event') if last.get('event') in ('finished', 'stopped') else 'training' if age < 600 else 'silent'
        result.append({'name': log.parent.name, 'status': status, 'minutes_since_log': round(age / 60),
                       'started': start.get('time') or 0, 'total_hours': start.get('total_hours'),
                       'steps': steps, 'passes': passes})
    # Oldest first, so a new run takes the next colour and the others keep theirs.
    result.sort(key=lambda run: run['started'])
    return {'runs': result, 'read_at': time.strftime('%a %H:%M')}


def page(*, standalone: bool, runs: Path = RUNS) -> str:
    """The whole page with the current data inside it. `standalone` adds a page background and live refresh."""
    data = json.dumps(load(runs)).replace('<', '\\u003c')  # A folder name must not close the script.
    return PAGE.read_text().replace('/*DATA*/null', data).replace('<body>', '<body class="standalone">' if standalone else '<body>')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body, kind = page(standalone=True), 'text/html; charset=utf-8'
        elif self.path == '/data.json':
            body, kind = json.dumps(load()), 'application/json'
        else:
            self.send_error(404)
            return
        body = body.encode()
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', required=True, help='Address to listen on, e.g. this machine\'s Tailscale address.')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
