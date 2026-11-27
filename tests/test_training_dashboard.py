"""The training page reads run logs; it needs only the standard library."""
import json

from training.dashboard import load, page


def write_log(folder, rows, tail=''):
    folder.mkdir(parents=True)
    (folder / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows) + tail)


def step(hours, ctc=5.9):
    return {'step': int(hours * 10), 'epoch': 0, 'hours_seen': hours, 'loss': 5.0, 'ctc': ctc, 'attention': 4.5,
            'teacher': 1.2, 'lr': 2e-4, 'gpu_gib': 4.0, 'video_hours_per_hour': 3000}


def test_load_reads_curves_and_drops_hours_logged_twice(tmp_path):
    write_log(tmp_path / 'pilot', [
        {'event': 'start', 'hours_seen': 0.0, 'total_hours': 2000.0, 'time': 300}, step(10), step(20), step(30),
        # Killed without a final save: the run resumes from the checkpoint at 20 hours.
        {'event': 'start', 'hours_seen': 20.0, 'total_hours': 2000.0, 'time': 900}, step(30, ctc=5.5),
        {'event': 'epoch', 'epoch': 0, 'step': 300, 'hours_seen': 30.0, 'dev_loss': 4.9, 'dev_wer': 0.97, 'dev_clips': 600},
    ], tail='{"step": 40')  # A line still being written.
    write_log(tmp_path / 'done', [step(5), {'event': 'finished', 'step': 50}])
    (tmp_path / 'empty').mkdir()
    write_log(tmp_path / 'zz-older', [{'event': 'start', 'hours_seen': 0.0, 'time': 100}, step(1)])
    listed = load(tmp_path)['runs']
    # Oldest first by first start; logs from before start times were written come first.
    assert [run['name'] for run in listed] == ['done', 'zz-older', 'pilot']
    runs = {run['name']: run for run in listed}
    assert [row[:2] for row in runs['pilot']['steps']] == [[10, 5.9], [20, 5.9], [30, 5.5]]
    assert runs['pilot']['steps'][0][5] == 1.2
    assert runs['pilot']['passes'] == [[1, 30.0, 0.97, 4.9]]
    assert runs['pilot']['status'] == 'training' and runs['pilot']['total_hours'] == 2000.0
    assert runs['done']['status'] == 'finished' and runs['done']['total_hours'] is None


def test_page_embeds_the_data_without_breaking_out_of_the_script(tmp_path):
    write_log(tmp_path / 'a<!--<script>b', [step(1)])
    html = page(standalone=True, runs=tmp_path)
    assert '<body class="standalone">' in html and 'a\\u003c!--\\u003cscript>b' in html
    assert html.count('<script>') == 1
    assert '<body>' in page(standalone=False, runs=tmp_path)
