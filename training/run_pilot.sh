#!/bin/bash
# Pilot run in two stages, as published from-scratch lipreading recipes do:
#   1. data/train/runs/pilot-short: clips of 4 seconds or less, until the model reads lips at all;
#   2. data/train/runs/pilot: every clip, starting from the stage 1 weights.
# Both stages also learn to predict a Whisper teacher's codes for the clip's audio (training/teacher.py).
# Waits for the download and conversion units and builds the vocabulary once.
# Rerunning resumes the unfinished stage from its last.pt.
# Start it as a user unit so it survives the terminal:
#   systemd-run --user --unit=lipreading-pilot-train --property=KillMode=mixed \
#     --property=TimeoutStopSec=180 --property=WorkingDirectory="$PWD" training/run_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python=.venv-model/bin/python
data=data/train
short=$data/runs/pilot-short
while systemctl --user is-active -q lipreading-avyt-download || systemctl --user is-active -q lipreading-avyt-convert; do
  sleep 60
done
if [ ! -f "$data/vocabulary/unigram1000.model" ]; then
  "$python" training/text.py --shards "$data"/lips/avyt-talking-*.tar "$data"/avyt/vox2/vox2-dev-*.tar \
    --output "$data/vocabulary/unigram1000" --max-shards 40
fi
# Return freed GPU memory instead of holding fragmented blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vox2-test stays out of training: its speakers are the held-out set.
train=("$python" training/train.py
  --train "$data/lips/avyt-talking-*.tar" "$data/avyt/vox2/vox2-dev-*.tar"
  --dev "$data/lips/avyt-talking-000000.tar" "$data/avyt/vox2/vox2-dev-000000.tar"
  --vocabulary "$data/vocabulary/unigram1000.model" --frames-per-batch 9600 --workers 12 --memory-gib 38
  --audio-weight 1)

if ! grep -qs '"event": "finished"' "$short/train.jsonl"; then
  "${train[@]}" --output "$short" --max-frames 100 --hours 144 --epochs 14 --lr 3e-4 --warmup 0.1 &
  # A stop request reaches only this script: pass it on and wait for the checkpoint.
  trap "kill -TERM $!" TERM INT
  wait $! || { wait $! || true; exit 1; }
  grep -qs '"event": "finished"' "$short/train.jsonl" || exit 1
  trap - TERM INT
fi
# Long clips teach nothing until the model reads short ones (the first attempt sat at 100% WER for 2,800 hours).
tail -n 1 "$short/train.jsonl" | "$python" -c 'import json, sys; sys.exit(not json.load(sys.stdin)["best_wer"] < 0.9)' \
  || { echo 'Stage 1 ended with dev WER of 90% or more; not starting stage 2.' >&2; exit 1; }
exec "${train[@]}" --output "$data/runs/pilot" --init "$short/best.pt" --hours 2400 --epochs 20 --lr 7e-4
