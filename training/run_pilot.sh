#!/bin/bash
# Pilot run, in three steps:
#   1. data/train/features: FLARE runs once over every clip and saves what training reads
#      (its lip front-end output as the input, its encoder output as the target);
#   2. data/train/runs/flare-18: a small text side learns to read FLARE's whole encoder;
#   3. data/train/runs/keep-12: encoder layers are dropped and the rest is trained to make up for them.
# Waits for the download and conversion units and builds the vocabulary once. Rerunning skips
# finished feature shards and resumes training from last.pt.
# Start it as a user unit so it survives the terminal:
#   systemd-run --user --unit=lipreading-pilot-train --property=MemoryMax=60G --property=KillMode=mixed \
#     --property=TimeoutStopSec=180 --property=WorkingDirectory="$PWD" training/run_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python=.venv-model/bin/python
data=data/train
while systemctl --user is-active -q lipreading-avyt-download || systemctl --user is-active -q lipreading-avyt-convert; do
  sleep 60
done
if [ ! -f "$data/vocabulary/unigram1000.model" ]; then
  "$python" training/text.py --shards "$data"/lips/avyt-talking-*.tar "$data"/avyt/vox2/vox2-dev-*.tar \
    --output "$data/vocabulary/unigram1000" --max-shards 40
fi
# Return freed GPU memory instead of holding fragmented blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vox2-test stays out of both steps: its speakers are the held-out set.
"$python" training/extract_features.py --output "$data/features" \
  --shards "$data/lips/avyt-talking-*.tar" "$data/avyt/vox2/vox2-dev-*.tar"
# Stage 1: the text side learns to read FLARE's whole encoder, which stays as it is.
if ! grep -qs '"event": "finished"' "$data/runs/flare-18/train.jsonl"; then
  "$python" training/train.py --features "$data/features" --train "avyt-talking-*" "vox2-dev-*" \
    --dev avyt-talking-000000 vox2-dev-000000 --vocabulary "$data/vocabulary/unigram1000.model" \
    --output "$data/runs/flare-18" --hours 2400 --epochs 5 --frames-per-batch 9600 --lr 5e-4 &
  # A stop request reaches only this script: pass it on and wait for the checkpoint.
  trap "kill -TERM $!" TERM INT
  wait $! || { wait $! || true; exit 1; }
  grep -qs '"event": "finished"' "$data/runs/flare-18/train.jsonl" || exit 1
  trap - TERM INT
fi
# Stage 2: drop encoder layers and train what is left, text side included, to give what the whole encoder gave.
exec "$python" training/train.py --features "$data/features" --train "avyt-talking-*" "vox2-dev-*" \
  --dev avyt-talking-000000 vox2-dev-000000 --vocabulary "$data/vocabulary/unigram1000.model" \
  --output "$data/runs/keep-12" --init "$data/runs/flare-18/best.pt" \
  --keep-layers 0 1 5 6 7 9 11 12 13 14 15 17 --hours 2400 --epochs 1 --frames-per-batch 9600 --lr 1e-4
