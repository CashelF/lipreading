#!/bin/bash
# Pilot run: waits for the download and conversion units, builds the vocabulary
# once, then trains. Rerunning resumes from data/train/runs/pilot/last.pt.
# Start it as a user unit so it survives the terminal:
#   systemd-run --user --unit=lipreading-pilot-train --property=KillMode=mixed \
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
# vox2-test stays out of training: its speakers are the held-out set.
exec "$python" training/train.py \
  --train "$data/lips/avyt-talking-*.tar" "$data/avyt/vox2/vox2-dev-*.tar" \
  --dev "$data/lips/avyt-talking-000000.tar" "$data/avyt/vox2/vox2-dev-000000.tar" \
  --vocabulary "$data/vocabulary/unigram1000.model" --output "$data/runs/pilot" \
  --hours 2500 --epochs 20 --frames-per-batch 9600 --lr 7e-4 --workers 12
