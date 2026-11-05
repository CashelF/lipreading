#!/usr/bin/env python3
"""Fetch pinned official Light-ASD artifacts for local visual speaking detection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.visual_speech import (  # noqa: E402
    CHECKPOINTS, ENCODER_SHA256, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY,
)


def download(relative: str, destination: Path, expected: str | None = None) -> str:
    url = f'https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/{UPSTREAM_COMMIT}/{relative}'
    with urlopen(url, timeout=60) as response:
        body = response.read()
    digest = hashlib.sha256(body).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Unexpected downloaded checksum for {relative}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.download')
    temp.write_bytes(body)
    temp.replace(destination)
    return digest


def main() -> None:
    source_dir = ROOT / 'vendor' / 'light-asd'
    weights_dir = ROOT / 'models' / 'visual_speech'
    files = {}
    for filename in ('model/Encoder.py', 'model/Model.py', 'model/Classifier.py',
                     'ASD.py', 'loss.py', 'dataLoader.py', 'Columbia_test.py',
                     'README.md', 'LICENSE'):
        expected = ENCODER_SHA256 if filename == 'model/Encoder.py' else None
        files[filename] = download(filename, source_dir / filename, expected)
    for filename, digest in CHECKPOINTS.values():
        files['weight/' + filename] = download(
            'weight/' + filename, weights_dir / filename, digest)
    manifest = {'repository': UPSTREAM_REPOSITORY, 'commit': UPSTREAM_COMMIT,
                'license': 'MIT', 'sha256': files,
                'purpose': 'Frozen visual-only speaking detector; admission settings are defined separately.'}
    (source_dir / 'PROVENANCE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Installed pinned Light-ASD source and both official checkpoints for local visual speaking detection.')
    print('The live speech gate has not been changed.')


if __name__ == '__main__':
    main()
