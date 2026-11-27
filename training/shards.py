"""Tar shards of short mouth clips with transcripts.

Two layouts are read: the upstream AVYT release (``.video``/``.label``) and
the compact layout written by ``prepare_avyt`` (``.mp4``/``.txt``).
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterator

VIDEO_EXT = ('mp4', 'video')
TEXT_EXT = ('txt', 'label')


def read_samples(path: str | Path) -> Iterator[dict]:
    """Yield ``{'key', 'video', 'text'}`` for each complete sample, in file order."""
    current, key = {}, None
    with tarfile.open(path, 'r|') as tar:
        for member in tar:
            if not member.isfile() or '.' not in member.name:
                continue
            stem, ext = member.name.rsplit('.', 1)
            if stem != key:
                if _complete(current):
                    yield current
                current, key = {'key': stem}, stem
            if ext in VIDEO_EXT:
                current['video'] = tar.extractfile(member).read()
            elif ext in TEXT_EXT:
                current['text'] = tar.extractfile(member).read().decode('utf-8', 'replace').strip()
        if _complete(current):
            yield current


def _complete(sample: dict) -> bool:
    return 'video' in sample and 'text' in sample


class ShardWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pending = self.path.with_name('.' + self.path.name + '.pending')
        self.tar = tarfile.open(self.pending, 'w')
        self.count = 0

    def add(self, key: str, video: bytes, text: str, audio: bytes | None = None):
        members = [('mp4', video), ('txt', text.encode('utf-8'))]
        if audio:
            members.append(('ogg', audio))
        for ext, data in members:
            info = tarfile.TarInfo(f'{key}.{ext}')
            info.size = len(data)
            self.tar.addfile(info, io.BytesIO(data))
        self.count += 1

    def close(self):
        """Publish atomically, so a reader never sees a half-written shard."""
        self.tar.close()
        self.pending.rename(self.path)

    def abort(self):
        self.tar.close()
        self.pending.unlink(missing_ok=True)
