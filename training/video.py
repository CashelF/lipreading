"""Decode and encode small grayscale mouth clips with PyAV."""
from __future__ import annotations

import io

import av
import numpy as np


def frame_count(data: bytes) -> int:
    """Frame count from the container header, without decoding; 0 when the header omits it."""
    with av.open(io.BytesIO(data)) as container:
        return container.streams.video[0].frames


# Video-range luma (16–235) to full-range gray, as one table lookup per clip.
_FULL_RANGE = np.clip(np.round((np.arange(256) - 16) * 255 / 219), 0, 255).astype(np.uint8)


def decode_gray(data: bytes) -> np.ndarray:
    """Return full-range uint8 frames shaped (T, H, W).

    Reads the luma plane directly instead of converting each frame, which
    roughly doubles decoding speed. Clips are stored in video range.
    """
    planes = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'NONE'  # Frames are tiny; threads only add overhead.
        for frame in container.decode(stream):
            if frame.format.name not in ('yuv420p', 'yuvj420p'):
                frame = frame.reformat(format='yuv420p')
            plane = frame.planes[0]
            luma = np.frombuffer(plane, np.uint8).reshape(frame.height, plane.line_size)[:, :frame.width]
            planes.append(luma if frame.format.name == 'yuvj420p' else luma.copy())
            full_range = frame.format.name == 'yuvj420p'
    if not planes:
        raise ValueError('Clip has no video frames.')
    stacked = np.stack(planes)
    return stacked if full_range else _FULL_RANGE[stacked]


def encode_gray(frames: np.ndarray, fps: int = 25, crf: int = 20) -> bytes:
    output = io.BytesIO()
    with av.open(output, 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(crf), 'preset': 'medium'}
        for image in frames:
            # Convert full-range gray to standard video-range luma explicitly. Handing
            # gray frames to the encoder stores them unscaled, and decoding then
            # stretches the contrast a second time.
            frame = av.VideoFrame.from_ndarray(image, format='gray').reformat(format='yuv420p')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def transcode_audio(data: bytes, bitrate: int = 16000) -> bytes | None:
    """Mono 16 kHz Opus copy of a clip's soundtrack, for later Whisper-feature targets."""
    output = io.BytesIO()
    with av.open(io.BytesIO(data)) as source:
        if not source.streams.audio:
            return None
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        with av.open(output, 'w', format='ogg') as container:
            stream = container.add_stream('libopus', rate=16000)
            stream.bit_rate = bitrate
            stream.layout = 'mono'
            for frame in source.decode(audio=0):
                for resampled in resampler.resample(frame):
                    for packet in stream.encode(resampled):
                        container.mux(packet)
            for resampled in resampler.resample(None):
                for packet in stream.encode(resampled):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return output.getvalue()
