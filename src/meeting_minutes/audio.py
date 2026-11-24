"""Decode arbitrary meeting recordings into the one format both models want.

Whisper and pyannote both expect 16 kHz mono PCM. Rather than let each library
decode the source file independently — which would mean two different resampling
paths and, worse, two slightly different time axes — we decode once up front and
feed the same samples to both. That shared time axis is what makes the merge
step in `merge.py` trustworthy.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


class FfmpegMissingError(RuntimeError):
    """Raised when ffmpeg is not on PATH."""


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegMissingError(
            "ffmpeg not found on PATH. Install it with:\n"
            "    sudo apt install ffmpeg"
        )
    return ffmpeg


def decode_to_wav(src: Path, dest: Path) -> Path:
    """Decode `src` to 16 kHz mono 16-bit PCM at `dest`.

    Accepts anything ffmpeg can read (m4a, mp3, mp4, ogg, wav, ...).
    """
    ffmpeg = _require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",  # overwrite; the destination is a derived artifact
            "-i", str(src),
            "-ac", "1",  # downmix to mono
            "-ar", str(TARGET_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(dest),
        ],
        check=True,
        # ffmpeg writes progress to stderr even on success, so only surface it
        # when the command actually fails.
        capture_output=True,
    )
    return dest


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a wav file as float32 samples in [-1, 1], plus its sample rate."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:  # defensive: decode_to_wav already downmixed
        samples = samples.mean(axis=1)
    return samples, sample_rate


def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    return len(samples) / sample_rate


def format_timestamp(seconds: float) -> str:
    """Seconds -> HH:MM:SS, for timestamps in the transcript given to Claude."""
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
