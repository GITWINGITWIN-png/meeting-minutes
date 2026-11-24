"""Speaker diarization: "who spoke when", independent of what was said.

pyannote returns speaker *turns* on the same time axis as the transcript, which
`merge.py` then joins against. The models are gated on Hugging Face: you must
accept the licence for both `pyannote/speaker-diarization-3.1` and its
dependency `pyannote/segmentation-3.0`, then supply a read token.

Why this takes samples rather than a file path
----------------------------------------------
pyannote 4.x decodes audio with torchcodec, whose native libraries are built
against a specific CUDA version — the PyPI build targets CUDA 13 while this
project pins torch to the cu128 index for Blackwell support, so those libraries
cannot load here. pyannote documents the way out in that very error message:
hand it an in-memory ``{"waveform": ..., "sample_rate": ...}`` and the whole
decoding layer is skipped.

That is a better fit anyway. `audio.py` has already decoded the recording once
with ffmpeg, and feeding those exact samples to both the ASR and this stage is
what makes the shared time axis in `merge.py` a guarantee rather than a
coincidence — there is only ever one decode.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


@dataclass
class Turn:
    """A contiguous span attributed to one speaker."""

    start: float
    end: float
    speaker: str


class MissingTokenError(RuntimeError):
    """Raised when no Hugging Face token is available for the gated models."""


def _resolve_token(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for var in TOKEN_ENV_VARS:
        token = os.environ.get(var)
        if token:
            return token
    raise MissingTokenError(
        "Speaker diarization needs a Hugging Face token.\n"
        "  1. Accept the licence at https://hf.co/pyannote/speaker-diarization-3.1\n"
        "  2. Accept the licence at https://hf.co/pyannote/segmentation-3.0\n"
        "  3. Create a read token at https://hf.co/settings/tokens\n"
        "  4. export HF_TOKEN=hf_xxx\n"
        "Or skip this stage with --no-diarize."
    )


def _import_pipeline():
    """Import pyannote's Pipeline, muting the torchcodec load failure.

    Importing pyannote prints a multi-line traceback when torchcodec's native
    libraries will not load. Here that warning is a false alarm: we never take
    the code path that needs them (see the module docstring). It is silenced
    narrowly, at this one import, so a genuine problem elsewhere still surfaces.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torchcodec.*")
        from pyannote.audio import Pipeline

    return Pipeline


def _as_annotation(output):
    """Get the speaker `Annotation` out of whatever the pipeline returned.

    pyannote 4.x returns a `DiarizeOutput` dataclass, except for pipelines
    loaded in legacy mode which still return a bare `Annotation` as 3.x did.
    Both shapes are accepted so the checkpoint version stays an implementation
    detail of the config file rather than something this project pins.

    `exclusive_speaker_diarization` is preferred because it collapses
    overlapping speech to the dominant speaker instead of reporting two at
    once. pyannote describes it as "adapted to downstream transcription",
    which is exactly this pipeline: `merge.py` has to choose one speaker per
    transcript chunk regardless, so resolving overlaps with the segmentation
    model's own frame-level evidence beats resolving them by overlap
    arithmetic after the fact. It caps the active-speaker count per frame
    rather than deleting those frames, so the timeline stays fully covered.
    """
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, attr, None)
        if annotation is not None:
            return annotation
    return output  # legacy pipelines hand back the Annotation directly


def _as_waveform(samples: np.ndarray) -> torch.Tensor:
    """numpy samples -> the (channel, time) float32 tensor pyannote validates for."""
    waveform = torch.from_numpy(np.ascontiguousarray(samples)).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def diarize(
    samples: np.ndarray,
    sample_rate: int,
    *,
    pipeline_id: str = DEFAULT_PIPELINE,
    token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Return speaker turns for 16 kHz mono `samples`.

    If you know how many people were in the room, pass `num_speakers` — pyannote
    otherwise has to estimate the count, and mis-estimating it is the single
    biggest source of bad speaker labels.
    """
    # Imported lazily: pyannote pulls in a large dependency tree, and users who
    # run with --no-diarize should not pay that import cost.
    Pipeline = _import_pipeline()

    hf_token = _resolve_token(token)

    log.info("Loading diarization pipeline %s", pipeline_id)
    pipeline = Pipeline.from_pretrained(pipeline_id, token=hf_token)
    if pipeline is None:
        raise MissingTokenError(
            f"Could not load {pipeline_id}. This usually means the licence has "
            "not been accepted for this account, or the token lacks read access."
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    constraints = {
        k: v
        for k, v in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        )
        if v is not None
    }

    log.info("Diarizing%s...", f" with {constraints}" if constraints else "")
    output = pipeline(
        {"waveform": _as_waveform(samples), "sample_rate": sample_rate},
        **constraints,
    )
    annotation = _as_annotation(output)

    turns = [
        Turn(start=segment.start, end=segment.end, speaker=speaker)
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start)

    log.info(
        "Found %d turns from %d speakers",
        len(turns),
        len({t.speaker for t in turns}),
    )
    return turns
