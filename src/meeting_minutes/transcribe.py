"""Speech-to-text using a Thai-tuned Whisper checkpoint.

Why not plain `openai/whisper-large-v3`: it handles Thai, but drops and garbles
enough that downstream summaries inherit the errors. The Typhoon and Thonburian
checkpoints are the same architecture fine-tuned on Thai speech, so they are a
drop-in swap for a large accuracy gain.

Why `transformers` rather than `faster-whisper`: faster-whisper runs on
CTranslate2, which (a) would need these checkpoints converted to its own format
first and (b) has had unreliable support for sm_120 (Blackwell / RTX 50-series).
transformers runs the published checkpoints as-is on any CUDA build that
supports the card.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

log = logging.getLogger(__name__)

# This checkpoint's tokenizer config doesn't mark Whisper's control tokens
# (language, task, start-of-transcript, ...) as special, so the pipeline's
# `skip_special_tokens=True` does not strip them. Verified on a real 44-minute
# recording: on about 6% of 30 s windows the model briefly emits one of these
# as literal output text — e.g. "...ต่อมาคือ<|th|><|transcribe|> ครับ..." —
# instead of just the transcribed words. They are never legitimate transcript
# content, so any window that produces one is cleaned up here.
_CONTROL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")


def _clean_text(text: str) -> str:
    stripped = _CONTROL_TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()

# Thai-tuned Whisper checkpoints, best first. All are large-v3 fine-tunes and
# take ~3.1 GB of VRAM in float16.
DEFAULT_MODEL = "typhoon-ai/typhoon-whisper-large-v3"
ALTERNATIVE_MODELS = (
    "typhoon-ai/typhoon-whisper-turbo",          # faster, slightly less accurate
    "biodatlab/whisper-th-large-v3-combined",    # Thonburian Whisper
)


@dataclass
class Chunk:
    """One transcribed span of audio, before speakers are attached."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    chunks: list[Chunk] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    language: str = "th"

    @property
    def text(self) -> str:
        return " ".join(c.text.strip() for c in self.chunks if c.text.strip())


def _select_device() -> tuple[str, torch.dtype]:
    """Prefer fp16 on GPU; fall back to fp32 on CPU where fp16 is slow."""
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    log.warning("No CUDA device found — running on CPU. Expect this to be slow.")
    return "cpu", torch.float32


# Whisper's encoder always processes a fixed 30 s mel-spectrogram window
# (shorter audio is silence-padded to it), so windowing at exactly 30 s gets
# the model's full native context for free; anything smaller only buys finer
# timestamp granularity at a proportional cost in extra encoder passes.
WINDOW_SECONDS = 30


def _split_into_windows(
    samples: np.ndarray, sample_rate: int, window_seconds: int = WINDOW_SECONDS
) -> list[tuple[float, np.ndarray]]:
    """Cut `samples` into fixed-length windows, paired with each one's start time."""
    window_len = window_seconds * sample_rate
    return [
        (start / sample_rate, samples[start : start + window_len])
        for start in range(0, len(samples), window_len)
    ]


def transcribe(
    samples: np.ndarray,
    sample_rate: int,
    *,
    model_id: str = DEFAULT_MODEL,
    language: str = "th",
    batch_size: int = 8,
    word_timestamps: bool = True,
) -> Transcript:
    """Transcribe 16 kHz mono `samples` into timestamped chunks.

    Audio is windowed into fixed WINDOW_SECONDS spans ourselves, rather than
    handed to the pipeline whole with its `chunk_length_s` auto-splitter.
    transformers documents that splitter as "very experimental" for seq2seq
    models, and it earns that label here: on this Typhoon checkpoint it
    reliably fails to predict *any* internal closing timestamp across a long
    recording, so the auto-splitter's boundary-detection (which looks for a
    pair of adjacent timestamp tokens) never fires — a 44-minute meeting came
    back as one 2622-second "chunk" with useless timestamps and, if speakers
    are attached later, everyone in the room collapsed onto whichever speaker
    held the single largest turn.

    Windowing here sidesteps that: each window's start/end is known before the
    model ever runs, so the chunk's timestamp comes from arithmetic on the
    audio, not from a timestamp token this checkpoint won't reliably emit.
    """
    device, dtype = _select_device()
    total_duration = len(samples) / sample_rate

    log.info("Loading ASR model %s on %s (%s)", model_id, device, dtype)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        # transformers v5 renamed `torch_dtype` to `dtype`. The old name still
        # works on the model loader but only warns; `pipeline()` below dropped
        # it entirely and would silently swallow it via **kwargs.
        dtype=dtype,
        low_cpu_mem_usage=True,
        # Scaled dot-product attention is built into torch and needs no extra
        # dependency, unlike flash-attn which must be compiled per CUDA version.
        attn_implementation="sdpa",
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # No `chunk_length_s`: every input handed to this pipeline is already
    # <= WINDOW_SECONDS, so each call uses Whisper's one native context window
    # instead of the pipeline's own long-form splitter.
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
    )

    generate_kwargs = {"language": language, "task": "transcribe"}
    windows = _split_into_windows(samples, sample_rate)
    log.info("Transcribing %d window(s) of %ds...", len(windows), WINDOW_SECONDS)

    # Word-level timestamps come from cross-attention alignment, which needs
    # `alignment_heads` in the checkpoint's generation config and a good deal
    # of extra VRAM. Whether it fits is a property of the card, not of any one
    # window's content, so a single failure here is treated as permanent for
    # the rest of the file instead of being retried per window.
    attempt_word_timestamps = word_timestamps

    chunks: list[Chunk] = []
    for start_time, piece in windows:
        if piece.size == 0:
            continue
        end_time = min(start_time + WINDOW_SECONDS, total_duration)

        modes = ["word", True] if attempt_word_timestamps else [True]
        result = None
        for mode in modes:
            try:
                result = asr(
                    {"raw": piece, "sampling_rate": sample_rate},
                    return_timestamps=mode,
                    generate_kwargs=generate_kwargs,
                )
                break
            except (ValueError, KeyError, RuntimeError) as exc:
                if mode == modes[-1]:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.warning(
                    "Word-level timestamps unavailable (%s); falling back to "
                    "window-level for the rest of this file.", exc,
                )
                attempt_word_timestamps = False

        assert result is not None
        if mode == "word":
            # The model's own word timestamps are trustworthy here since this
            # attempt succeeded; use them instead of the coarse window span.
            for word in result.get("chunks") or []:
                text = _clean_text(word.get("text") or "")
                if not text:
                    continue
                w_start, w_end = word.get("timestamp", (None, None))
                chunks.append(
                    Chunk(
                        start=start_time + (w_start if w_start is not None else 0.0),
                        end=start_time + (w_end if w_end is not None else (end_time - start_time)),
                        text=text,
                    )
                )
        else:
            text = _clean_text(result.get("text") or "")
            if text:
                chunks.append(Chunk(start=start_time, end=end_time, text=text))

    log.info("Transcribed %d chunks covering %.1f s", len(chunks), total_duration)
    return Transcript(chunks=chunks, model=model_id, language=language)
