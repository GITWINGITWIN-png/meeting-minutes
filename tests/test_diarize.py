"""Tests for the diarization stage's adapters.

The pipeline itself needs a gated model and a GPU, so it is not exercised here.
What *is* exercised is the two places this project has to absorb pyannote's API:
the shape of the waveform going in, and the shape of the result coming out.
Both changed between pyannote 3.x and 4.x, and both fail confusingly when wrong.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from meeting_minutes.diarize import (
    MissingTokenError,
    _as_annotation,
    _as_waveform,
    _resolve_token,
)


class TestAsWaveform:
    def test_mono_samples_gain_a_channel_axis(self):
        """pyannote validates for (channel, time) and rejects a bare 1-D array."""
        waveform = _as_waveform(np.zeros(16_000, dtype=np.float32))

        assert tuple(waveform.shape) == (1, 16_000)

    def test_output_is_float32(self):
        """soundfile can hand back float64; the models want float32."""
        waveform = _as_waveform(np.zeros(1_000, dtype=np.float64))

        assert waveform.dtype.itemsize == 4

    def test_channel_axis_stays_shorter_than_time_axis(self):
        """pyannote rejects a waveform whose channel count exceeds its length.

        A short clip is the case that would trip it, so check the orientation
        survives one that is only a few samples long.
        """
        waveform = _as_waveform(np.zeros(4, dtype=np.float32))

        assert waveform.shape[0] <= waveform.shape[1]


@dataclass
class _FakeAnnotation:
    name: str


class TestAsAnnotation:
    def test_prefers_the_overlap_free_annotation(self):
        """pyannote 4.x: `merge.py` must pick one speaker per chunk anyway."""

        @dataclass
        class Output:
            speaker_diarization: _FakeAnnotation
            exclusive_speaker_diarization: _FakeAnnotation

        output = Output(
            speaker_diarization=_FakeAnnotation("overlapping"),
            exclusive_speaker_diarization=_FakeAnnotation("exclusive"),
        )

        assert _as_annotation(output).name == "exclusive"

    def test_falls_back_to_the_plain_annotation(self):
        """Not every 4.x pipeline exposes an exclusive variant."""

        @dataclass
        class Output:
            speaker_diarization: _FakeAnnotation

        output = Output(speaker_diarization=_FakeAnnotation("overlapping"))

        assert _as_annotation(output).name == "overlapping"

    def test_legacy_pipelines_return_the_annotation_directly(self):
        """pyannote 3.x, and 4.x pipelines loaded with `legacy: true`."""
        annotation = _FakeAnnotation("bare")

        assert _as_annotation(annotation) is annotation


class TestResolveToken:
    def test_explicit_token_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")

        assert _resolve_token("hf_explicit") == "hf_explicit"

    def test_falls_back_through_the_known_variable_names(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_third_choice")

        assert _resolve_token() == "hf_third_choice"

    def test_no_token_anywhere_raises_the_actionable_error(self, monkeypatch):
        """The message is the user's only instruction sheet — keep it in it."""
        for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(MissingTokenError) as excinfo:
            _resolve_token()

        assert "--no-diarize" in str(excinfo.value)
