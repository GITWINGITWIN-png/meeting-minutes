"""Tests for the parts of the ASR stage that do not need the model.

`_split_into_windows` is where a long recording is cut into the fixed spans
whose *known* boundaries become chunk timestamps — see the note in
transcribe.py about why those boundaries are trusted over the model's own
(unreliable, on this checkpoint) timestamp predictions.
"""

import numpy as np

from meeting_minutes.transcribe import (
    WINDOW_SECONDS,
    Chunk,
    Transcript,
    _clean_text,
    _split_into_windows,
)


class TestSplitIntoWindows:
    def test_exact_multiple_of_window_length(self):
        sample_rate = 16_000
        samples = np.zeros(WINDOW_SECONDS * 2 * sample_rate, dtype=np.float32)

        windows = _split_into_windows(samples, sample_rate)

        assert [start for start, _ in windows] == [0.0, float(WINDOW_SECONDS)]
        assert all(len(piece) == WINDOW_SECONDS * sample_rate for _, piece in windows)

    def test_trailing_partial_window_is_kept_short(self):
        """The last window of a real recording is rarely a clean multiple —
        it must not be padded or dropped."""
        sample_rate = 16_000
        samples = np.zeros(int(WINDOW_SECONDS * 1.5) * sample_rate, dtype=np.float32)

        windows = _split_into_windows(samples, sample_rate)

        assert len(windows) == 2
        assert len(windows[1][1]) == int(WINDOW_SECONDS * 0.5) * sample_rate

    def test_audio_shorter_than_one_window_is_a_single_window(self):
        sample_rate = 16_000
        samples = np.zeros(5 * sample_rate, dtype=np.float32)

        windows = _split_into_windows(samples, sample_rate)

        assert len(windows) == 1
        assert windows[0][0] == 0.0

    def test_start_times_advance_by_window_length_in_seconds(self):
        sample_rate = 16_000
        samples = np.zeros(WINDOW_SECONDS * 3 * sample_rate, dtype=np.float32)

        windows = _split_into_windows(samples, sample_rate)

        assert [start for start, _ in windows] == [0.0, 30.0, 60.0]


class TestCleanText:
    def test_strips_leaked_control_tokens(self):
        """Seen on a real recording: this checkpoint's tokenizer doesn't mark
        its own language/task tokens as special, so they occasionally survive
        into the output text verbatim instead of being decode-stripped."""
        text = _clean_text("ต่อมาคือ<|th|><|transcribe|> ครับ ก็คือโต๊ะ")

        assert "<|" not in text
        assert text == "ต่อมาคือ ครับ ก็คือโต๊ะ"

    def test_ordinary_text_is_unchanged(self):
        assert _clean_text("สวัสดีครับ") == "สวัสดีครับ"

    def test_collapses_the_whitespace_left_behind(self):
        assert _clean_text("ก่อน   <|sv|>   หลัง") == "ก่อน หลัง"


class TestTranscriptText:
    def test_joins_chunks_and_skips_blanks(self):
        transcript = Transcript(
            chunks=[
                Chunk(start=0.0, end=1.0, text=" สวัสดี "),
                Chunk(start=1.0, end=2.0, text="   "),
                Chunk(start=2.0, end=3.0, text="ครับ"),
            ]
        )

        assert transcript.text == "สวัสดี ครับ"
