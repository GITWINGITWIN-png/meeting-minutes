"""Tests for the audio helpers that do not shell out to ffmpeg."""

import numpy as np

from meeting_minutes.audio import duration_seconds, format_timestamp


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00"

    def test_seconds_and_minutes(self):
        assert format_timestamp(65) == "00:01:05"

    def test_past_one_hour(self):
        """Meetings routinely run over an hour; the hour field must not wrap."""
        assert format_timestamp(3725) == "01:02:05"

    def test_past_ten_hours_keeps_all_digits(self):
        assert format_timestamp(36_000) == "10:00:00"

    def test_fractional_seconds_truncate(self):
        """Whisper timestamps are floats; the label is whole seconds."""
        assert format_timestamp(59.9) == "00:00:59"


class TestDurationSeconds:
    def test_matches_sample_count_over_rate(self):
        samples = np.zeros(16_000 * 3, dtype=np.float32)

        assert duration_seconds(samples, 16_000) == 3.0
