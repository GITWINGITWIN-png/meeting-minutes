"""Tests for transcript persistence.

`transcript.json` is the seam the whole `--from-transcript` workflow rests on:
transcription is the expensive stage, prompt iteration is the repeated one, and
the file is what lets the second happen without the first. If the round trip
loses a field, the loss shows up as a worse summary rather than as an error.
"""

import json

from meeting_minutes.cli import _load_transcript, _save_transcript
from meeting_minutes.transcribe import Chunk, Transcript


def _sample_transcript() -> Transcript:
    return Transcript(
        chunks=[
            Chunk(start=0.0, end=2.5, text="เปิดประชุมครับ", speaker="SPEAKER_00"),
            Chunk(start=2.5, end=6.0, text="ขอรายงานยอดขาย", speaker="SPEAKER_01"),
        ],
        model="typhoon-ai/typhoon-whisper-large-v3",
        language="th",
    )


class TestTranscriptRoundTrip:
    def test_every_field_survives(self, tmp_path):
        path = tmp_path / "transcript.json"
        original = _sample_transcript()

        _save_transcript(original, path)
        restored = _load_transcript(path)

        assert restored == original

    def test_thai_is_stored_readably_not_escaped(self, tmp_path):
        """`ensure_ascii=False` is deliberate: this file gets read by humans
        when a summary looks wrong, and \\u0e40\\u0e1b\\u0e34\\u0e14 is not readable."""
        path = tmp_path / "transcript.json"

        _save_transcript(_sample_transcript(), path)

        assert "เปิดประชุมครับ" in path.read_text(encoding="utf-8")

    def test_speaker_labels_survive(self, tmp_path):
        """The labels are what `--from-transcript` re-renders the speakers from."""
        path = tmp_path / "transcript.json"

        _save_transcript(_sample_transcript(), path)
        restored = _load_transcript(path)

        assert [c.speaker for c in restored.chunks] == ["SPEAKER_00", "SPEAKER_01"]

    def test_unlabelled_transcripts_round_trip_as_none(self, tmp_path):
        """The --no-diarize output must reload without inventing a speaker."""
        path = tmp_path / "transcript.json"
        _save_transcript(
            Transcript(chunks=[Chunk(start=0.0, end=1.0, text="ครับ")]), path
        )

        restored = _load_transcript(path)

        assert restored.chunks[0].speaker is None

    def test_missing_optional_metadata_uses_defaults(self, tmp_path):
        """Hand-edited or older transcript files should still load."""
        path = tmp_path / "transcript.json"
        path.write_text(
            json.dumps({"chunks": [{"start": 0.0, "end": 1.0, "text": "ก"}]}),
            encoding="utf-8",
        )

        restored = _load_transcript(path)

        assert restored.language == "th"
        assert restored.model == ""
