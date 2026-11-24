"""Tests for the ASR-to-diarization join.

This is the one stage with no model in it — pure arithmetic over two timelines.
That makes it both the easiest thing to test and the most valuable: when speaker
labels come out wrong, these tests tell you whether the join is at fault or
whether the models upstream simply disagreed.
"""

from meeting_minutes.diarize import Turn
from meeting_minutes.merge import (
    UNKNOWN_SPEAKER,
    assign_speakers,
    group_utterances,
    render_transcript,
    speaker_labels,
)
from meeting_minutes.transcribe import Chunk, Transcript


def _transcript(*chunks: Chunk) -> Transcript:
    return Transcript(chunks=list(chunks))


class TestAssignSpeakers:
    def test_picks_the_speaker_holding_most_of_the_chunk(self):
        """The case the module docstring is written around.

        A chunk spanning 0-10 s whose midpoint (5 s) falls in B's turn, but of
        which A holds 6 s and B only 4 s. Midpoint-containment would answer B;
        overlap correctly answers A.
        """
        transcript = _transcript(Chunk(start=0.0, end=10.0, text="สวัสดีครับ"))
        turns = [
            Turn(start=0.0, end=6.0, speaker="A"),
            Turn(start=6.0, end=12.0, speaker="B"),
        ]

        assign_speakers(transcript, turns)

        assert transcript.chunks[0].speaker == "A"

    def test_chunk_with_no_overlapping_turn_is_unknown(self):
        """Silence-gap chunks must not be silently attributed to someone."""
        transcript = _transcript(Chunk(start=100.0, end=105.0, text="ครับ"))
        turns = [Turn(start=0.0, end=10.0, speaker="A")]

        assign_speakers(transcript, turns)

        assert transcript.chunks[0].speaker == UNKNOWN_SPEAKER

    def test_zero_length_overlap_does_not_count(self):
        """A turn that merely touches the chunk boundary shares no audio."""
        transcript = _transcript(Chunk(start=10.0, end=20.0, text="ทดสอบ"))
        turns = [Turn(start=0.0, end=10.0, speaker="A")]

        assign_speakers(transcript, turns)

        assert transcript.chunks[0].speaker == UNKNOWN_SPEAKER

    def test_later_turns_are_still_found(self):
        """Guards the early `break` in the scan.

        The loop stops at the first turn starting at or after the chunk's end.
        If that condition were ever loosened to `>` on the wrong field, the
        last speaker in a recording would stop being matched.
        """
        transcript = _transcript(Chunk(start=50.0, end=60.0, text="ปิดประชุม"))
        turns = [
            Turn(start=0.0, end=10.0, speaker="A"),
            Turn(start=10.0, end=20.0, speaker="B"),
            Turn(start=48.0, end=62.0, speaker="C"),
        ]

        assign_speakers(transcript, turns)

        assert transcript.chunks[0].speaker == "C"

    def test_returns_the_same_transcript_object(self):
        """Documented as in-place; callers in cli.py rely on that."""
        transcript = _transcript(Chunk(start=0.0, end=1.0, text="ก"))

        assert assign_speakers(transcript, []) is transcript


class TestGroupUtterances:
    def test_consecutive_same_speaker_chunks_collapse(self):
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="สวัสดี", speaker="A"),
            Chunk(start=2.0, end=4.0, text="ครับ", speaker="A"),
            Chunk(start=4.0, end=6.0, text="เชิญ", speaker="B"),
        )

        utterances = group_utterances(transcript)

        assert len(utterances) == 2
        assert utterances[0].speaker == "A"
        assert utterances[0].text == "สวัสดี ครับ"
        assert utterances[0].start == 0.0
        assert utterances[0].end == 4.0  # extended to cover both chunks
        assert utterances[1].text == "เชิญ"

    def test_speaker_returning_later_starts_a_new_utterance(self):
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="หนึ่ง", speaker="A"),
            Chunk(start=2.0, end=4.0, text="สอง", speaker="B"),
            Chunk(start=4.0, end=6.0, text="สาม", speaker="A"),
        )

        utterances = group_utterances(transcript)

        assert [u.speaker for u in utterances] == ["A", "B", "A"]

    def test_unlabelled_chunks_group_under_unknown(self):
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="หนึ่ง"),
            Chunk(start=2.0, end=4.0, text="สอง"),
        )

        utterances = group_utterances(transcript)

        assert len(utterances) == 1
        assert utterances[0].speaker == UNKNOWN_SPEAKER


class TestRenderTranscript:
    def test_speaker_lines_carry_timestamps(self):
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="เปิดประชุม", speaker="SPEAKER_00"),
            Chunk(start=3725.0, end=3730.0, text="ปิดประชุม", speaker="SPEAKER_01"),
        )

        rendered = render_transcript(transcript)

        assert rendered.splitlines() == [
            "[00:00:00] SPEAKER_00: เปิดประชุม",
            "[01:02:05] SPEAKER_01: ปิดประชุม",
        ]

    def test_without_speakers_falls_back_to_bare_chunks(self):
        """The --no-diarize / missing-token path must still be readable."""
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="สวัสดี"),
            Chunk(start=2.0, end=4.0, text="ครับ"),
        )

        rendered = render_transcript(transcript, with_speakers=False)

        assert rendered.splitlines() == ["[00:00:00] สวัสดี", "[00:00:02] ครับ"]

    def test_speaker_names_are_substituted(self):
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="สวัสดี", speaker="SPEAKER_00")
        )

        rendered = render_transcript(
            transcript, speaker_names={"SPEAKER_00": "คุณสมชาย"}
        )

        assert rendered == "[00:00:00] คุณสมชาย: สวัสดี"

    def test_blank_chunks_are_dropped(self):
        """Whisper emits empty spans over silence; they add nothing but tokens."""
        transcript = _transcript(
            Chunk(start=0.0, end=2.0, text="   ", speaker="A"),
            Chunk(start=2.0, end=4.0, text="จริง", speaker="B"),
        )

        assert render_transcript(transcript) == "[00:00:02] B: จริง"


class TestSpeakerLabels:
    def test_ordered_by_first_appearance_not_alphabetically(self):
        transcript = _transcript(
            Chunk(start=0.0, end=1.0, text="ก", speaker="SPEAKER_02"),
            Chunk(start=1.0, end=2.0, text="ข", speaker="SPEAKER_00"),
            Chunk(start=2.0, end=3.0, text="ค", speaker="SPEAKER_02"),
        )

        assert speaker_labels(transcript) == ["SPEAKER_02", "SPEAKER_00"]
