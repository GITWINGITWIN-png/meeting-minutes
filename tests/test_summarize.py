"""Tests for the prompt assembly, which is the half of summarization that is
deterministic. The Claude call itself is not tested here — it needs the CLI
login and its output is not a fixed string.
"""

from meeting_minutes.summarize import DEFAULT_PROMPT_PATH, build_user_message, load_prompt


class TestBuildUserMessage:
    def test_metadata_comes_before_the_transcript(self):
        """Framing first: the model should know what it is reading before it
        starts on a body of text that may be hours long."""
        message = build_user_message(
            "[00:00:00] SPEAKER_00: สวัสดี",
            title="Sprint planning",
            audio_duration="01:02:05",
            speakers=["SPEAKER_00"],
            meeting_date="2026-08-07",
        )

        assert message.index("Sprint planning") < message.index("## Transcript")

    def test_absent_metadata_becomes_an_explicit_placeholder(self):
        """Never leave a field blank — the prompt forbids guessing, so the
        model has to be able to see that the value is genuinely unknown."""
        message = build_user_message(
            "…", title=None, audio_duration=None, speakers=None, meeting_date=None
        )

        assert "- หัวข้อ: [ยังไม่ระบุ]" in message
        assert "- วันที่: [ยังไม่ระบุ]" in message

    def test_no_speakers_says_so_rather_than_staying_silent(self):
        """Otherwise the model reads an unlabelled transcript as a monologue."""
        message = build_user_message(
            "…", title=None, audio_duration=None, speakers=None, meeting_date=None
        )

        assert "ไม่ได้แยกผู้พูด" in message

    def test_detected_speakers_are_listed(self):
        message = build_user_message(
            "…",
            title=None,
            audio_duration=None,
            speakers=["SPEAKER_00", "SPEAKER_01"],
            meeting_date=None,
        )

        assert "SPEAKER_00, SPEAKER_01" in message

    def test_transcript_body_is_included_verbatim(self):
        body = "[00:00:00] SPEAKER_00: ขอเสนอให้เลื่อนกำหนดส่ง"

        message = build_user_message(
            body, title=None, audio_duration=None, speakers=None, meeting_date=None
        )

        assert body in message


class TestLoadPrompt:
    def test_the_packaged_default_prompt_resolves(self):
        """`DEFAULT_PROMPT_PATH` is built by walking up from this module's
        location, so it breaks silently if the package layout moves."""
        assert DEFAULT_PROMPT_PATH.is_file()
        assert load_prompt().strip()
