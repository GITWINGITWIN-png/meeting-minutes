"""Join the two independent views of the audio into one labelled transcript.

ASR gives us *what* was said and when; diarization gives us *who* spoke and
when. Neither knows about the other, so we align them by time: each transcript
chunk is attributed to whichever speaker overlaps it most.

Overlap rather than midpoint-containment matters at speaker changes — a chunk
that straddles a handoff has its midpoint in whichever speaker happens to hold
the middle, while overlap correctly picks whoever holds most of the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio import format_timestamp
from .diarize import Turn
from .transcribe import Chunk, Transcript

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass
class Utterance:
    """Consecutive chunks from one speaker, collapsed into a single line."""

    start: float
    end: float
    speaker: str
    text: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(transcript: Transcript, turns: list[Turn]) -> Transcript:
    """Attach a speaker label to every chunk, in place, and return it."""
    for chunk in transcript.chunks:
        best_speaker, best_overlap = UNKNOWN_SPEAKER, 0.0
        for turn in turns:
            # Turns are time-sorted, so once a turn starts after this chunk
            # ends, no later turn can overlap it either.
            if turn.start >= chunk.end:
                break
            overlap = _overlap(chunk.start, chunk.end, turn.start, turn.end)
            if overlap > best_overlap:
                best_speaker, best_overlap = turn.speaker, overlap
        chunk.speaker = best_speaker
    return transcript


def group_utterances(transcript: Transcript) -> list[Utterance]:
    """Collapse runs of same-speaker chunks into readable utterances.

    Without this the transcript is a wall of 2–5 second fragments, which reads
    badly and wastes tokens repeating the speaker label on every line.
    """
    utterances: list[Utterance] = []
    for chunk in transcript.chunks:
        speaker = chunk.speaker or UNKNOWN_SPEAKER
        if utterances and utterances[-1].speaker == speaker:
            current = utterances[-1]
            current.end = chunk.end
            current.text = f"{current.text} {chunk.text}".strip()
        else:
            utterances.append(
                Utterance(
                    start=chunk.start,
                    end=chunk.end,
                    speaker=speaker,
                    text=chunk.text.strip(),
                )
            )
    return utterances


def render_transcript(
    transcript: Transcript,
    *,
    with_speakers: bool = True,
    speaker_names: dict[str, str] | None = None,
) -> str:
    """Render the transcript as timestamped lines for the summarizer.

    Timestamps are kept so Claude can cite where a decision was made, and so a
    human can jump to that point in the recording to verify it.
    """
    speaker_names = speaker_names or {}

    if not with_speakers:
        return "\n".join(
            f"[{format_timestamp(c.start)}] {c.text.strip()}"
            for c in transcript.chunks
            if c.text.strip()
        )

    lines = []
    for utt in group_utterances(transcript):
        if not utt.text:
            continue
        label = speaker_names.get(utt.speaker, utt.speaker)
        lines.append(f"[{format_timestamp(utt.start)}] {label}: {utt.text}")
    return "\n".join(lines)


def speaker_labels(transcript: Transcript) -> list[str]:
    """Distinct speaker labels, in order of first appearance."""
    seen: list[str] = []
    for chunk in transcript.chunks:
        speaker = chunk.speaker or UNKNOWN_SPEAKER
        if speaker not in seen:
            seen.append(speaker)
    return seen
