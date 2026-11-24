"""Command-line entry point wiring the pipeline together.

    meeting-minutes recording.m4a --speakers 4 --title "Sprint planning"

Every stage writes its output to disk before the next one starts. That is
deliberate: transcription is the slow, expensive step, and prompt iteration is
the step you actually want to repeat. Once `transcript.json` exists you can
re-run summarization against it in seconds:

    meeting-minutes --from-transcript out/transcript.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

from . import audio, merge, summarize as summarize_mod, transcribe as transcribe_mod
from .transcribe import Chunk, Transcript

log = logging.getLogger("meeting_minutes")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-minutes",
        description="Turn meeting audio into Thai meeting minutes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "audio", nargs="?", type=Path,
        help="Meeting recording (m4a, mp3, wav, mp4, ...).",
    )
    parser.add_argument(
        "--from-transcript", type=Path, metavar="PATH",
        help="Skip ASR and diarization; re-summarize an existing transcript.json.",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("out"),
        help="Output directory (default: ./out).",
    )

    asr = parser.add_argument_group("transcription")
    asr.add_argument(
        "--model", default=transcribe_mod.DEFAULT_MODEL,
        help=f"ASR checkpoint (default: {transcribe_mod.DEFAULT_MODEL}). "
             f"Alternatives: {', '.join(transcribe_mod.ALTERNATIVE_MODELS)}",
    )
    asr.add_argument("--language", default="th", help="Spoken language (default: th).")
    asr.add_argument(
        "--batch-size", type=int, default=8,
        help="ASR batch size. Lower this if you hit CUDA out-of-memory (default: 8).",
    )
    asr.add_argument(
        "--no-word-timestamps", action="store_true",
        help="Use chunk-level timestamps only; skips the word-alignment attempt.",
    )

    dia = parser.add_argument_group("speaker diarization")
    dia.add_argument("--no-diarize", action="store_true", help="Skip speaker separation.")
    dia.add_argument(
        "--speakers", type=int, metavar="N",
        help="Exact number of speakers. Set this when you know it — guessing "
             "the count wrong is the main cause of bad speaker labels.",
    )
    dia.add_argument("--min-speakers", type=int, help="Lower bound on speaker count.")
    dia.add_argument("--max-speakers", type=int, help="Upper bound on speaker count.")
    dia.add_argument("--hf-token", help="Hugging Face token (else read from $HF_TOKEN).")

    summ = parser.add_argument_group("summarization")
    summ.add_argument("--title", help="Meeting title, passed to the summarizer.")
    summ.add_argument("--date", dest="meeting_date", help="Meeting date, e.g. 2026-08-07.")
    summ.add_argument(
        "--prompt", type=Path,
        help="Custom prompt template (default: prompts/minutes_th.md).",
    )
    summ.add_argument("--claude-model", help="Override the Claude model used.")
    summ.add_argument(
        "--transcribe-only", action="store_true",
        help="Stop after the transcript; do not generate minutes.",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def _save_transcript(transcript: Transcript, path: Path) -> None:
    path.write_text(
        json.dumps(dataclasses.asdict(transcript), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_transcript(path: Path) -> Transcript:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(
        chunks=[Chunk(**c) for c in raw["chunks"]],
        model=raw.get("model", ""),
        language=raw.get("language", "th"),
    )


def _run_pipeline(args: argparse.Namespace) -> int:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = out_dir / "transcript.json"
    duration_label: str | None = None

    if args.from_transcript:
        log.info("Loading transcript from %s", args.from_transcript)
        transcript = _load_transcript(args.from_transcript)
    else:
        # --- decode ------------------------------------------------------
        wav_path = out_dir / "audio_16k.wav"
        log.info("Decoding %s -> %s", args.audio, wav_path)
        audio.decode_to_wav(args.audio, wav_path)
        samples, sample_rate = audio.load_wav(wav_path)
        total_seconds = audio.duration_seconds(samples, sample_rate)
        duration_label = audio.format_timestamp(total_seconds)
        log.info("Audio duration: %s", duration_label)

        # --- transcribe --------------------------------------------------
        started = time.monotonic()
        transcript = transcribe_mod.transcribe(
            samples,
            sample_rate,
            model_id=args.model,
            language=args.language,
            batch_size=args.batch_size,
            word_timestamps=not args.no_word_timestamps,
        )
        log.info("Transcription took %.1f s", time.monotonic() - started)

        # --- diarize + merge ---------------------------------------------
        if not args.no_diarize:
            from . import diarize as diarize_mod

            try:
                # Same samples the ASR just saw, not a second decode of the
                # wav — that shared time axis is what makes the merge valid.
                turns = diarize_mod.diarize(
                    samples,
                    sample_rate,
                    token=args.hf_token,
                    num_speakers=args.speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                merge.assign_speakers(transcript, turns)
            except diarize_mod.MissingTokenError as exc:
                # Not fatal: an unlabelled transcript is still a usable
                # transcript, and forcing a re-run of ASR to recover would
                # waste the expensive step we just paid for.
                log.warning("Skipping diarization.\n%s", exc)

        _save_transcript(transcript, transcript_path)
        log.info("Wrote %s", transcript_path)

    # --- render ----------------------------------------------------------
    speakers = merge.speaker_labels(transcript)
    has_speakers = any(c.speaker for c in transcript.chunks)
    rendered = merge.render_transcript(transcript, with_speakers=has_speakers)

    transcript_txt = out_dir / "transcript.txt"
    transcript_txt.write_text(rendered, encoding="utf-8")
    log.info("Wrote %s", transcript_txt)

    if args.transcribe_only:
        print(f"\nTranscript: {transcript_txt}")
        return 0

    # --- summarize -------------------------------------------------------
    minutes = asyncio.run(
        summarize_mod.summarize(
            rendered,
            title=args.title,
            audio_duration=duration_label,
            speakers=speakers if has_speakers else None,
            meeting_date=args.meeting_date,
            prompt_path=args.prompt,
            model=args.claude_model,
        )
    )

    minutes_path = out_dir / "minutes.md"
    minutes_path.write_text(minutes.markdown, encoding="utf-8")

    print(f"\nTranscript: {transcript_txt}")
    print(f"Minutes:    {minutes_path}")
    if minutes.cost_usd is not None:
        print(f"Cost:       ${minutes.cost_usd:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.audio and not args.from_transcript:
        parser.error("provide an audio file, or --from-transcript to reuse an existing one")
    if args.audio and not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")

    try:
        return _run_pipeline(args)
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't traceback
        log.error("%s", exc)
        if args.verbose:
            raise
        log.error("Re-run with --verbose for the full traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
