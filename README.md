# meeting-minutes

Turns a recording of a Thai meeting into written minutes in Thai. Speech recognition and speaker diarization run locally on an NVIDIA GPU. The summary step sends the merged transcript to Claude through the Claude Agent SDK, using an existing Claude Code login, so it needs no separate API key.

```
audio → ffmpeg (16 kHz mono) → Typhoon Whisper → transcript
                             → pyannote        → who spoke when
                             → merge → Claude  → minutes.md
```

The minutes group the discussion by topic, list decisions apart from the discussion, leave a "not specified" placeholder instead of guessing an owner or a deadline, and give each decision a timestamp you can play back.

## Setup

```bash
sudo apt install ffmpeg
uv sync
```

Speaker diarization needs a Hugging Face token because the pyannote models are gated. Accept the licences for [speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://hf.co/pyannote/segmentation-3.0), create a read token at https://hf.co/settings/tokens, and set `HF_TOKEN`. Without a token the pipeline still runs and skips diarization.

## Usage

```bash
uv run meeting-minutes meeting.m4a --speakers 4 --title "Sprint planning" --date 2026-08-07
uv run meeting-minutes --from-transcript out/transcript.json   # summarize again without transcribing
uv run meeting-minutes meeting.m4a --transcribe-only
uv run meeting-minutes meeting.m4a --no-diarize
```

Output goes to `out/`: the decoded `audio_16k.wav`, `transcript.json` with timestamps and speakers per chunk (the input for `--from-transcript`), a readable `transcript.txt`, and `minutes.md`. `uv run meeting-minutes --help` lists every option.

Most of the quality of the minutes comes from the prompt in `prompts/minutes_th.md`, which is in Thai because the minutes are. After editing it, rerun with `--from-transcript`; that takes seconds.

## Design notes

- Typhoon Whisper is Whisper fine-tuned for Thai, and transformers runs the checkpoint directly. faster-whisper would need a CTranslate2 conversion, and its support for RTX 50-series cards (sm_120) is patchy.
- The audio is cut into 30-second windows, and the window edges become the timestamps. The model's own long-form splitting never predicted a closing timestamp and returned a 44-minute recording as a single chunk.
- pyannote 4.x gets a waveform array instead of a file path, so ffmpeg is the only decoder and the transcript and the speaker turns share one timeline. It also keeps torchcodec, whose CUDA build has to match torch exactly, out of the diarization path.
- On an 8 GB card, word-level timestamps run out of memory at any batch size. The tool tries them once, then keeps 30-second resolution for the rest of the file. If the model itself does not fit, use `--model typhoon-ai/typhoon-whisper-turbo`.
- pyannote labels speakers `SPEAKER_00`, `SPEAKER_01` and so on. Pass `--speakers N` when you know the count, because a wrong guess is the most common cause of mixed-up labels.

[PROGRESS.md](PROGRESS.md) has the full decision log.

## Tests

```bash
uv run pytest
```

The tests need no GPU, audio files or network. They cover the speaker-to-transcript merge, timestamp repair, the transcript file format and prompt assembly.
