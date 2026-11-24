# Progress

Thai meeting audio to Thai meeting minutes. Started 7 August 2026.

## Machine constraints

- GPU: RTX 5060 Laptop with 8 GB of VRAM, a Blackwell card (sm_120), so torch has to be a build with sm_120 kernels.
- Driver 580.173.02. `nvcc` is not needed, because the torch wheels ship the CUDA runtime.
- The system Python is 3.14. The project pins 3.12 in its own venv, since the ML packages had not caught up with 3.14.
- `uv` manages the project.

## Milestones

- [x] Survey the machine and pick the stack: transformers, Typhoon Whisper, pyannote and the Claude Agent SDK.
- [x] Confirm the Agent SDK API from the installed package source (v0.2.132): `query(prompt=..., options=ClaudeAgentOptions(...))`, with `tools=[]` and `setting_sources=[]` for isolation.
- [x] Write the five pipeline stages: audio, transcribe, diarize, merge and summarize.
- [x] Write the Thai minutes prompt.
- [x] Write the CLI and the README.
- [x] Install the dependencies with `uv sync`: torch 2.11.0 sees CUDA and reports the card as sm_120.
- [x] Install ffmpeg 8.0.1.
- [x] Fix pyannote failing at import (torchaudio 2.11 removed APIs that pyannote 3.x uses) by moving to 4.x.
- [x] Avoid torchcodec's CUDA 13 build by passing the waveform straight into the pipeline.
- [x] Write 47 offline tests (`uv run pytest`) for merging, timestamps, the transcript round trip and prompt assembly.
- [x] Move torch, torchaudio and torchcodec to the PyPI defaults (CUDA 13). The cu128 index stopped publishing torchcodec after 0.11.1, and the CUDA mismatch broke transformers' torchcodec import during real transcription, not only during diarization.
- [x] Fix a serious bug that collapsed a 44-minute recording into one chunk (see below).
- [x] Test on a real 43.8-minute project presentation: transcription time measured, 30-second timestamps correct.
- [ ] Set `HF_TOKEN` and accept the pyannote licences, which diarization needs. This is a manual step.
- [ ] Tune the prompt on real output. The first round was already good (see below).

## Decisions

### Why not faster-whisper

It was the first choice, but CTranslate2, the backend of faster-whisper, does not yet support sm_120 reliably, and the Typhoon checkpoint would need converting to the CT2 format first. transformers runs the published checkpoint as it is, at the cost of being about 30 to 40% slower.

### Why transcript.json is kept

Transcription is the expensive step, and the prompt is what changes most often. Keeping the intermediate file lets `--from-transcript` rerun the summary in seconds.

### Why pyannote moved to 4.x despite the old `<4` pin

The pin said the 3.1 pipeline needed the 3.x API, but pyannote 3.4 would not even import: it references `torchaudio.AudioMetaData` and `torchaudio.info` at module level, and torchaudio removed both in 2.9. The choice was to drop torch below the build this card needs, which would make the card unusable, or to move pyannote up. The pin meant to keep things safe was itself the problem, because it was written from an assumption instead of from an install.

### Why pyannote gets a waveform instead of a file path

pyannote 4.x decodes files with torchcodec, whose PyPI wheel links `libcudart.so.13`, a different CUDA runtime from the torch build used at the time, and the cu128 index only had torchcodec up to 0.11.1. Matching torchcodec, torch and FFmpeg versions by hand would break again on every upgrade. pyannote accepts `{"waveform", "sample_rate"}` directly, which skips its decode layer entirely; torchcodec appears only in `core/io.py`, behind the file-path branch. As a result ffmpeg is the only decoder in the system, so the shared timeline that `merge.py` relies on holds by construction, not because two resamplers happen to agree.

### Why the cu128 index was removed instead of pinning cu130

The PyPI metadata for torch 2.11.0 already pulls in `nvidia-*-cu13`, so the PyPI default is CUDA 13 and no custom index is needed. Pinning one by hand would mean checking, on every upgrade, whether that index still has a matching torchcodec. With every wheel from PyPI, the CUDA versions match by construction. The block added to prevent problems became one once the ecosystem moved, so check the real packages instead of trusting an old comment.

### Bug: a 44-minute recording transcribed as one chunk

The first run on a real 43.8-minute recording produced a `transcript.json` with a single chunk covering 0 to 2622 seconds, 32,279 characters long, so every timestamp in the minutes read [00:00:00]. With diarization on it would have been worse: `merge.py` assigns speakers per chunk, not per word, so one chunk would have merged everyone in the room into one speaker.

Cause: `transcribe.py` passed the whole file to `pipeline(chunk_length_s=30)` and let transformers split it and join it back (long-form auto-split). transformers itself warns that this is very experimental for seq2seq models and suggests calling `generate()` directly. This checkpoint never predicted a closing timestamp token anywhere in the file, so the boundary detection, which looks for consecutive timestamp tokens, never fired and everything was joined into one chunk.

Fix: cut the audio into 30-second windows (`_split_into_windows`) and use the known window edges as the timestamps instead of waiting for the model to predict them. Thirty seconds matches the fixed 30-second mel frame that Whisper's encoder always processes, padding shorter audio with silence. Smaller windows would only give finer timestamps and multiply the run time by 30 divided by the window length, because the encoder costs the same per call however short the audio is. The "chunk_length_s is very experimental" warning went away too, since that path is no longer used.

### Word-level timestamps do not fit on this card at any batch size

Tested on its own: even a single 30-second window, with no build-up from a long file, runs out of memory on the 8 GB card when word-level timestamps are on. The cross-attention alignment matrix kept for word alignment is too large for this card. The limit comes from the hardware, not the audio or the batch size; `--batch-size` barely changed the VRAM in use at the point of failure. The old README advice to lower `--batch-size` on out-of-memory errors therefore did not apply to this problem. The tool now tries word level once and turns it off for the rest of the file, instead of retrying, and failing, on every window.

### Quality of the first real run

On a real project presentation, in Thai mixed with English technical terms such as PostgreSQL, MongoDB and Prisma ORM, the summary was accurate and correctly grouped by topic, and it said plainly that there were no decisions, because the session was all questions and answers. Before the chunking bug was fixed, Claude also flagged the timestamp problem in a note at the end instead of inventing timestamps, so the prompt's rule that every decision needs a timestamp worked as intended.

### Why summarize uses `tools=[]` and `setting_sources=[]`

Summarizing is pure text transformation and needs no tools, and a CLAUDE.md in whatever directory it runs from should not silently change the style of the output. The output then depends only on the prompt and the transcript.
