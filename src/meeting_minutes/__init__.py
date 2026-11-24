"""Turn Thai meeting audio into structured meeting minutes.

Pipeline stages, each usable on its own:

    audio      decode any input file to 16 kHz mono (the rate both models want)
    transcribe Typhoon/Whisper ASR with timestamps
    diarize    pyannote speaker segmentation ("who spoke when")
    merge      attach a speaker label to every transcript chunk
    summarize  hand the labelled transcript to Claude for the minutes
"""

__version__ = "0.1.0"
