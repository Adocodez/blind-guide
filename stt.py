"""
stt.py

SERVER-side speech-to-text using faster-whisper. Takes raw PCM16 mono audio
bytes (as sent by the client over /ws/control after a push-to-talk
recording) and returns a transcript. Runs on the server so all the AI
pieces — detection, STT, the VLM — live in one place; the client is just
mic/camera/speaker I/O plus hotkeys.

Install: pip install faster-whisper numpy

First run downloads the whisper model weights (a few hundred MB depending
on model_size) and caches them locally.
"""

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # client must record + send audio at this rate, mono, int16


class SpeechToText:
    def __init__(self, model_size: str = "small.en", device: str = "cuda", compute_type: str = "int8"):
        """
        model_size: "tiny.en" = fastest/lowest accuracy, "base.en" = good
        default balance, "small.en"/"medium.en" = better accuracy, slower.
        Set device="cuda", compute_type="float16" if the server has a spare
        GPU for whisper (separate from whatever YOLO/Ollama are using).
        """
        print(f"[stt] loading whisper model '{model_size}' ({device}/{compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type,download_root=r"E:\AI_Assistant\hf_cache\whisper")

    def transcribe(self, pcm16_bytes: bytes) -> str:
        """
        pcm16_bytes: raw mono PCM16 audio at SAMPLE_RATE (16kHz), exactly as
        sent by the client after a push-to-talk recording. Returns "" for
        empty/too-short audio (e.g. an accidental tap) instead of erroring.
        """
        if not pcm16_bytes or len(pcm16_bytes) < int(SAMPLE_RATE * 0.2) * 2:  # ~200ms of int16 samples
            return ""

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
