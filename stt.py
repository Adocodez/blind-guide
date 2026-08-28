"""
stt.py

SERVER-side speech-to-text using faster-whisper. Takes raw PCM16 mono audio
bytes and returns a transcript.

Install: pip install faster-whisper numpy scipy
"""

import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

INPUT_SAMPLE_RATE = 48000  # Client input sample rate
WHISPER_SAMPLE_RATE = 16000  # Whisper required sample rate


class SpeechToText:
    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cuda",
        compute_type: str = "float16",  # Use 'float16' or 'int8_float16' for CUDA
    ):
        print(f"[stt] loading whisper model '{model_size}' ({device}/{compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=r"E:\AI_Assistant\hf_cache\whisper",
        )

    def transcribe(self, pcm16_bytes: bytes) -> str:
        # Require at least ~200ms of audio at 48kHz (2 bytes per sample)
        if not pcm16_bytes or len(pcm16_bytes) < int(INPUT_SAMPLE_RATE * 0.2) * 2:
            return ""

        # 1. Convert PCM16 raw bytes to float32 NumPy array [-1.0, 1.0]
        audio_48k = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # 2. Resample from 48kHz down to 16kHz (3:1 downsampling)
        audio_16k = resample_poly(audio_48k, WHISPER_SAMPLE_RATE, INPUT_SAMPLE_RATE)

        # 3. Transcribe audio array
        segments, info = self.model.transcribe(
            audio_16k,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Extract text from generator
        text = " ".join(seg.text.strip() for seg in segments).strip()

        print(f"[stt] Transcribed: '{text}'")

        if not text:
            print("[stt] Transcription empty or too short. Using fallback.")
            text = "what is in front of me?"

        return text