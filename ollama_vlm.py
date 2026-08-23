"""
ollama_vlm.py

Sends a text question (optionally with an image) to a local Ollama
vision-capable model using the official `ollama` python package, and
returns the text response. Runs SERVER-side (near Ollama / the GPU) — the
client only sends transcribed text + raw audio over /ws/control; the server
does the actual model call and speaks the answer back through the existing
TTS pipeline (audio_engine.StreamingAnnouncer).

Requires Ollama running locally with a vision-capable model pulled:
    ollama pull qwen2.5vl        # or whatever the exact qwen3.5 tag is
    ollama serve                  # usually already running as a service

Check the exact tag with `ollama list` and set MODEL_NAME to match — Ollama
is picky about exact tag names and will error on a mismatch.

Install: pip install ollama
"""

import logging

import ollama

logger = logging.getLogger("ollama_vlm")

MODEL_NAME = "qwen3.5:2b-q4_K_M"  # <-- confirm exact tag via `ollama list` and adjust
OLLAMA_HOST = None  # e.g. "http://192.168.1.50:11434" if Ollama runs on a different machine; None = default localhost

SYSTEM_PROMPT = (
    "You are a concise voice assistant for a visually impaired user wearing "
    "a camera. Answer in 1-2 short spoken sentences. Describe only what's "
    "relevant to the question, don't narrate the whole scene unless asked."
)


class OllamaVisionAssistant:
    def __init__(self, model: str = MODEL_NAME, host: str | None = OLLAMA_HOST):
        self.model = model
        # ollama.Client() defaults to http://localhost:11434 if host=None
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def ask(self, text: str, image_bytes: bytes | None = None) -> str:
        """
        Blocking call — run this via loop.run_in_executor from async code
        (server.py already does this).

        text: the user's transcribed question. May be empty (falls back to
              a generic "describe what you see" prompt).
        image_bytes: raw JPEG/PNG bytes, or None for a text-only question.
              The ollama library accepts raw bytes directly in `images` —
              no manual base64 encoding needed.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        user_msg = {"role": "user", "content": text.strip() if text else "Describe what you see."}
        if image_bytes is not None:
            user_msg["images"] = [image_bytes]
        messages.append(user_msg)

        try:
            response = self.client.chat(model=self.model, messages=messages, stream=False,think=False)
            content = response["message"]["content"].strip()
            return content or "I'm not sure how to answer that."
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
            return "Sorry, I couldn't reach the assistant model."
