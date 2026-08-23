"""
server.py

Blind-Guide server: video streams in over a websocket, audio streams out over
another websocket. Detection runs in a thread pool so the async event loop
stays free to handle multiple socket connections concurrently.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
    ws://<host>:8000/ws/video   client -> server: binary JPEG frames
    ws://<host>:8000/ws/audio   server -> client: binary PCM16 mono @ 24kHz

WHERE GEMINI PLUGS IN LATER:
    - Audio-in: add a ws/ws_audio_in.py-style route that receives mic audio
      from the client the same way /ws/video receives frames, buffers it into
      utterances (e.g. via VAD), and sends it to the Gemini Live API. Gemini's
      text/audio response can then either be spoken via `announcer.say(...)`
      (reusing the exact same audio-out pipeline already wired up here) or,
      if Gemini returns audio directly, broadcast through
      `announcer._broadcast()` the same way TTS chunks are.
    - Scene understanding: process_frame() in pipeline.py is a natural spot
      to occasionally hand a frame to Gemini (e.g. every N seconds, or on
      request from the audio-in flow: "what's in front of me?") for a richer
      description than the fixed YOLO-World class list gives you.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pipeline import DetectionPipeline
from audio_engine import StreamingAnnouncer
from ollama_vlm import OllamaVisionAssistant
from stt import SpeechToText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()

# Single shared pipeline + announcer for now (one user, one earpiece).
# If you ever need multiple independent users, key these dicts by a
# session/client id instead of using module-level globals.
pipeline: DetectionPipeline | None = None
announcer: StreamingAnnouncer | None = None
vlm: OllamaVisionAssistant | None = None
stt_engine: SpeechToText | None = None

# Toggled by the "toggle_detection" control command. When False, /ws/video
# still receives frames (so latest_frame stays fresh for assistant queries)
# but skips running YOLO / speaking detection alerts.
detection_enabled = True

# Raw JPEG bytes of the most recent frame received on /ws/video. Assistant
# queries use this directly instead of the client sending a separate image —
# the server already sees every frame, no need to duplicate that traffic.
latest_frame: bytes | None = None

# YOLO inference, whisper transcription, and the Ollama HTTP call are all
# blocking — run them on worker threads so they don't tie up the event loop.
executor = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
async def startup():
    global pipeline, announcer, vlm, stt_engine
    loop = asyncio.get_event_loop()
    logger.info("Loading detection pipeline...")
    pipeline = DetectionPipeline()
    logger.info("Starting TTS announcer...")
    announcer = StreamingAnnouncer(voice="af_heart", loop=loop)
    logger.info("Loading Whisper STT model...")
    stt_engine = SpeechToText()
    logger.info("Connecting to Ollama vision model...")
    vlm = OllamaVisionAssistant()
    logger.info("Ready.")


@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    """
    Client streams binary JPEG frames. Each frame is decoded, run through the
    detection pipeline (in a thread, so we don't block the event loop), and
    any resulting announcements are spoken via the shared announcer — which
    pushes audio out to whoever is connected on /ws/audio.
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    logger.info("video client connected")

    global latest_frame

    try:
        while True:
            data = await websocket.receive_bytes()
            latest_frame = data  # cache raw JPEG for assistant queries, regardless of detection toggle

            if not detection_enabled:
                continue

            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            announcements = await loop.run_in_executor(executor, pipeline.process_frame, frame)

            for ann in announcements:
                await announcer.say(ann["text"], urgent=ann["urgent"])

    except WebSocketDisconnect:
        logger.info("video client disconnected")


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    """
    Server pushes binary PCM16 mono @ 24kHz audio chunks to the client
    whenever the pipeline has something to say. This connection is
    receive-only from the client's perspective right now; audio-in (mic ->
    Gemini) will be a separate endpoint so the two directions stay decoupled.
    """
    await websocket.accept()
    announcer.register_client(websocket)

    try:
        while True:
            # Nothing expected from the client on this socket today; just
            # keep the connection open. Replace with real handling once
            # audio-in / control messages are added.
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        announcer.unregister_client(websocket)


@app.websocket("/ws/control")
async def control_ws(websocket: WebSocket):
    """
    Client -> server messages on this one connection, two kinds:

    TEXT (JSON):
      {"cmd": "toggle_detection", "value": true|false}
          Turns YOLO detection + its announcements on/off.

      {"cmd": "interrupt"}
          Immediately stops whatever the TTS is currently saying. Sent right
          when the push-to-talk key goes down, so detection chatter doesn't
          talk over the user starting a question.

      {"cmd": "assistant_audio_query"}
          Announces that the client's NEXT message on this connection is
          raw audio for a question. No fields needed — the server uses its
          own cached `latest_frame` (from /ws/video) as the image.

      {"cmd": "assistant_query", "text": "..."}
          Text-only fallback path (no audio, no STT) — still uses the
          cached latest_frame if one is available.

    BINARY:
      Raw PCM16 mono audio @ stt.SAMPLE_RATE (16kHz), sent immediately after
      an "assistant_audio_query" text message. Transcribed with Whisper,
      then answered via Ollama + spoken through the normal announcer.
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    logger.info("control client connected")

    global detection_enabled
    awaiting_audio = False

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            text_payload = message.get("text")
            bytes_payload = message.get("bytes")

            if text_payload is not None:
                try:
                    msg = json.loads(text_payload)
                except json.JSONDecodeError:
                    logger.warning("control: bad JSON, ignoring: %r", text_payload)
                    continue

                cmd = msg.get("cmd")

                if cmd == "toggle_detection":
                    detection_enabled = bool(msg.get("value", not detection_enabled))
                    logger.info("detection_enabled -> %s", detection_enabled)

                elif cmd == "interrupt":
                    announcer.stream.stop()

                elif cmd == "assistant_audio_query":
                    awaiting_audio = True

                elif cmd == "assistant_query":
                    answer = await loop.run_in_executor(executor, vlm.ask, msg.get("text", ""), latest_frame)
                    logger.info("assistant Q: %r -> A: %r", msg.get("text"), answer)
                    await announcer.say(answer, urgent=True)

                else:
                    logger.warning("control: unknown cmd %r", cmd)

            elif bytes_payload is not None:
                if not awaiting_audio:
                    logger.warning("control: got audio with no pending assistant_audio_query, ignoring")
                    continue
                awaiting_audio = False

                text = await loop.run_in_executor(executor, stt_engine.transcribe, bytes_payload)
                logger.info("heard: %r", text)
                if not text:
                    continue

                answer = await loop.run_in_executor(executor, vlm.ask, text, latest_frame)
                logger.info("assistant Q: %r -> A: %r", text, answer)
                await announcer.say(answer, urgent=True)

    except WebSocketDisconnect:
        logger.info("control client disconnected")


@app.get("/health")
async def health():
    return {"status": "ok", "audio_clients": len(announcer.clients) if announcer else 0}
