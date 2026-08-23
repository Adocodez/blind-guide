"""
audio_engine.py

Wraps RealtimeTTS + KokoroEngine so speech is captured as raw PCM chunks and
broadcast to connected `/ws/audio` clients, instead of played on local
speakers. This is what lets the server run headless (e.g. on a Jetson/laptop
with no audio output) while a phone or earpiece client plays the audio.

NOTE ON RealtimeTTS VERSIONS: the on_audio_chunk callback is passed to
play()/play_async(), not the TextToAudioStream constructor, in the versions
this was tested against. If you're on a different RealtimeTTS release and
this errors, check `TextToAudioStream.play_async.__doc__` for the current
signature.

RELIABILITY NOTE: we do NOT trust stream.is_playing() to know whether
playback is active. If the RealtimeTTS worker thread throws (a Kokoro
error, a broken audio-chunk callback, etc.) mid-utterance, that thread can
die without is_playing() ever flipping back to False — every future say()
call then just appends text to a stream nobody is consuming, i.e. total
silence forever with no error visible to the caller. Instead we track
"is something playing" ourselves via a threading.Event, driven off
RealtimeTTS's on_audio_stream_start/stop callbacks, and we always clear it
in a finally block if play_async() itself throws.
"""

import asyncio
import logging
import threading

from fastapi import WebSocket
from RealtimeTTS import TextToAudioStream, KokoroEngine

logger = logging.getLogger("audio_engine")

# Kokoro outputs 16-bit mono PCM at 24kHz. Send this to clients so they know
# how to configure playback (see client_test.py).
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # bytes (16-bit)
CHANNELS = 1


class StreamingAnnouncer:
    def __init__(self, voice: str = "af_heart", loop: asyncio.AbstractEventLoop | None = None):
        self.engine = KokoroEngine(voice=voice)
        self.stream = TextToAudioStream(
            self.engine,
            on_audio_stream_start=self._on_stream_start,
            on_audio_stream_stop=self._on_stream_stop,
        )
        self.clients: set[WebSocket] = set()
        # RealtimeTTS calls our callbacks from its own worker thread, so we
        # hop back onto the asyncio loop with run_coroutine_threadsafe.
        self.loop = loop or asyncio.get_event_loop()
        self._playing = threading.Event()

    def register_client(self, ws: WebSocket):
        self.clients.add(ws)
        logger.info("audio client connected (%d total)", len(self.clients))

    def unregister_client(self, ws: WebSocket):
        self.clients.discard(ws)
        logger.info("audio client disconnected (%d total)", len(self.clients))

    def _on_stream_start(self):
        self._playing.set()

    def _on_stream_stop(self):
        self._playing.clear()

    def _on_audio_chunk(self, chunk: bytes):
        # Called from the TTS worker thread. Anything raised here that
        # escapes could kill that thread and silently wedge playback, so we
        # catch broadly and just log.
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(chunk), self.loop)
        except Exception:
            logger.exception("failed to schedule audio broadcast")

    async def _broadcast(self, chunk: bytes):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_bytes(chunk)
            except Exception as e:
                logger.warning("dropping dead audio client: %s", e)
                #await ws.close()
                #dead.append(ws)
        for ws in dead :
            self.clients.discard(ws)

    async def say(self, text: str, urgent: bool = False):
        """Feed text to the TTS stream. Safe to call from async server code."""
        if urgent:
            self.stream.stop()
            self._playing.clear()

        self.stream.feed(text)

        if not self._playing.is_set():
            self._playing.set()
            try:
                # muted=True: don't touch local audio devices, we only want
                # the raw chunks via the callback.
                self.stream.play_async(on_audio_chunk=self._on_audio_chunk, muted=True)
            except Exception:
                logger.exception("play_async failed to start — clearing playing flag so next say() retries")
                self._playing.clear()
