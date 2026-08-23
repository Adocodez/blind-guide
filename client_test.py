"""
client_test.py

Stand-in for the eventual phone/earpiece client. Pure I/O — all the AI
(detection, STT, the vision-language model) lives on the server. Three
concurrent pieces:
  - send_video:       grabs webcam frames, JPEG-encodes, sends to /ws/video
  - receive_audio:    plays PCM16 audio from /ws/audio (detection alerts AND
                      assistant answers both arrive here, since the server
                      speaks both through the same announcer)
  - control_handler: sends hotkey-triggered commands + mic audio to
                      /ws/control
  - keyboard listener: runs on its own thread (pynput), posts events into
                      an asyncio.Queue that control_handler consumes

Hotkeys:
  F8         toggle image-recognition detection on/off
  hold F9    push-to-talk: interrupts current speech, records raw mic audio,
             sends it to the server on release. Server transcribes with
             Whisper, asks Ollama (using its own latest cached frame), and
             speaks the answer back over /ws/audio.

Install: pip install opencv-python websockets sounddevice numpy pynput

Usage:
    python client_test.py --host 192.168.1.20 --port 8000
"""

import argparse
import asyncio
import json

import cv2
import numpy as np
import sounddevice as sd
import websockets
from pynput import keyboard

SAMPLE_RATE = 24000      # audio OUT from server, must match audio_engine.SAMPLE_RATE
MIC_SAMPLE_RATE = 16000  # audio IN to server, must match stt.SAMPLE_RATE
CHANNELS = 1
FRAME_SEND_HZ = 10      # cap outgoing frame rate to keep bandwidth/CPU sane


async def send_video(uri: str):
    """Captures webcam frames and streams JPEG buffers to /ws/video."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[video] Error: Could not open webcam")
        return

    interval = 1.0 / FRAME_SEND_HZ

    while True:
        try:
            print(f"[video] connecting to {uri}...")
            async with websockets.connect(uri, max_size=None) as ws:
                print(f"[video] connected to {uri}")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(interval)
                        continue

                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        await ws.send(buf.tobytes())

                    await asyncio.sleep(interval)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[video] connection lost ({e}). Reconnecting in 2.5s...")
            await asyncio.sleep(2.5)
        except Exception as e:
            print(f"[video] unexpected error: {e}")
            await asyncio.sleep(2.5)


async def receive_audio(uri: str):
    """Receives and plays raw PCM16 mono audio streams from /ws/audio."""
    while True:
        try:
            print(f"[audio] connecting to {uri}...")
            async with websockets.connect(uri, max_size=None) as ws:
                print(f"[audio] connected to {uri}")
                with sd.RawOutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=1024,
                ) as out_stream:
                    async for message in ws:
                        if isinstance(message, (bytes, bytearray)):
                            out_stream.write(np.frombuffer(message, dtype=np.int16))
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[audio] connection lost ({e}). Reconnecting in 2.5s...")
            await asyncio.sleep(2.5)
        except Exception as e:
            print(f"[audio] unexpected error: {e}")
            await asyncio.sleep(2.5)


async def control_handler(uri: str, hotkey_queue: asyncio.Queue):
    """Handles control commands, detection toggles, and push-to-talk mic uploads."""
    mic_frames: list[np.ndarray] = []
    mic_stream: sd.InputStream | None = None

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] status: {status}")
        mic_frames.append(indata.copy())

    detection_on = True

    while True:
        try:
            print(f"[control] connecting to {uri}...")
            async with websockets.connect(uri, max_size=None) as ws:
                print(f"[control] connected to {uri}")

                while True:
                    event = await hotkey_queue.get()
                    kind = event["type"]

                    if kind == "toggle_detection":
                        detection_on = not detection_on
                        await ws.send(json.dumps({"cmd": "toggle_detection", "value": detection_on}))
                        print(f"[control] detection -> {'ON' if detection_on else 'OFF'}")

                    elif kind == "ptt_down":
                        await ws.send(json.dumps({"cmd": "interrupt"}))
                        mic_frames = []
                        mic_stream = sd.InputStream(
                            samplerate=MIC_SAMPLE_RATE,
                            channels=1,
                            dtype="int16",
                            callback=mic_callback,
                        )
                        mic_stream.start()
                        print("[control] listening...")

                    elif kind == "ptt_up":
                        if mic_stream is None:
                            continue

                        mic_stream.stop()
                        mic_stream.close()
                        mic_stream = None

                        if not mic_frames:
                            print("[control] warning: no audio recorded")
                            continue

                        audio_bytes = np.concatenate(mic_frames, axis=0).flatten().tobytes()
                        mic_frames = []

                        await ws.send(json.dumps({"cmd": "assistant_audio_query"}))
                        await ws.send(audio_bytes)
                        print("[control] sent query audio. waiting for server response...")

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[control] connection lost ({e}). Reconnecting in 2.5s...")
            if mic_stream is not None:
                mic_stream.stop()
                mic_stream.close()
                mic_stream = None
            await asyncio.sleep(2.5)
        except Exception as e:
            print(f"[control] unexpected error: {e}")
            await asyncio.sleep(2.5)


function_listener = None


def start_keyboard_listener(loop: asyncio.AbstractEventLoop, hotkey_queue: asyncio.Queue) -> keyboard.Listener:
    ptt_active = False

    def on_press(key):
        nonlocal ptt_active
        if key == keyboard.Key.f8:
            asyncio.run_coroutine_threadsafe(hotkey_queue.put({"type": "toggle_detection"}), loop)
        elif key == keyboard.Key.f9 and not ptt_active:
            ptt_active = True
            asyncio.run_coroutine_threadsafe(hotkey_queue.put({"type": "ptt_down"}), loop)

    def on_release(key):
        nonlocal ptt_active
        if key == keyboard.Key.f9:
            ptt_active = False
            asyncio.run_coroutine_threadsafe(hotkey_queue.put({"type": "ptt_up"}), loop)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


async def main():
    parser = argparse.ArgumentParser(description="Blind Guide Test Client")
    parser.add_argument("--host", default="localhost", help="Server host IP or domain")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    args = parser.parse_args()

    base_url = "localhost:8000"
    video_uri = f"ws://{base_url}/ws/video"
    audio_uri = f"ws://{base_url}/ws/audio"
    control_uri = f"ws://{base_url}/ws/control"

    loop = asyncio.get_event_loop()
    hotkey_queue: asyncio.Queue = asyncio.Queue()

    listener = start_keyboard_listener(loop, hotkey_queue)
    print(f"Targeting server: {base_url}")
    print("Hotkeys: F8 = toggle image recognition | hold F9 = ask the assistant\n")

    try:
        await asyncio.gather(
            send_video(video_uri),
            receive_audio(audio_uri),
            control_handler(control_uri, hotkey_queue),
        )
    finally:
        listener.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting client.")