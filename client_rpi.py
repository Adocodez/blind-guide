"""
client_test.py - Raspberry Pi 3B Client (Bookworm 12 32-bit)

Handles I/O on Raspberry Pi 3B:
- Video: RPi Camera v1.3 captured via Picamera2
- Audio IN: USB Microphone via sounddevice
- Audio OUT: 3.5mm AUX Speaker via sounddevice
- Control: Physical GPIO Pushbuttons via gpiozero
"""

import asyncio
import json
import socket
import cv2
import numpy as np
import sounddevice as sd
import websockets
from gpiozero import Button
from picamera2 import Picamera2
from zeroconf import Zeroconf

# ----------------------------------------------------------------------
# Configuration & Hardware Settings
# ----------------------------------------------------------------------
MANUAL_HOST: str | None = None
MANUAL_PORT: int = 8000

# GPIO Pin Mapping (BCM Numbering)
PIN_TOGGLE_DETECTION = 17  # Button to toggle detection on/off
PIN_PTT = 27               # Button to hold for Push-To-Talk

# Sound Device Index Settings (Set to None for system default, or integer index)
AUDIO_INPUT_DEVICE = None   # USB Microphone
AUDIO_OUTPUT_DEVICE = "plughw:3,0"  # 3.5mm AUX Jack

SAMPLE_RATE = 24000      # Audio OUT from server (matches audio_engine.SAMPLE_RATE)
MIC_SAMPLE_RATE = 48000  # Audio IN to server (matches stt.SAMPLE_RATE)
CHANNELS = 1
FRAME_SEND_HZ = 10       # Frame rate cap

SERVICE_TYPE = "_blindguide._tcp.local."
SERVICE_NAME = "blindguide." + SERVICE_TYPE

zc = Zeroconf()


def resolve_server(zc_inst: Zeroconf, timeout_s: float = 5.0) -> tuple[str, int] | None:
    info = zc_inst.get_service_info(SERVICE_TYPE, SERVICE_NAME, timeout=int(timeout_s * 1000))
    if info is None or not info.addresses:
        return None
    ip = socket.inet_ntoa(info.addresses[0])
    return ip, info.port


async def get_server_uri(path: str) -> str | None:
    if MANUAL_HOST:
        return f"ws://{MANUAL_HOST}:{MANUAL_PORT}{path}"

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, resolve_server, zc, 5.0)
    if result is None:
        return None
    ip, port = result
    return f"ws://{ip}:{port}{path}"


async def send_video():
    """Captures frames from RPi Camera v1.3 using Picamera2 and streams JPEG buffers."""
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("[video] RPi Camera v1.3 initialized via Picamera2")
    except Exception as e:
        print(f"[video] Fatal Camera Error: {e}")
        return

    interval = 1.0 / FRAME_SEND_HZ

    try:
        while True:
            uri = await get_server_uri("/ws/video")
            if uri is None:
                print("[video] Server not found via mDNS, retrying in 3s...")
                await asyncio.sleep(3)
                continue

            try:
                print(f"[video] Connecting to {uri}...")
                async with websockets.connect(uri, max_size=None) as ws:
                    print(f"[video] Connected to {uri}")
                    while True:
                        frame_rgb = picam2.capture_array()
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            await ws.send(buf.tobytes())

                        await asyncio.sleep(interval)
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                print(f"[video] Connection lost ({e}). Re-resolving server in 2.5s...")
                await asyncio.sleep(2.5)
            except Exception as e:
                print(f"[video] Unexpected error: {e}")
                await asyncio.sleep(2.5)
    finally:
        picam2.stop()


async def receive_audio():
    """Receives and plays raw PCM16 mono audio streams over 3.5mm AUX speaker."""
    while True:
        uri = await get_server_uri("/ws/audio")
        if uri is None:
            print("[audio] Server not found via mDNS, retrying in 3s...")
            await asyncio.sleep(3)
            continue

        try:
            print(f"[audio] Connecting to {uri}...")
            async with websockets.connect(uri, max_size=None) as ws:
                print(f"[audio] Connected to {uri}")
                with sd.RawOutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=1024,
                    device=AUDIO_OUTPUT_DEVICE,
                ) as out_stream:
                    async for message in ws:
                        if isinstance(message, (bytes, bytearray)):
                            out_stream.write(np.frombuffer(message, dtype=np.int16))
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[audio] Connection lost ({e}). Re-resolving server in 2.5s...")
            await asyncio.sleep(2.5)
        except Exception as e:
            print(f"[audio] Unexpected error: {e}")
            await asyncio.sleep(2.5)


async def control_handler(hotkey_queue: asyncio.Queue):
    """Handles commands, detection toggles, and USB mic audio uploads."""
    mic_frames: list[np.ndarray] = []
    mic_stream: sd.InputStream | None = None

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] status: {status}")
        mic_frames.append(indata.copy())

    detection_on = True

    while True:
        uri = await get_server_uri("/ws/control")
        if uri is None:
            print("[control] Server not found via mDNS, retrying in 3s...")
            await asyncio.sleep(3)
            continue

        try:
            print(f"[control] Connecting to {uri}...")
            async with websockets.connect(uri, max_size=None) as ws:
                print(f"[control] Connected to {uri}")

                while True:
                    event = await hotkey_queue.get()
                    kind = event["type"]

                    if kind == "toggle_detection":
                        detection_on = not detection_on
                        await ws.send(json.dumps({"cmd": "toggle_detection", "value": detection_on}))
                        print(f"[control] Detection -> {'ON' if detection_on else 'OFF'}")

                    elif kind == "ptt_down":
                        await ws.send(json.dumps({"cmd": "interrupt"}))
                        mic_frames = []
                        mic_stream = sd.InputStream(
                            samplerate=MIC_SAMPLE_RATE,
                            channels=1,
                            dtype="int16",
                            callback=mic_callback,
                            device=AUDIO_INPUT_DEVICE,
                        )
                        mic_stream.start()
                        print("[control] Listening...")

                    elif kind == "ptt_up":
                        if mic_stream is None:
                            continue

                        mic_stream.stop()
                        mic_stream.close()
                        mic_stream = None

                        if not mic_frames:
                            print("[control] Warning: no audio recorded")
                            continue

                        audio_bytes = np.concatenate(mic_frames, axis=0).flatten().tobytes()
                        mic_frames = []

                        await ws.send(json.dumps({"cmd": "assistant_audio_query"}))
                        await ws.send(audio_bytes)
                        print("[control] Sent query audio. Waiting for server response...")

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[control] Connection lost ({e}). Re-resolving server in 2.5s...")
            if mic_stream is not None:
                mic_stream.stop()
                mic_stream.close()
                mic_stream = None
            await asyncio.sleep(2.5)
        except Exception as e:
            print(f"[control] Unexpected error: {e}")
            await asyncio.sleep(2.5)


def setup_gpio_buttons(loop: asyncio.AbstractEventLoop, hotkey_queue: asyncio.Queue) -> tuple[Button, Button]:
    """Initializes hardware pushbuttons connected to GPIO pins."""
    btn_toggle = Button(PIN_TOGGLE_DETECTION, pull_up=True, bounce_time=0.05)
    btn_ptt = Button(PIN_PTT, pull_up=True, bounce_time=0.05)

    btn_toggle.when_pressed = lambda: asyncio.run_coroutine_threadsafe(
        hotkey_queue.put({"type": "toggle_detection"}), loop
    )
    
    btn_ptt.when_pressed = lambda: asyncio.run_coroutine_threadsafe(
        hotkey_queue.put({"type": "ptt_down"}), loop
    )
    btn_ptt.when_released = lambda: asyncio.run_coroutine_threadsafe(
        hotkey_queue.put({"type": "ptt_up"}), loop
    )

    print(f"GPIO initialized: Pin {PIN_TOGGLE_DETECTION} (Toggle) | Pin {PIN_PTT} (Push-To-Talk)")
    return btn_toggle, btn_ptt


async def main():
    if MANUAL_HOST:
        print(f"Using manual server address: {MANUAL_HOST}:{MANUAL_PORT}")
    else:
        print("Auto-discovering server via mDNS...")

    loop = asyncio.get_running_loop()
    hotkey_queue: asyncio.Queue = asyncio.Queue()

    btn_toggle, btn_ptt = setup_gpio_buttons(loop, hotkey_queue)

    try:
        await asyncio.gather(
            send_video(),
            receive_audio(),
            control_handler(hotkey_queue),
        )
    finally:
        btn_toggle.close()
        btn_ptt.close()
        zc.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting client.")