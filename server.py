import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import asyncio
import json
import logging
import socket
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from pipeline import DetectionPipeline
from audio_engine import StreamingAnnouncer
from ollama_vlm import OllamaVisionAssistant
from stt import SpeechToText
from discovery import SERVICE_TYPE, SERVICE_NAME, HOSTNAME, DEFAULT_PORT, get_local_ip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()

pipeline: DetectionPipeline | None = None
announcer: StreamingAnnouncer | None = None
vlm: OllamaVisionAssistant | None = None
stt_engine: SpeechToText | None = None
aiozeroconf: AsyncZeroconf | None = None
zeroconf_service_info: ServiceInfo | None = None

SERVER_PORT = DEFAULT_PORT
detection_enabled = True
latest_frame: bytes | None = None

executor = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
async def startup():
    global pipeline, announcer, vlm, stt_engine, aiozeroconf, zeroconf_service_info
    loop = asyncio.get_running_loop()

    local_ip = get_local_ip()
    logger.info("Advertising via mDNS as %s at %s:%d", HOSTNAME, local_ip, SERVER_PORT)
    
    zeroconf_service_info = ServiceInfo(
        SERVICE_TYPE,
        SERVICE_NAME,
        addresses=[socket.inet_aton(local_ip)],
        port=SERVER_PORT,
        server=f"{HOSTNAME}.local.",
    )
    aiozeroconf = AsyncZeroconf()
    await aiozeroconf.async_register_service(zeroconf_service_info)

    logger.info("Loading detection pipeline...")
    pipeline = await asyncio.to_thread(DetectionPipeline)

    logger.info("Starting TTS announcer...")
    announcer = StreamingAnnouncer(voice="af_heart", loop=loop)

    logger.info("Loading Whisper STT model...")
    stt_engine = await asyncio.to_thread(SpeechToText)

    logger.info("Connecting to Ollama vision model...")
    vlm = await asyncio.to_thread(OllamaVisionAssistant)

    logger.info("Ready.")


@app.on_event("shutdown")
async def shutdown():
    global aiozeroconf
    cv2.destroyAllWindows()  # Close OpenCV window on exit
    if aiozeroconf:
        logger.info("Unregistering mDNS service...")
        await aiozeroconf.async_unregister_all_services()
        await aiozeroconf.async_close()
        logger.info("mDNS service stopped.")


@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    logger.info("video client connected")

    global latest_frame

    try:
        while True:
            data = await websocket.receive_bytes()
            latest_frame = data

            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if not detection_enabled:
                # If detection is off, show the raw feed without bounding boxes
                cv2.imshow("Blind-Guide Server Feed", frame)
                cv2.waitKey(1)
                continue

            # Process frame through YOLO pipeline and retrieve drawn frame
            announcements, annotated_frame = await loop.run_in_executor(
                executor, pipeline.process_frame, frame
            )

            # Display annotated stream on server GUI
            cv2.imshow("Blind-Guide Server Feed", annotated_frame)
            cv2.waitKey(1)

            for ann in announcements:
                await announcer.say(ann["text"], urgent=ann["urgent"])

    except WebSocketDisconnect:
        logger.info("video client disconnected")
        cv2.destroyAllWindows()


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    announcer.register_client(websocket)

    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        announcer.unregister_client(websocket)


@app.websocket("/ws/control")
async def control_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    
    client_ip = websocket.client.host if websocket.client else "Unknown"
    logger.info("[control] Client connected from IP: %s", client_ip)

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
                rpi_reported_ip = msg.get("ip", client_ip)

                if cmd == "init_client":
                    detection_enabled = bool(msg.get("value", True))
                    logger.info(">>> RPi Registered | IP: %s | Detection Switch State: %s", 
                                rpi_reported_ip, "ON" if detection_enabled else "OFF")

                elif cmd == "toggle_detection":
                    detection_enabled = bool(msg.get("value", not detection_enabled))
                    logger.info(">>> Switch Update from IP %s -> Detection Enabled: %s", 
                                rpi_reported_ip, detection_enabled)

                elif cmd == "interrupt":
                    logger.info(">>> Interrupt requested by RPi (%s)", rpi_reported_ip)
                    announcer.stream.stop()

                elif cmd == "assistant_audio_query":
                    logger.info(">>> Audio query incoming from RPi (%s)", rpi_reported_ip)
                    awaiting_audio = True

                elif cmd == "assistant_query":
                    logger.info(">>> Query from RPi (%s): %r", rpi_reported_ip, msg.get("text"))
                    answer = await loop.run_in_executor(executor, vlm.ask, msg.get("text", ""), latest_frame)
                    logger.info("assistant Q: %r -> A: %r", msg.get("text"), answer)
                    await announcer.say(answer, urgent=True)

                else:
                    logger.warning("control: unknown cmd %r from %s", cmd, rpi_reported_ip)

            elif bytes_payload is not None:
                if not awaiting_audio:
                    logger.warning("control: got audio with no pending assistant_audio_query, ignoring")
                    continue
                awaiting_audio = False

                text = await loop.run_in_executor(executor, stt_engine.transcribe, bytes_payload)
                logger.info("heard from %s: %r", client_ip, text)
                if not text:
                    continue

                answer = await loop.run_in_executor(executor, vlm.ask, text, latest_frame)
                logger.info("assistant Q: %r -> A: %r", text, answer)
                await announcer.say(answer, urgent=True)

    except WebSocketDisconnect:
        logger.info("control client (%s) disconnected", client_ip)


@app.get("/health")
async def health():
    return {"status": "ok", "audio_clients": len(announcer.clients) if announcer else 0}