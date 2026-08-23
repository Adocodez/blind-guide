import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
from ultralytics import YOLOWorld
import os
config_path = os.path.abspath("custom_bytetrack.yaml")
# RealtimeTTS with Kokoro Engine
from RealtimeTTS import TextToAudioStream, KokoroEngine


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

CLASSES = [
    "door", "door handle", "person", "chair", "stairs", "table",
    "phone", "window", "laptop", "keyboard", "mouse", "remote control",
    "book", "bottle", "cup", "fork", "pen", "spoon", "apple","fan"
]

CONF_THRESHOLD = 0.35
HAZARD_CLASSES = {"apple"}#{"stairs", "chair", "table", "door"}

DISTANCE_BUCKETS = [
    (0.60, "very close"),
    (0.35, "close"),
    (0.18, "nearby"),
    (0.0, "far"),
]

REWARN_SECONDS = {
    "very close": 4.0,
    "close": 10.0,
    "nearby": 999999.0,
    "far": 999999999.0,
}


def distance_bucket(box_height_ratio: float) -> str:
    for threshold, label in DISTANCE_BUCKETS:
        if box_height_ratio >= threshold:
            return label
    return "far"


def position_bucket(x_center: float, frame_width: int) -> str:
    third = frame_width / 3
    if x_center < third:
        return "on your left"
    elif x_center < 2 * third:
        return "ahead"
    else:
        return "on your right"


def pluralize(label: str) -> str:
    """Very small heuristic pluralizer, good enough for our fixed CLASSES list."""
    if label.endswith(("s", "sh", "ch", "x", "z")):
        return label + "es"
    return label + "s"


# ----------------------------------------------------------------------
# TTS — RealtimeTTS + Kokoro Engine Implementation
# ----------------------------------------------------------------------

class VoiceAnnouncer:
    def __init__(self, voice: str = "af_heart"):
        """
        Initializes KokoroEngine for RealtimeTTS.
        Popular Kokoro voices: 'af_heart', 'af_bella', 'am_adam', 'bf_emma'
        """
        self.engine = KokoroEngine(voice=voice)
        self.stream = TextToAudioStream(self.engine)

    def speak(self, msg: str, urgent: bool = False):
        """
        Feeds text to RealtimeTTS stream. If urgent, interrupts active speech.
        """
        if urgent:
            self.stream.stop()  # Cut off active speech immediately

        self.stream.feed(msg)

        if not self.stream.is_playing():
            self.stream.play_async()


# ----------------------------------------------------------------------
# Per-track memory
# ----------------------------------------------------------------------

@dataclass
class TrackState:
    label: str
    last_distance: str
    last_position: str
    last_box_ratio: float  # Added to track raw size ratio
    last_announced_at: float = field(default_factory=time.time)
    first_seen_at: float = field(default_factory=time.time)


class AlertMemory:
    DISTANCE_ORDER = ["far", "nearby", "close", "very close"]

    def __init__(self, ratio_change_threshold: float = 0.08):
        """
        :param ratio_change_threshold: Minimum change in box_height_ratio needed 
                                        to trigger a 'got closer' alert (e.g., 0.08 = 8% frame height increase).
        """
        self.tracks: dict[int, TrackState] = {}
        self.ratio_change_threshold = ratio_change_threshold

    def evaluate(
        self, 
        track_id: int, 
        label: str, 
        distance: str, 
        position: str, 
        box_h_ratio: float
    ) -> tuple[bool, bool]:
        now = time.time()
        prev = self.tracks.get(track_id)

        # Brand new object detected
        if prev is None:
            self.tracks[track_id] = TrackState(
                label, distance, position, last_box_ratio=box_h_ratio
            )
            urgent = distance in ("very close", "close") and label in HAZARD_CLASSES
            return True, urgent

        # Check 1: Must enter a closer bucket index
        entered_closer_bucket = (
            self.DISTANCE_ORDER.index(distance) > self.DISTANCE_ORDER.index(prev.last_distance)
        )
        
        # Check 2: Must grow larger than the threshold (prevents micro-jitter triggers)
        significant_growth = (box_h_ratio - prev.last_box_ratio) >= self.ratio_change_threshold

        got_closer = entered_closer_bucket and significant_growth
        moved = position != prev.last_position
        stale_enough = (now - prev.last_announced_at) > REWARN_SECONDS.get(distance, 999999)

        should = got_closer or (moved and distance in ("very close", "close")) or (
            label in HAZARD_CLASSES and stale_enough
        )

        if should:
            urgent = distance == "very close"
            prev.last_distance = distance
            prev.last_position = position
            prev.last_box_ratio = box_h_ratio  # Update stored size ratio
            prev.last_announced_at = now
            return True, urgent

        return False, False


    def forget_stale(self, active_ids: set, max_age: float = 5.0):
        now = time.time()
        to_drop = [
            tid for tid, st in self.tracks.items()
            if tid not in active_ids and (now - st.last_announced_at) > max_age
        ]
        for tid in to_drop:
            del self.tracks[tid]


def phrase(label: str, distance: str, position: str, count: int = 1) -> str:
    """
    Builds an announcement phrase. When count > 1, groups same-class objects
    that share a distance/position bucket into a single spoken line instead
    of firing off one utterance per object (e.g. "3 people ahead, close."
    instead of three separate "Person ahead, close." calls).
    """
    if count > 1:
        label_text = pluralize(label)
        if distance == "very close":
            return f"Careful — {count} {label_text} very close, {position}."
        return f"{count} {label_text} {position}, {distance}."

    if distance == "very close":
        return f"Careful — {label} very close, {position}."
    return f"{label.capitalize()} {position}, {distance}."


def group_announcements(events: list[dict]) -> list[dict]:
    """
    Groups per-track announcement events that share the same (label, distance,
    position) into a single announcement with a count, so multiple instances
    of the same object (e.g. several chairs) produce one combined utterance
    instead of one per object. Urgency is preserved if any event in the group
    is urgent, and hazard classes are never merged away silently.
    """
    groups: dict[tuple, dict] = {}
    for ev in events:
        key = (ev["label"], ev["distance"], ev["position"])
        if key not in groups:
            groups[key] = {
                "label": ev["label"],
                "distance": ev["distance"],
                "position": ev["position"],
                "count": 0,
                "urgent": False,
            }
        groups[key]["count"] += 1
        groups[key]["urgent"] = groups[key]["urgent"] or ev["urgent"]
    return list(groups.values())


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(CLASSES)

    # Initialize Kokoro with preferred voice (e.g., 'af_heart' or 'am_adam')
    announcer = VoiceAnnouncer(voice="af_heart")
    memory = AlertMemory()

    cap = cv2.VideoCapture(0)  # Replace with your camera URL or device index

    print("Starting inference. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        results = model.track(
            frame,
            device=0,
            conf=CONF_THRESHOLD,
            persist=True,
            tracker="custom_bytetrack.yaml",
            verbose=False,
        )

        active_ids = set()
        pending_announcements = []  # collected this frame, spoken as grouped batch

        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue

            for box in boxes:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                active_ids.add(track_id)

                box_h_ratio = (y2 - y1) / frame_h
                x_center = (x1 + x2) / 2

                distance = distance_bucket(box_h_ratio)
                position = position_bucket(x_center, frame_w)

                should_announce, urgent = memory.evaluate(track_id, label, distance, position, box_h_ratio)
                if should_announce:
                    pending_announcements.append({
                        "track_id": track_id,
                        "label": label,
                        "distance": distance,
                        "position": position,
                        "urgent": urgent,
                    })

                color = (0, 0, 255) if distance == "very close" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"#{track_id} {label} {conf:.2f} [{distance}]",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2,
                )

        # Speak one grouped line per (label, distance, position) instead of
        # one line per individual track, so multiple same-class objects
        # (e.g. three chairs, all "close, ahead") don't spam the TTS queue.
        for group in group_announcements(pending_announcements):
            msg = phrase(group["label"], group["distance"], group["position"], count=group["count"])
            announcer.speak(msg, urgent=group["urgent"])

        memory.forget_stale(active_ids, max_age=30.0)

        cv2.imshow("Blind-Guide Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
