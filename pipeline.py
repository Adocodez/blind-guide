"""
pipeline.py

Detection + tracking + alert logic, extracted from the original single-file
script. Has no camera, TTS, or UI code in it — it just takes frames in and
gives back a list of announcement dicts. This makes it reusable from a server,
a local test script, or (later) fed by a different frame source entirely.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLOWorld

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

CLASSES = [
    "door", "door handle", "person", "chair", "stairs", "table",
    "phone", "window", "laptop", "keyboard", "mouse", "remote control",
    "book", "bottle", "cup", "fork", "knife", "spoon", "bowl", "apple", "fan"
]

CONF_THRESHOLD = 0.35
HAZARD_CLASSES = {"apple"}  # {"stairs", "chair", "table", "door"}

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


def phrase(label: str, distance: str, position: str, count: int = 1) -> str:
    """Builds an announcement phrase, grouping same-class objects when count > 1."""
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
    Groups per-track announcement events sharing (label, distance, position)
    into one combined announcement with a count, so e.g. three chairs at the
    same distance/position produce one utterance instead of three.
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
# Per-track memory
# ----------------------------------------------------------------------

@dataclass
class TrackState:
    label: str
    last_distance: str
    last_position: str
    last_box_ratio: float
    last_announced_at: float = field(default_factory=time.time)
    first_seen_at: float = field(default_factory=time.time)


class AlertMemory:
    DISTANCE_ORDER = ["far", "nearby", "close", "very close"]

    def __init__(self, ratio_change_threshold: float = 0.08):
        self.tracks: dict[int, TrackState] = {}
        self.ratio_change_threshold = ratio_change_threshold

    def evaluate(self, track_id, label, distance, position, box_h_ratio) -> tuple[bool, bool]:
        now = time.time()
        prev = self.tracks.get(track_id)

        if prev is None:
            self.tracks[track_id] = TrackState(label, distance, position, last_box_ratio=box_h_ratio)
            urgent = distance in ("very close", "close") and label in HAZARD_CLASSES
            return True, urgent

        entered_closer_bucket = (
            self.DISTANCE_ORDER.index(distance) > self.DISTANCE_ORDER.index(prev.last_distance)
        )
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
            prev.last_box_ratio = box_h_ratio
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


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

class DetectionPipeline:
    """
    Wraps the YOLO-World model + AlertMemory. Call process_frame(frame) with a
    BGR numpy array (e.g. from cv2.imdecode) and get back a list of ready-to-speak
    announcement dicts: {"text": str, "urgent": bool}.

    This class does blocking, synchronous work (model inference) — callers on an
    asyncio event loop should run it in a thread pool executor.
    """

    def __init__(self, weights: str = "yolov8s-world.pt", tracker_config: str = "custom_bytetrack.yaml", device=0):
        self.model = YOLOWorld(weights)
        self.model.set_classes(CLASSES)
        self.tracker_config = tracker_config
        self.device = device
        self.memory = AlertMemory()

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        frame_h, frame_w = frame.shape[:2]

        results = self.model.track(
            frame,
            device=self.device,
            conf=CONF_THRESHOLD,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        active_ids = set()
        pending_events = []

        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue

            for box in boxes:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                active_ids.add(track_id)

                box_h_ratio = (y2 - y1) / frame_h
                x_center = (x1 + x2) / 2

                distance = distance_bucket(box_h_ratio)
                position = position_bucket(x_center, frame_w)

                should_announce, urgent = self.memory.evaluate(
                    track_id, label, distance, position, box_h_ratio
                )
                if should_announce:
                    pending_events.append({
                        "track_id": track_id,
                        "label": label,
                        "distance": distance,
                        "position": position,
                        "urgent": urgent,
                    })

        self.memory.forget_stale(active_ids, max_age=30.0)

        announcements = []
        for group in group_announcements(pending_events):
            text = phrase(group["label"], group["distance"], group["position"], count=group["count"])
            announcements.append({"text": text, "urgent": group["urgent"]})

        return announcements
