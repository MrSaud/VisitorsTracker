"""Person detection, centroid tracking and line-crossing visitor counting.

Two detector backends are available:

``hog``
    OpenCV's built-in HOG + linear-SVM pedestrian detector. Needs no model
    file at all, so the project runs straight after ``pip install``. Scores
    are raw SVM decision values (typically 0.2 - 1.5), not probabilities.

``onnx``
    A YOLOv8 person detector exported to ONNX and dropped into
    ``counter/ml_models/``. Much more accurate; used automatically when
    ``COUNTER_DETECTOR=onnx`` and the file exists.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "ml_models"

# COCO class index for "person" in a standard YOLOv8 export.
PERSON_CLASS_ID = 0

HORIZONTAL = "horizontal"
VERTICAL = "vertical"

DIRECTION_IN = "in"
DIRECTION_OUT = "out"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def non_max_suppression(boxes, scores, iou_threshold: float = 0.45):
    """Greedy NMS. ``boxes`` is an (N, 4) array of x1, y1, x2, y2."""
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)
        order = rest[iou <= iou_threshold]

    return keep


def _letterbox(frame, size: int):
    """Resize keeping aspect ratio, pad to a square. Returns padded, scale, dx, dy."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

class HogPersonDetector:
    """OpenCV HOG pedestrian detector. No model file required."""

    name = "hog"

    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = float(min_confidence)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        rects, weights = self._hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if len(rects) == 0:
            return []

        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        boxes, scores = [], []
        for (x, y, w, h), score in zip(rects, weights):
            if score < self.min_confidence:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            scores.append(float(score))

        keep = non_max_suppression(boxes, scores, iou_threshold=0.4)
        return [(boxes[i], scores[i]) for i in keep]


class OnnxPersonDetector:
    """YOLOv8 person detector run through ``cv2.dnn``."""

    name = "onnx"

    def __init__(self, model_path, min_confidence: float = 0.45, input_size: int = 640):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.min_confidence = float(min_confidence)
        self.input_size = int(input_size)
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def detect(self, frame):
        padded, scale, dx, dy = _letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        output = self._net.forward()

        # YOLOv8 exports (1, 4 + num_classes, num_boxes); transpose to rows.
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            return []
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes, scores = [], []
        for row in predictions:
            class_scores = row[4:]
            if class_scores.size <= PERSON_CLASS_ID:
                continue
            confidence = float(class_scores[PERSON_CLASS_ID])
            if confidence < self.min_confidence:
                continue
            # Only keep rows where "person" is the winning class.
            if int(np.argmax(class_scores)) != PERSON_CLASS_ID:
                continue
            cx, cy, w, h = (float(v) for v in row[:4])
            x1 = (cx - w / 2 - dx) / scale
            y1 = (cy - h / 2 - dy) / scale
            x2 = (cx + w / 2 - dx) / scale
            y2 = (cy + h / 2 - dy) / scale
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)

        keep = non_max_suppression(boxes, scores, iou_threshold=0.45)
        return [(boxes[i], scores[i]) for i in keep]


def build_detector(backend: str, min_confidence: float, model_name: str = "yolov8n.onnx"):
    """Return a detector, falling back to HOG when the ONNX model is absent."""
    backend = (backend or "hog").strip().lower()
    if backend == "onnx":
        candidate = Path(model_name)
        if not candidate.is_absolute():
            candidate = MODEL_DIR / candidate
        try:
            return OnnxPersonDetector(candidate, min_confidence=min_confidence)
        except (FileNotFoundError, cv2.error):
            # Fall through to HOG so the app still runs without the model.
            pass
    return HogPersonDetector(min_confidence=min_confidence)


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    centroid: tuple
    box: tuple
    disappeared: int = 0
    side: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=24))


class CentroidTracker:
    """Nearest-centroid tracker: cheap, and good enough for a counting line."""

    def __init__(self, max_disappeared: int = 25, max_distance: float = 120.0):
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)
        self.tracks: "OrderedDict[int, Track]" = OrderedDict()
        self._next_id = 1

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _register(self, box):
        centroid = self._centroid(box)
        track = Track(track_id=self._next_id, centroid=centroid, box=tuple(box))
        track.trail.append(centroid)
        self.tracks[track.track_id] = track
        self._next_id += 1
        return track

    def update(self, boxes):
        """Feed this frame's boxes; returns the live tracks."""
        if not boxes:
            for track in list(self.tracks.values()):
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    del self.tracks[track.track_id]
            return self.tracks

        if not self.tracks:
            for box in boxes:
                self._register(box)
            return self.tracks

        track_ids = list(self.tracks.keys())
        existing = np.array([self.tracks[t].centroid for t in track_ids], dtype=np.float32)
        incoming = np.array([self._centroid(b) for b in boxes], dtype=np.float32)

        # Pairwise distances, then greedy nearest matching.
        distances = np.linalg.norm(existing[:, None, :] - incoming[None, :, :], axis=2)

        used_rows, used_cols = set(), set()
        for row, col in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
            row, col = int(row), int(col)
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            track = self.tracks[track_ids[row]]
            track.centroid = tuple(float(v) for v in incoming[col])
            track.box = tuple(boxes[col])
            track.disappeared = 0
            track.trail.append(track.centroid)
            used_rows.add(row)
            used_cols.add(col)

        for row, track_id in enumerate(track_ids):
            if row not in used_rows:
                track = self.tracks[track_id]
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    del self.tracks[track_id]

        for col, box in enumerate(boxes):
            if col not in used_cols:
                self._register(box)

        return self.tracks


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------

class LineCounter:
    """Counts tracks crossing a virtual line, with a dead band against jitter."""

    def __init__(
        self,
        position: float = 0.5,
        orientation: str = HORIZONTAL,
        invert: bool = False,
        margin: int = 10,
    ):
        self.position = float(position)
        self.orientation = orientation if orientation in (HORIZONTAL, VERTICAL) else HORIZONTAL
        self.invert = bool(invert)
        self.margin = int(margin)
        self.total_in = 0
        self.total_out = 0

    def line_coordinate(self, frame_shape) -> int:
        height, width = frame_shape[:2]
        extent = height if self.orientation == HORIZONTAL else width
        return int(round(self.position * extent))

    def _axis_value(self, centroid):
        # Horizontal line -> compare the y axis; vertical line -> the x axis.
        return centroid[1] if self.orientation == HORIZONTAL else centroid[0]

    def update(self, tracks, frame_shape):
        """Returns a list of ``(track_id, direction)`` for crossings this frame."""
        line = self.line_coordinate(frame_shape)
        events = []

        for track in tracks.values():
            value = self._axis_value(track.centroid)
            delta = value - line

            # Inside the dead band we do not commit to a side.
            if abs(delta) < self.margin:
                continue

            side = 1 if delta > 0 else -1
            previous = track.side

            if previous == 0:
                track.side = side
                continue

            if side != previous:
                # -1 -> 1 means crossing in the direction of increasing axis.
                forward = side > previous
                if self.invert:
                    forward = not forward
                direction = DIRECTION_IN if forward else DIRECTION_OUT
                if direction == DIRECTION_IN:
                    self.total_in += 1
                else:
                    self.total_out += 1
                events.append((track.track_id, direction))
                track.side = side

        return events


class VisitorCounter:
    """Detector + tracker + line counter over a stream of frames."""

    def __init__(
        self,
        detector,
        line: LineCounter | None = None,
        detect_every: int = 3,
        max_disappeared: int = 25,
        frame_width: int = 640,
    ):
        self.detector = detector
        self.line = line or LineCounter()
        self.detect_every = max(1, int(detect_every))
        self.frame_width = int(frame_width)
        self.tracker = CentroidTracker(max_disappeared=max_disappeared)
        self._frame_index = 0
        self._last_boxes: list = []

    @property
    def total_in(self) -> int:
        return self.line.total_in

    @property
    def total_out(self) -> int:
        return self.line.total_out

    @property
    def occupancy(self) -> int:
        return max(0, self.line.total_in - self.line.total_out)

    @property
    def active_tracks(self) -> int:
        return len(self.tracker.tracks)

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if width <= self.frame_width:
            return frame
        scale = self.frame_width / float(width)
        return cv2.resize(frame, (self.frame_width, int(round(height * scale))))

    def process(self, frame):
        """Run one frame. Returns ``(annotated_frame, events)``."""
        frame = self._resize(frame)

        if self._frame_index % self.detect_every == 0:
            detections = self.detector.detect(frame)
            self._last_boxes = [box for box, _score in detections]
        self._frame_index += 1

        tracks = self.tracker.update(self._last_boxes)
        events = self.line.update(tracks, frame.shape)
        return self.annotate(frame, tracks), events

    def annotate(self, frame, tracks):
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        line = self.line.line_coordinate(canvas.shape)

        if self.line.orientation == HORIZONTAL:
            cv2.line(canvas, (0, line), (width, line), (0, 215, 255), 2)
        else:
            cv2.line(canvas, (line, 0), (line, height), (0, 215, 255), 2)

        for track in tracks.values():
            x1, y1, x2, y2 = (int(v) for v in track.box)
            cx, cy = (int(v) for v in track.centroid)
            colour = (80, 220, 120) if track.disappeared == 0 else (150, 150, 150)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(canvas, (cx, cy), 4, colour, -1)
            cv2.putText(
                canvas, f"#{track.track_id}", (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )
            points = list(track.trail)
            for a, b in zip(points, points[1:]):
                cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, 1)

        banner = f"IN {self.total_in}   OUT {self.total_out}   INSIDE {self.occupancy}"
        cv2.rectangle(canvas, (0, 0), (width, 30), (28, 28, 32), -1)
        cv2.putText(
            canvas, banner, (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )
        return canvas
