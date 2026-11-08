"""SORT-style tracking: Kalman prediction, IoU matching, optimal assignment.

The centroid tracker this replaces matches on nearest distance alone. That
fails in exactly the situation a doorway produces constantly: two people
walking past each other. At the moment they overlap, each detection is
nearest to the *other* person's last position, so the identities swap. A
swap around a counting line turns one in and one out into two of the same,
or into nothing at all.

Three things fix it:

* **Motion.** Each track runs a constant-velocity Kalman filter, so it is
  matched against where it is *predicted* to be, not where it last was.
  Two people crossing have opposite velocities, which keeps them apart.
* **Shape.** Matching on box overlap rather than centre distance uses size
  and aspect, so a near person and a far one are not confused.
* **Optimality.** Assignment is solved globally rather than greedily, so one
  confident early match cannot force a cascade of wrong ones.

Two-stage association (the ByteTrack idea) is used when the detector
reports calibrated probabilities: match confident detections first, then
give unmatched tracks a second chance against the weak detections that a
partly occluded person produces, rather than dropping them.
"""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np

from .vision import Track


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def iou_batch(boxes_a, boxes_b):
    """Pairwise IoU. Returns an (len(a), len(b)) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def linear_assignment(cost):
    """Minimum-cost assignment (Hungarian / Jonker-Volgenant).

    Returns a list of ``(row, column)`` pairs. Implemented here rather than
    pulled from scipy: it is the only function we would need from it, and
    the runtime stays numpy-only.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []

    rows, cols = cost.shape
    transposed = rows > cols
    if transposed:
        cost = cost.T
        rows, cols = cols, rows

    INF = float("inf")
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=int)     # column -> row
    way = np.zeros(cols + 1, dtype=int)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, INF)
        used = np.zeros(cols + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(p[j] - 1, j - 1) for j in range(1, cols + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return pairs


# ---------------------------------------------------------------------------
# motion model
# ---------------------------------------------------------------------------

class KalmanBox:
    """Constant-velocity Kalman filter over a box as (centre, area, aspect)."""

    def __init__(self, box):
        self.state = np.zeros((7, 1), dtype=np.float64)
        self.state[:4] = self._to_measurement(box)

        self.F = np.eye(7)
        for index in range(3):
            self.F[index, index + 4] = 1.0

        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0        # velocities start very uncertain

        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0          # area and aspect are noisier than position

    @staticmethod
    def _to_measurement(box):
        x1, y1, x2, y2 = (float(v) for v in box)
        width = max(1e-6, x2 - x1)
        height = max(1e-6, y2 - y1)
        return np.array([
            [x1 + width / 2.0],
            [y1 + height / 2.0],
            [width * height],
            [width / height],
        ])

    @staticmethod
    def _to_box(state):
        # state is a (7, 1) column vector, so index both axes: state[i]
        # alone is a 1-element array and numpy no longer coerces that.
        cx, cy, area, ratio = (float(state[i, 0]) for i in range(4))
        area = max(area, 1e-6)
        ratio = max(ratio, 1e-6)
        width = float(np.sqrt(area * ratio))
        height = area / max(width, 1e-6)
        return (cx - width / 2.0, cy - height / 2.0,
                cx + width / 2.0, cy + height / 2.0)

    @property
    def box(self):
        return self._to_box(self.state)

    def predict(self):
        # Stop a shrinking area from going negative and inverting the box.
        if float(self.state[6, 0]) + float(self.state[2, 0]) <= 0:
            self.state[6, 0] = 0.0
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box):
        z = self._to_measurement(box)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        return self.box


# ---------------------------------------------------------------------------
# tracker
# ---------------------------------------------------------------------------

def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class _Candidate:
    """Internal bookkeeping for one tracked person.

    It owns a single long-lived :class:`~counter.vision.Track` and mutates
    it in place. That matters: the line counter records which side of the
    line a track is on by writing to ``track.side``, so handing out a fresh
    object each frame would discard that and no crossing would ever fire.
    """

    __slots__ = ("kalman", "hits", "time_since_update", "track")

    def __init__(self, track_id, box):
        box = tuple(float(v) for v in box)
        self.kalman = KalmanBox(box)
        self.hits = 1
        self.time_since_update = 0
        self.track = Track(track_id=track_id, centroid=_centre(box), box=box)
        self.track.trail.append(_centre(box))

    @property
    def track_id(self):
        return self.track.track_id

    @property
    def box(self):
        return self.track.box

    def place(self, box):
        """Move the track to ``box`` and extend its trail."""
        box = tuple(float(v) for v in box)
        self.track.box = box
        self.track.centroid = _centre(box)
        self.track.disappeared = self.time_since_update
        self.track.trail.append(self.track.centroid)


class SortTracker:
    """Kalman + IoU + optimal assignment, with optional two-stage matching."""

    def __init__(self, max_disappeared: int = 25, match_threshold: float = 0.2,
                 min_hits: int = 2, high_confidence: float | None = None,
                 low_match_threshold: float = 0.15,
                 gate_factor: float = 2.0, proximity_weight: float = 0.7):
        self.max_disappeared = int(max_disappeared)
        self.match_threshold = float(match_threshold)
        self.min_hits = max(1, int(min_hits))
        self.high_confidence = high_confidence
        self.low_match_threshold = float(low_match_threshold)
        # Centre may move up to gate_factor box-diagonals between frames.
        self.gate_factor = float(gate_factor)
        self.proximity_weight = float(proximity_weight)

        self._candidates: list[_Candidate] = []
        self._next_id = 1

    # -- public surface mirrors CentroidTracker -------------------------

    @property
    def tracks(self):
        """Confirmed tracks, as the counter and annotator expect them.

        Tentative tracks are withheld until they have been seen
        ``min_hits`` times, so a one-frame false positive cannot be counted.

        The Track objects are live and are mutated in place on later
        frames: the dict is a view of who is present now, not a snapshot
        of where they were. Copy what you need if you want history.
        """
        return OrderedDict(
            (c.track_id, c.track)
            for c in self._candidates
            if c.hits >= self.min_hits
        )

    def advance(self):
        """Predict forward without new measurements (a skipped frame)."""
        for candidate in self._candidates:
            candidate.place(candidate.kalman.predict())
        return self.tracks

    def forget_tracks(self):
        """Drop every track, keeping the id counter.

        See :meth:`counter.vision.CentroidTracker.forget_tracks`: the
        Kalman state of a track from before a stream gap is worse than no
        state at all.
        """
        self._candidates.clear()

    def update_detections(self, detections):
        """Common interface, shared with CentroidTracker."""
        return self.update(detections)

    def update(self, detections):
        """``detections`` is a list of ``(box, score)``."""
        boxes = [d[0] for d in detections]
        scores = [float(d[1]) for d in detections]

        # 1. Predict every existing track forward.
        predicted = [c.kalman.predict() for c in self._candidates]

        unmatched_tracks = list(range(len(self._candidates)))
        unmatched_dets = list(range(len(boxes)))

        if self.high_confidence is None:
            strong = list(range(len(boxes)))
            weak = []
        else:
            strong = [i for i, s in enumerate(scores) if s >= self.high_confidence]
            weak = [i for i, s in enumerate(scores) if s < self.high_confidence]

        # 2. Confident detections first.
        matches, unmatched_tracks, leftover_strong = self._associate(
            predicted, boxes, unmatched_tracks, strong, self.match_threshold)

        # 3. Second chance for still-unmatched tracks against weak
        #    detections: a partly occluded person scores low but is still
        #    there, and dropping them is what breaks a count.
        if weak and unmatched_tracks:
            more, unmatched_tracks, _ = self._associate(
                predicted, boxes, unmatched_tracks, weak, self.low_match_threshold)
            matches.extend(more)

        matched_dets = {d for _t, d in matches}
        unmatched_dets = [i for i in leftover_strong if i not in matched_dets]

        # 4. Apply.
        for track_index, det_index in matches:
            candidate = self._candidates[track_index]
            candidate.hits += 1
            candidate.time_since_update = 0
            candidate.place(candidate.kalman.update(boxes[det_index]))

        for track_index in unmatched_tracks:
            candidate = self._candidates[track_index]
            candidate.time_since_update += 1
            candidate.place(predicted[track_index])

        for det_index in unmatched_dets:
            self._candidates.append(_Candidate(self._next_id, boxes[det_index]))
            self._next_id += 1

        self._candidates = [
            c for c in self._candidates
            if c.time_since_update <= self.max_disappeared
        ]
        return self.tracks

    # -- internals -------------------------------------------------------

    def affinity(self, track_boxes, det_boxes):
        """How well each predicted track matches each detection.

        IoU alone is not enough. Once a person moves about their own box
        width between detections -- which happens at any realistic walking
        pace with COUNTER_DETECT_EVERY above 1 -- consecutive boxes barely
        overlap and IoU collapses towards zero. So overlap is combined with
        how close the centres are, measured in box-diagonals so it scales
        with how near the person is to the camera.

        Pairs beyond the distance gate score -1 and are never matched,
        which keeps the shape term from pairing distant strangers.
        """
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)

        iou = iou_batch(track_boxes, det_boxes)

        tracks = np.asarray(track_boxes, dtype=np.float64)
        dets = np.asarray(det_boxes, dtype=np.float64)
        track_centres = np.stack(
            [(tracks[:, 0] + tracks[:, 2]) / 2, (tracks[:, 1] + tracks[:, 3]) / 2], axis=1)
        det_centres = np.stack(
            [(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], axis=1)
        distance = np.linalg.norm(
            track_centres[:, None, :] - det_centres[None, :, :], axis=2)

        widths = np.clip(tracks[:, 2] - tracks[:, 0], 1e-6, None)
        heights = np.clip(tracks[:, 3] - tracks[:, 1], 1e-6, None)
        diagonal = np.sqrt(widths ** 2 + heights ** 2)[:, None]
        gate = np.maximum(self.gate_factor * diagonal, 1e-6)

        proximity = np.clip(1.0 - distance / gate, 0.0, 1.0)
        score = iou + self.proximity_weight * proximity
        return np.where(distance <= gate, score, -1.0).astype(np.float32)

    def _associate(self, predicted, boxes, track_indices, det_indices, threshold):
        """Match a subset of tracks to a subset of detections."""
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)

        track_boxes = [predicted[i] for i in track_indices]
        det_boxes = [boxes[i] for i in det_indices]
        score = self.affinity(track_boxes, det_boxes)

        # Hungarian minimises, so cost is the negated affinity.
        pairs = linear_assignment(-score)

        matches = []
        matched_rows, matched_cols = set(), set()
        for row, col in pairs:
            if score[row, col] < threshold:
                continue
            matches.append((track_indices[row], det_indices[col]))
            matched_rows.add(row)
            matched_cols.add(col)

        unmatched_tracks = [t for i, t in enumerate(track_indices)
                            if i not in matched_rows]
        unmatched_dets = [d for i, d in enumerate(det_indices)
                          if i not in matched_cols]
        return matches, unmatched_tracks, unmatched_dets


def build_tracker(name, max_disappeared=25, detector=None):
    """Build a tracker by name, falling back to the centroid one.

    Two-stage association is only enabled when the detector reports
    calibrated probabilities. HOG returns raw SVM decision values, where a
    fixed confidence split would be meaningless.
    """
    from .vision import CentroidTracker

    name = (name or "sort").strip().lower()
    if name == "centroid":
        return CentroidTracker(max_disappeared=max_disappeared)

    high = None
    if getattr(detector, "score_is_probability", False):
        floor = float(getattr(detector, "min_confidence", 0.35) or 0.35)
        high = max(0.5, floor)

    return SortTracker(max_disappeared=max_disappeared, high_confidence=high)
