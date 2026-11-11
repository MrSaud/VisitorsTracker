"""Tests for tracking, line counting and the API.

A stub detector stands in for HOG/ONNX so nothing here needs a camera,
a model file, or any particular machine speed.
"""

import csv
import http.server
import itertools
import math
import json
import os
import pathlib
import platform
import plistlib
import re
import shutil
import socket
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import time as time_module
from io import StringIO
from unittest.mock import patch
from datetime import date, datetime, time, timedelta, timezone as dt_timezone
from datetime import date as date_cls
from unittest import skipUnless

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core import mail
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone

from . import camera as camera_module
from . import exports
from . import firstrun
from . import secretkey
from config import paths
from . import desktop
from .camera import CounterWorker, is_file_source
from .management.commands import build_app, export_model
from .supervisor import (
    NO_AUTOSTART_COMMANDS,
    Supervisor,
    autostart_enabled,
)
from .daywindow import occupancy_window_start, parse_reset_time
from .tracking import (
    KalmanBox,
    SortTracker,
    build_tracker,
    iou_batch,
    linear_assignment,
)
from .health import (
    DISABLED,
    FAILED,
    FINISHED,
    OFFLINE,
    OK,
    RECONNECTING,
    STALLED,
    STATUS_DEGRADED,
    STATUS_DOWN,
    STATUS_OK,
    camera_health,
    overall_status,
)
from .alerts import (
    Condition,
    alert_channels,
    capacity_conditions,
    current_conditions,
    deliver,
    health_conditions,
    notification_text,
    pending,
    reconcile_alerts,
    run_checks,
    send_pending,
)
from .models import Alert, Camera, Crossing, Event, active_event_for
from .reports import (
    MAX_DAYS,
    busiest_hours,
    daily_counts,
    hourly_profile,
    opening_span,
    weekday_hour_matrix,
    local_today,
    occupancy_peaks,
    parse_date,
    peak_occupancy,
    resolve_range,
    summarise,
)
from .vision import (
    CrossingEvent,
    DEFAULT_HOG_CONFIDENCE,
    DEFAULT_ONNX_CONFIDENCE,
    PERSON_CLASS_ID,
    DIRECTION_IN,
    DIRECTION_OUT,
    HORIZONTAL,
    VERTICAL,
    CentroidTracker,
    DEFAULT_COMMIT_FRACTION,
    HogPersonDetector,
    LINE_POSITION_MAX,
    LINE_POSITION_MIN,
    LineCounter,
    OnnxPersonDetector,
    Track,
    VisitorCounter,
    build_detector,
    non_max_suppression,
    resolve_model_path,
    suggest_line_setup,
)

FRAME_H, FRAME_W = 240, 320


def blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def box_at(cx, cy, w=40, h=80):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class StubDetector:
    """Replays a scripted list of per-frame boxes."""

    name = "stub"

    def __init__(self, script, scores=None):
        self.script = list(script)
        self.scores = list(scores) if scores else None
        self.calls = 0

    def detect(self, frame):
        boxes = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if self.scores is None:
            return [(box, 1.0) for box in boxes]
        # Cycle, so one score can stand for every frame.
        return [(box, self.scores[i % len(self.scores)])
                for i, box in enumerate(boxes)]


class NonMaxSuppressionTests(TestCase):
    def test_empty_input(self):
        self.assertEqual(non_max_suppression([], []), [])

    def test_drops_heavy_overlap_keeps_best(self):
        boxes = [[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 560, 560]]
        scores = [0.9, 0.95, 0.8]
        keep = sorted(non_max_suppression(boxes, scores, iou_threshold=0.4))
        self.assertEqual(len(keep), 2)
        self.assertIn(1, keep)  # the higher-scoring of the overlapping pair
        self.assertIn(2, keep)


class CentroidTrackerTests(TestCase):
    def test_keeps_identity_while_moving(self):
        tracker = CentroidTracker(max_disappeared=5)
        tracker.update([box_at(100, 50)])
        first_id = next(iter(tracker.tracks))

        for y in (60, 70, 80, 90):
            tracker.update([box_at(100, y)])

        self.assertEqual(list(tracker.tracks), [first_id])
        self.assertAlmostEqual(tracker.tracks[first_id].centroid[1], 90, places=3)

    def test_separate_people_get_separate_ids(self):
        tracker = CentroidTracker()
        tracker.update([box_at(60, 100), box_at(260, 100)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_track_expires_after_max_disappeared(self):
        tracker = CentroidTracker(max_disappeared=3)
        tracker.update([box_at(100, 100)])
        self.assertEqual(len(tracker.tracks), 1)
        for _ in range(4):
            tracker.update([])
        self.assertEqual(len(tracker.tracks), 0)

    def test_far_jump_is_treated_as_a_new_person(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=50)
        tracker.update([box_at(40, 40)])
        tracker.update([box_at(300, 220)])
        self.assertEqual(len(tracker.tracks), 2)


class LineCounterTests(TestCase):
    def _run(self, counter, y_values):
        tracker = CentroidTracker(max_disappeared=10)
        events = []
        for y in y_values:
            tracks = tracker.update([box_at(160, y)])
            events.extend(counter.update(tracks, (FRAME_H, FRAME_W, 3)))
        return events

    def test_downward_crossing_counts_in(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL)
        events = self._run(counter, [40, 80, 100, 140, 180, 200])
        self.assertEqual([e.direction for e in events], [DIRECTION_IN])
        self.assertEqual((counter.total_in, counter.total_out), (1, 0))

    def test_upward_crossing_counts_out(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL)
        events = self._run(counter, [200, 180, 140, 100, 80, 40])
        self.assertEqual([e.direction for e in events], [DIRECTION_OUT])
        self.assertEqual((counter.total_in, counter.total_out), (0, 1))

    def test_round_trip_counts_both_directions(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL)
        events = self._run(counter, [40, 100, 180, 200, 180, 100, 40])
        self.assertEqual([e.direction for e in events], [DIRECTION_IN, DIRECTION_OUT])
        self.assertEqual((counter.total_in, counter.total_out), (1, 1))

    def test_jitter_inside_dead_band_never_counts(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL, margin=10)
        # All samples sit within +/-10px of the line at y=120.
        events = self._run(counter, [118, 122, 119, 121, 120, 123, 117])
        self.assertEqual(events, [])
        self.assertEqual((counter.total_in, counter.total_out), (0, 0))

    def test_approach_and_retreat_on_one_side_never_counts(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL, margin=10)
        events = self._run(counter, [40, 80, 105, 80, 40])
        self.assertEqual(events, [])

    def test_invert_swaps_direction(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL, invert=True)
        events = self._run(counter, [40, 100, 180, 200])
        self.assertEqual([e.direction for e in events], [DIRECTION_OUT])

    def test_vertical_line_uses_x_axis(self):
        counter = LineCounter(position=0.5, orientation=VERTICAL)
        tracker = CentroidTracker(max_disappeared=10)
        events = []
        for x in (40, 100, 150, 200, 260):
            tracks = tracker.update([box_at(x, 120)])
            events.extend(counter.update(tracks, (FRAME_H, FRAME_W, 3)))
        self.assertEqual([e.direction for e in events], [DIRECTION_IN])

    def test_line_coordinate_follows_orientation(self):
        horizontal = LineCounter(position=0.25, orientation=HORIZONTAL)
        vertical = LineCounter(position=0.25, orientation=VERTICAL)
        shape = (FRAME_H, FRAME_W, 3)
        self.assertEqual(horizontal.line_coordinate(shape), 60)   # 0.25 * 240
        self.assertEqual(vertical.line_coordinate(shape), 80)     # 0.25 * 320


class VisitorCounterTests(TestCase):
    def test_two_people_crossing_are_counted_once_each(self):
        script = []
        for y in (40, 80, 110, 150, 190, 210):
            script.append([box_at(100, y), box_at(240, y)])
        detector = StubDetector(script)
        counter = VisitorCounter(
            detector=detector,
            line=LineCounter(position=0.5, orientation=HORIZONTAL),
            detect_every=1,
            frame_width=FRAME_W,
        )

        total_events = []
        for _ in script:
            _annotated, events = counter.process(blank_frame())
            total_events.extend(events)

        self.assertEqual(len(total_events), 2)
        self.assertEqual({e.direction for e in total_events}, {DIRECTION_IN})
        self.assertEqual(counter.total_in, 2)
        self.assertEqual(counter.occupancy, 2)

    def test_occupancy_never_goes_negative(self):
        counter = VisitorCounter(
            detector=StubDetector([[]]),
            line=LineCounter(),
            detect_every=1,
            frame_width=FRAME_W,
        )
        counter.line.total_out = 5
        self.assertEqual(counter.occupancy, 0)

    def test_annotated_frame_keeps_shape_and_draws(self):
        counter = VisitorCounter(
            detector=StubDetector([[box_at(160, 60)]]),
            line=LineCounter(),
            detect_every=1,
            frame_width=FRAME_W,
        )
        annotated, _events = counter.process(blank_frame())
        self.assertEqual(annotated.shape, (FRAME_H, FRAME_W, 3))
        # The banner and line mean the frame is no longer all zeros.
        self.assertGreater(int(annotated.sum()), 0)

    def test_detect_every_skips_detection_but_keeps_tracking(self):
        detector = StubDetector([[box_at(160, 60)]])
        counter = VisitorCounter(
            detector=detector, line=LineCounter(), detect_every=3, frame_width=FRAME_W
        )
        for _ in range(6):
            counter.process(blank_frame())
        self.assertEqual(detector.calls, 2)  # frames 0 and 3


class HogDetectorTests(TestCase):
    def test_hog_detector_runs_on_a_real_frame(self):
        """Smoke test: the built-in detector loads and returns a usable list."""
        detector = HogPersonDetector(min_confidence=0.45)
        detections = detector.detect(np.zeros((256, 192, 3), dtype=np.uint8))
        self.assertIsInstance(detections, list)
        for box, score in detections:
            self.assertEqual(len(box), 4)
            self.assertIsInstance(score, float)


class CameraModelTests(TestCase):
    def test_webcam_index_resolves_to_int(self):
        self.assertEqual(Camera(source="0").resolved_source(), 0)
        self.assertEqual(Camera(source=" 2 ").resolved_source(), 2)

    def test_url_source_stays_a_string(self):
        camera = Camera(source="rtsp://host/stream")
        self.assertEqual(camera.resolved_source(), "rtsp://host/stream")

    def test_totals_derive_from_events(self):
        camera = Camera.objects.create(name="Door", source="0")
        for direction in (DIRECTION_IN, DIRECTION_IN, DIRECTION_IN, DIRECTION_OUT):
            Crossing.objects.create(camera=camera, track_id=1, direction=direction)
        self.assertEqual(camera.totals(), {"in": 3, "out": 1})
        self.assertEqual(camera.occupancy()["inside"], 2)

    def test_inside_clamps_at_zero(self):
        camera = Camera.objects.create(name="Door", source="0")
        Crossing.objects.create(camera=camera, track_id=1, direction=DIRECTION_OUT)
        occupancy = camera.occupancy()
        self.assertEqual(occupancy["inside"], 0)
        # The raw net stays visible so a reversed camera is diagnosable.
        self.assertEqual(occupancy["net"], -1)


class ApiTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Front door", source="0")
        Crossing.objects.create(camera=self.camera, track_id=1, direction=DIRECTION_IN)
        Crossing.objects.create(camera=self.camera, track_id=2, direction=DIRECTION_IN)
        Crossing.objects.create(camera=self.camera, track_id=3, direction=DIRECTION_OUT)

    def test_dashboard_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visitors per date")

    def test_cameras_page_lists_the_cameras(self):
        # The live views have a page of their own; the dashboard no
        # longer carries them.
        response = self.client.get("/cameras/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Front door")
        self.assertNotContains(self.client.get("/"), "Front door")

    def test_overall_stats(self):
        response = self.client.get("/api/v1/stats/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["totals"], {"in": 2, "out": 1})
        self.assertEqual(body["occupancy"]["inside"], 1)
        self.assertEqual(body["running_cameras"], 0)
        self.assertEqual(len(body["cameras"]), 1)

    def test_camera_list_and_detail(self):
        self.assertEqual(self.client.get("/api/v1/cameras/").status_code, 200)
        detail = self.client.get(f"/api/v1/cameras/{self.camera.pk}/")
        self.assertEqual(detail.status_code, 200)
        self.assertFalse(detail.json()["running"])

    def test_crossings_endpoint_respects_limit(self):
        response = self.client.get(f"/api/v1/cameras/{self.camera.pk}/crossings/?limit=2")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 2)

    def test_reset_deletes_events(self):
        self.client.force_login(
            User.objects.create_user("op", password="x", is_staff=True))
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 3)
        self.assertEqual(Crossing.objects.count(), 0)

    def test_stream_404_when_not_running(self):
        response = self.client.get(f"/api/v1/cameras/{self.camera.pk}/stream/")
        self.assertEqual(response.status_code, 404)

    def test_unknown_camera_is_404(self):
        self.assertEqual(self.client.get("/api/v1/cameras/9999/").status_code, 404)


class WorkerEndToEndTests(TransactionTestCase):
    """Drives the real worker thread over a real video file.

    Only the detector is stubbed -- the capture, frame loop, annotation,
    JPEG encoding and database writes are all the production code paths.
    ``TransactionTestCase`` is required because the worker writes from
    another thread, outside the test's transaction.
    """

    @staticmethod
    def _write_video(path, frames=36):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (FRAME_W, FRAME_H)
        )
        if not writer.isOpened():
            return False
        for index in range(frames):
            frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
            # A moving blob so the frames differ; the stub detector is what counts.
            y = int(40 + (160 * index / max(1, frames - 1)))
            cv2.rectangle(frame, (140, y - 40), (180, y + 40), (200, 200, 200), -1)
            writer.write(frame)
        writer.release()
        return path.exists() and path.stat().st_size > 0

    def test_worker_counts_from_a_video_file_and_saves_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = pathlib.Path(tmpdir) / "walk.avi"
            if not self._write_video(video):
                self.skipTest("OpenCV build cannot write MJPG video here")

            camera = Camera.objects.create(name="Clip", source=str(video))
            worker = CounterWorker(camera)

            # One person walking straight down across the line, frame by frame.
            script = [[box_at(160, int(40 + 160 * i / 35))] for i in range(36)]
            worker.counter.detector = StubDetector(script)
            worker.counter.detect_every = 1

            worker.start()
            worker.join(timeout=30)

            self.assertFalse(worker.is_alive(), "worker did not finish the clip")
            self.assertIsNone(worker.error)
            self.assertGreater(worker.frames_read, 0)

            # The annotated frame really was encoded as a JPEG.
            jpeg = worker.snapshot()
            self.assertIsNotNone(jpeg)
            self.assertTrue(jpeg.startswith(b"\xff\xd8"), "not JPEG SOI")
            self.assertTrue(jpeg.rstrip().endswith(b"\xff\xd9"), "not JPEG EOI")

            # Exactly one crossing, persisted, and counted as "in".
            self.assertEqual(worker.counter.total_in, 1)
            self.assertEqual(worker.counter.total_out, 0)

            events = list(Crossing.objects.filter(camera=camera))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].direction, DIRECTION_IN)
            self.assertEqual(camera.totals(), {"in": 1, "out": 0})
            self.assertEqual(camera.occupancy()["inside"], 1)

    def test_unopenable_source_reports_an_error_instead_of_crashing(self):
        camera = Camera.objects.create(name="Missing", source="/no/such/clip.mp4")
        worker = CounterWorker(camera)
        worker.start()
        worker.join(timeout=15)

        self.assertFalse(worker.is_alive())
        self.assertIsNotNone(worker.error)
        self.assertIn("Could not open video source", worker.error)
        self.assertEqual(Crossing.objects.count(), 0)


class OnnxDecodeTests(TestCase):
    """Decode maths for the YOLOv8 head, exercised without the model file.

    The network is stubbed so these run on any machine; the separate
    ``OnnxModelTests`` cover the real exported model when it is present.
    """

    INPUT = 640

    def _detector(self, forward_output):
        """An OnnxPersonDetector with a canned forward pass."""
        detector = object.__new__(OnnxPersonDetector)
        detector.min_confidence = 0.45
        detector.input_size = self.INPUT

        class FakeNet:
            def setInput(self, blob):
                self.blob = blob

            def forward(self_inner):
                return forward_output

        detector._net = FakeNet()
        return detector

    @staticmethod
    def _row(cx, cy, w, h, person_score, num_classes=80, other=None):
        row = np.zeros(4 + num_classes, dtype=np.float32)
        row[:4] = (cx, cy, w, h)
        row[4 + PERSON_CLASS_ID] = person_score
        if other is not None:
            index, score = other
            row[4 + index] = score
        return row

    @staticmethod
    def _output(rows, total=8400):
        """Shape real rows like an actual YOLOv8 head: (1, 84, 8400).

        Padding to the true candidate count matters -- the decoder works out
        tensor orientation from which axis is larger, so a toy 2-row tensor
        would be read sideways and prove nothing.
        """
        rows = np.stack(rows).astype(np.float32)
        pad = np.zeros((total - len(rows), rows.shape[1]), dtype=np.float32)
        return np.concatenate([rows, pad], axis=0).T[np.newaxis, ...]

    def test_box_maps_back_through_letterbox(self):
        # 640x480 frame -> scale 1.0, dy 80 (letterboxed top and bottom).
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Target box (100, 100) - (200, 300): centre (150, 200), 100x200.
        # In padded space that centre sits at y = 200 + 80 = 280.
        rows = [
            self._row(150, 280, 100, 200, 0.92),
            self._row(400, 400, 50, 50, 0.10),          # below threshold
        ]
        detections = self._detector(self._output(rows)).detect(frame)

        self.assertEqual(len(detections), 1)
        box, score = detections[0]
        self.assertAlmostEqual(score, 0.92, places=4)
        for actual, expected in zip(box, (100.0, 100.0, 200.0, 300.0)):
            self.assertAlmostEqual(actual, expected, delta=0.5)

    def test_non_person_winning_class_is_rejected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Person score clears the bar, but class 2 (car) wins the row.
        rows = [self._row(150, 280, 100, 200, 0.50, other=(2, 0.95))]
        detections = self._detector(self._output(rows)).detect(frame)
        self.assertEqual(detections, [])

    def test_boxes_are_clipped_to_the_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # A huge box centred near the top-left spills past the edges.
        rows = [self._row(20, 100, 400, 400, 0.90)]
        detections = self._detector(self._output(rows)).detect(frame)
        self.assertEqual(len(detections), 1)
        x1, y1, x2, y2 = detections[0][0]
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, 640.0)
        self.assertLessEqual(y2, 480.0)

    def test_overlapping_duplicates_are_suppressed(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rows = [
            self._row(150, 280, 100, 200, 0.80),
            self._row(152, 282, 100, 200, 0.95),   # near-identical, higher score
            self._row(500, 280, 100, 200, 0.85),   # well separated
        ]
        detections = self._detector(self._output(rows)).detect(frame)
        self.assertEqual(len(detections), 2)
        self.assertAlmostEqual(max(s for _b, s in detections), 0.95, places=4)

    def test_empty_and_malformed_output_are_handled(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        nothing = [self._row(150, 280, 100, 200, 0.01)]
        self.assertEqual(self._detector(self._output(nothing)).detect(frame), [])
        # A 1-D output is not a detection head at all.
        self.assertEqual(self._detector(np.zeros((10,), dtype=np.float32)).detect(frame), [])


class DetectorSelectionTests(TestCase):
    def test_hog_is_forced_when_requested(self):
        detector = build_detector("hog", 0.45)
        self.assertIsInstance(detector, HogPersonDetector)

    def test_each_backend_has_its_own_default_threshold(self):
        hog = HogPersonDetector()
        self.assertAlmostEqual(hog.min_confidence, DEFAULT_HOG_CONFIDENCE)
        self.assertNotAlmostEqual(DEFAULT_HOG_CONFIDENCE, DEFAULT_ONNX_CONFIDENCE)

    def test_explicit_threshold_overrides_the_default(self):
        self.assertAlmostEqual(HogPersonDetector(0.8).min_confidence, 0.8)

    def test_missing_model_falls_back_to_hog(self):
        detector = build_detector("onnx", 0.45, model_name="definitely-not-here.onnx")
        self.assertIsInstance(detector, HogPersonDetector)

    def test_auto_without_model_is_hog(self):
        detector = build_detector("auto", 0.45, model_name="definitely-not-here.onnx")
        self.assertIsInstance(detector, HogPersonDetector)

    @skipUnless(resolve_model_path().exists(), "yolov8n.onnx not exported")
    def test_auto_prefers_the_model_when_present(self):
        detector = build_detector("auto", 0.45)
        self.assertIsInstance(detector, OnnxPersonDetector)
        self.assertEqual(detector.name, "onnx")


@skipUnless(resolve_model_path().exists(), "yolov8n.onnx not exported")
class OnnxModelTests(TestCase):
    """Runs the genuinely exported model. Skipped when it has not been built."""

    FIXTURE = pathlib.Path(__file__).resolve().parent / "ml_models" / "people.jpg"

    def test_model_loads_and_runs_on_a_blank_frame(self):
        detector = OnnxPersonDetector(resolve_model_path())
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsInstance(detections, list)
        self.assertEqual(detections, [], "a blank frame should contain no people")

    @skipUnless(FIXTURE.exists(), "no people.jpg fixture")
    def test_finds_real_people_in_a_photograph(self):
        image = cv2.imread(str(self.FIXTURE))
        self.assertIsNotNone(image, "fixture could not be decoded")

        # Default threshold, i.e. what the app actually runs with.
        detector = OnnxPersonDetector(resolve_model_path())
        detections = detector.detect(image)

        # The fixture holds four people; the faintest scores ~0.44, which the
        # old shared 0.45 cutoff used to discard.
        self.assertGreaterEqual(len(detections), 4, "expected all four people")
        height, width = image.shape[:2]
        for box, score in detections:
            x1, y1, x2, y2 = box
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)
            self.assertGreaterEqual(score, detector.min_confidence)
            self.assertLessEqual(x2, width + 1)
            self.assertLessEqual(y2, height + 1)


class SuggestLineSetupTests(TestCase):
    """The recommendation that tells someone why their camera never counts."""

    SHAPE = (360, 640, 3)

    def test_walking_across_suggests_a_vertical_line(self):
        # Someone crossing a desk webcam: x sweeps, y barely moves.
        centroids = [(x, 180 + (x % 7)) for x in range(60, 580, 20)]
        result = suggest_line_setup(centroids, self.SHAPE)
        self.assertTrue(result["confident"])
        self.assertEqual(result["orientation"], VERTICAL)

    def test_walking_toward_the_camera_suggests_a_horizontal_line(self):
        centroids = [(320 + (y % 5), y) for y in range(40, 330, 12)]
        result = suggest_line_setup(centroids, self.SHAPE)
        self.assertTrue(result["confident"])
        self.assertEqual(result["orientation"], HORIZONTAL)

    def test_position_sits_at_the_middle_of_observed_travel(self):
        # Travel spans x 100..500 on a 640-wide frame -> midpoint 300/640.
        centroids = [(x, 180) for x in range(100, 501, 20)]
        result = suggest_line_setup(centroids, self.SHAPE)
        self.assertEqual(result["orientation"], VERTICAL)
        self.assertAlmostEqual(result["position"], 300 / 640, places=2)

    def test_travel_direction_is_reported(self):
        rightwards = suggest_line_setup([(x, 180) for x in range(60, 580, 20)], self.SHAPE)
        self.assertEqual(rightwards["travel"], "increasing")
        leftwards = suggest_line_setup([(x, 180) for x in range(560, 40, -20)], self.SHAPE)
        self.assertEqual(leftwards["travel"], "decreasing")

    def test_standing_still_is_not_confident(self):
        centroids = [(320 + (i % 3), 180 + (i % 3)) for i in range(40)]
        result = suggest_line_setup(centroids, self.SHAPE)
        self.assertFalse(result["confident"])
        self.assertIsNone(result["orientation"])

    def test_too_few_samples_returns_none(self):
        self.assertIsNone(suggest_line_setup([], self.SHAPE))
        self.assertIsNone(suggest_line_setup([(1.0, 2.0)], self.SHAPE))

    def test_axis_choice_is_normalised_by_frame_shape(self):
        # 200px of travel is a third of the width but over half the height,
        # so the dominant axis here is vertical despite equal pixel spans.
        centroids = [(100 + i * 20, 80 + i * 20) for i in range(11)]
        result = suggest_line_setup(centroids, (360, 640, 3))
        self.assertEqual(result["orientation"], HORIZONTAL)

    def test_position_is_clamped_into_frame(self):
        centroids = [(x, 180) for x in range(-200, 40, 10)]
        result = suggest_line_setup(centroids, self.SHAPE)
        self.assertGreaterEqual(result["position"], 0.05)
        self.assertLessEqual(result["position"], 0.95)


class DailyRangeTests(TestCase):
    def test_days_alone_ends_today(self):
        start, end = resolve_range(days=7)
        self.assertEqual(end, local_today())
        self.assertEqual((end - start).days, 6)      # inclusive of both ends

    def test_start_plus_days(self):
        start, end = resolve_range(start=date(2026, 3, 1), days=5)
        self.assertEqual((start, end), (date(2026, 3, 1), date(2026, 3, 5)))

    def test_end_plus_days(self):
        start, end = resolve_range(end=date(2026, 3, 10), days=3)
        self.assertEqual((start, end), (date(2026, 3, 8), date(2026, 3, 10)))

    def test_explicit_window_is_respected(self):
        start, end = resolve_range(start=date(2026, 3, 1), end=date(2026, 3, 4))
        self.assertEqual((start, end), (date(2026, 3, 1), date(2026, 3, 4)))

    def test_backwards_window_is_swapped(self):
        start, end = resolve_range(start=date(2026, 3, 9), end=date(2026, 3, 2))
        self.assertEqual((start, end), (date(2026, 3, 2), date(2026, 3, 9)))

    def test_window_is_capped(self):
        start, end = resolve_range(start=date(2000, 1, 1), end=date(2026, 1, 1))
        self.assertEqual((end - start).days + 1, MAX_DAYS)

    def test_days_is_clamped(self):
        _s, _e = resolve_range(days=99999)
        self.assertEqual((_e - _s).days + 1, MAX_DAYS)


class DailyCountsTests(TestCase):
    """Bucketing by local date, which is the whole point of this report."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.other = Camera.objects.create(name="Side", source="1")

    def _event(self, when, direction=DIRECTION_IN, camera=None):
        return Crossing.objects.create(
            camera=camera or self.camera, track_id=1,
            direction=direction, occurred_at=when,
        )

    def test_late_night_utc_lands_on_the_local_date(self):
        # 22:30 UTC on the 1st is 01:30 on the 2nd in Kuwait (UTC+3).
        self._event(datetime(2026, 3, 1, 22, 30, tzinfo=dt_timezone.utc))
        rows = daily_counts(start=date(2026, 3, 1), end=date(2026, 3, 2))
        by_date = {r["date"]: r for r in rows}
        self.assertEqual(by_date["2026-03-01"]["in"], 0, "must not use the UTC date")
        self.assertEqual(by_date["2026-03-02"]["in"], 1)

    def test_start_of_local_day_is_included(self):
        # 21:00 UTC == 00:00 local on the 2nd: the first instant of that day.
        self._event(datetime(2026, 3, 1, 21, 0, tzinfo=dt_timezone.utc))
        rows = daily_counts(start=date(2026, 3, 2), end=date(2026, 3, 2))
        self.assertEqual(rows[0]["in"], 1)

    def test_just_before_local_day_is_excluded(self):
        self._event(datetime(2026, 3, 1, 20, 59, tzinfo=dt_timezone.utc))
        rows = daily_counts(start=date(2026, 3, 2), end=date(2026, 3, 2))
        self.assertEqual(rows[0]["in"], 0)

    def test_quiet_days_are_zero_filled(self):
        self._event(datetime(2026, 3, 1, 9, 0, tzinfo=dt_timezone.utc))
        self._event(datetime(2026, 3, 4, 9, 0, tzinfo=dt_timezone.utc))
        rows = daily_counts(start=date(2026, 3, 1), end=date(2026, 3, 5))
        self.assertEqual(len(rows), 5)
        self.assertEqual([r["in"] for r in rows], [1, 0, 0, 1, 0])
        self.assertEqual([r["date"] for r in rows][1], "2026-03-02")

    def test_in_and_out_are_separated(self):
        for _ in range(3):
            self._event(datetime(2026, 3, 3, 9, 0, tzinfo=dt_timezone.utc))
        self._event(datetime(2026, 3, 3, 10, 0, tzinfo=dt_timezone.utc),
                    direction=DIRECTION_OUT)
        rows = daily_counts(start=date(2026, 3, 3), end=date(2026, 3, 3))
        self.assertEqual((rows[0]["in"], rows[0]["out"], rows[0]["net"]), (3, 1, 2))

    def test_camera_filter(self):
        self._event(datetime(2026, 3, 3, 9, 0, tzinfo=dt_timezone.utc))
        self._event(datetime(2026, 3, 3, 9, 0, tzinfo=dt_timezone.utc), camera=self.other)
        everything = daily_counts(start=date(2026, 3, 3), end=date(2026, 3, 3))
        one = daily_counts(start=date(2026, 3, 3), end=date(2026, 3, 3), camera=self.camera)
        self.assertEqual(everything[0]["in"], 2)
        self.assertEqual(one[0]["in"], 1)

    def test_events_outside_the_window_are_ignored(self):
        self._event(datetime(2026, 2, 20, 9, 0, tzinfo=dt_timezone.utc))
        rows = daily_counts(start=date(2026, 3, 1), end=date(2026, 3, 3))
        self.assertEqual(sum(r["in"] for r in rows), 0)


class SummariseTests(TestCase):
    def test_totals_and_busiest_day(self):
        rows = [
            {"date": "2026-03-01", "in": 2, "out": 1, "net": 1},
            {"date": "2026-03-02", "in": 9, "out": 4, "net": 5},
            {"date": "2026-03-03", "in": 1, "out": 0, "net": 1},
        ]
        summary = summarise(rows)
        self.assertEqual(summary["in"], 12)
        self.assertEqual(summary["out"], 5)
        self.assertEqual(summary["busiest_date"], "2026-03-02")
        self.assertEqual(summary["busiest_in"], 9)
        self.assertEqual(summary["daily_average_in"], 4.0)

    def test_all_zero_has_no_busiest_day(self):
        rows = [{"date": "2026-03-01", "in": 0, "out": 0, "net": 0}]
        self.assertIsNone(summarise(rows)["busiest_date"])

    def test_empty_report(self):
        self.assertEqual(summarise([])["in"], 0)


class ParseDateTests(TestCase):
    def test_valid(self):
        self.assertEqual(parse_date("2026-03-07"), date(2026, 3, 7))

    def test_blank_is_none(self):
        self.assertIsNone(parse_date(""))
        self.assertIsNone(parse_date(None))

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_date("07/03/2026")
        with self.assertRaises(ValueError):
            parse_date("not-a-date")


class DailyApiTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        for _ in range(4):
            Crossing.objects.create(
                camera=self.camera, track_id=1, direction=DIRECTION_IN,
                occurred_at=datetime(2026, 3, 3, 9, 0, tzinfo=dt_timezone.utc),
            )

    def test_explicit_window(self):
        response = self.client.get("/api/v1/daily/?start=2026-03-01&end=2026-03-05")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["start"], "2026-03-01")
        self.assertEqual(body["end"], "2026-03-05")
        self.assertEqual(len(body["days"]), 5)
        self.assertEqual(body["summary"]["in"], 4)

    def test_default_window_is_thirty_days(self):
        body = self.client.get("/api/v1/daily/").json()
        self.assertEqual(len(body["days"]), 30)
        self.assertEqual(body["end"], local_today().isoformat())

    def test_camera_filter_is_reported_back(self):
        body = self.client.get(
            f"/api/v1/daily/?camera={self.camera.pk}&start=2026-03-03&end=2026-03-03"
        ).json()
        self.assertEqual(body["camera_id"], self.camera.pk)
        self.assertEqual(body["camera_name"], "Door")
        self.assertEqual(body["days"][0]["in"], 4)

    def test_csv_export(self):
        response = self.client.get(
            "/api/v1/daily/?start=2026-03-03&end=2026-03-04&format=csv"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("attachment;", response["Content-Disposition"])
        lines = response.content.decode().strip().splitlines()
        self.assertEqual(lines[0], "event,date,in,out,net,peak_inside,peak_at")
        # Four arrivals, none of whom left, so all four were inside at once.
        row = next(csv.reader([lines[1]]))
        self.assertEqual(row[:6], ["All events", "2026-03-03", "4", "0", "4", "4"])
        self.assertEqual(lines[2], "All events,2026-03-04,0,0,0,0,")

    def test_bad_date_is_400(self):
        response = self.client.get("/api/v1/daily/?start=07-03-2026")
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_bad_days_is_400(self):
        self.assertEqual(self.client.get("/api/v1/daily/?days=lots").status_code, 400)
        self.assertEqual(self.client.get("/api/v1/daily/?days=0").status_code, 400)

    def test_unknown_camera_is_400(self):
        self.assertEqual(self.client.get("/api/v1/daily/?camera=9999").status_code, 400)

    def test_dashboard_includes_the_panel(self):
        response = self.client.get("/")
        self.assertContains(response, "Visitors per date")


class OccupancyWindowTests(TestCase):
    """The daily boundary occupancy is measured from."""

    def test_parses_hh_mm(self):
        self.assertEqual(parse_reset_time("06:30"), time(6, 30))
        self.assertEqual(parse_reset_time("0:00"), time(0, 0))
        self.assertEqual(parse_reset_time("23:59"), time(23, 59))

    def test_rubbish_falls_back_to_midnight(self):
        for bad in ("", "nonsense", "25:99", None, "6:xx"):
            self.assertEqual(parse_reset_time(bad), time(0, 0))

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="06:00")
    def test_boundary_is_today_when_reset_has_passed(self):
        now = timezone.make_aware(datetime(2026, 3, 10, 9, 0))
        start = occupancy_window_start(now)
        self.assertEqual(timezone.localtime(start).date(), date(2026, 3, 10))
        self.assertEqual(timezone.localtime(start).hour, 6)

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="06:00")
    def test_boundary_is_yesterday_before_the_reset(self):
        # 03:00 is before the 06:00 reset, so the window began yesterday.
        now = timezone.make_aware(datetime(2026, 3, 10, 3, 0))
        start = occupancy_window_start(now)
        self.assertEqual(timezone.localtime(start).date(), date(2026, 3, 9))
        self.assertEqual(timezone.localtime(start).hour, 6)

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_midnight_reset_matches_the_local_date(self):
        now = timezone.make_aware(datetime(2026, 3, 10, 13, 45))
        start = occupancy_window_start(now)
        self.assertEqual(timezone.localtime(start).date(), date(2026, 3, 10))
        self.assertEqual(timezone.localtime(start).hour, 0)


class OccupancyDoesNotDriftTests(TestCase):
    """The defect this fix exists for: missed exits must not accumulate."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _event(self, when, direction):
        Crossing.objects.create(camera=self.camera, track_id=1,
                                  direction=direction, occurred_at=when)

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_yesterdays_missed_exits_do_not_inflate_today(self):
        yesterday = timezone.localtime() - timedelta(days=1)
        # 50 entries yesterday, every single exit missed.
        for _ in range(50):
            self._event(yesterday, DIRECTION_IN)

        # Today: two in, one out.
        now = timezone.localtime()
        self._event(now, DIRECTION_IN)
        self._event(now, DIRECTION_IN)
        self._event(now, DIRECTION_OUT)

        self.assertEqual(self.camera.totals(), {"in": 52, "out": 1},
                         "lifetime footfall still counts everything")
        self.assertEqual(self.camera.occupancy()["inside"], 1,
                         "occupancy must reset daily, not carry 50 ghosts")

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_drift_across_many_days_is_bounded(self):
        # Three weeks of 20 entries/day with every exit missed.
        for days_back in range(21, 0, -1):
            when = timezone.localtime() - timedelta(days=days_back)
            for _ in range(20):
                self._event(when, DIRECTION_IN)

        self.assertEqual(self.camera.totals()["in"], 420)
        self.assertEqual(self.camera.occupancy()["inside"], 0,
                         "420 un-exited visitors must not show as inside")

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_negative_net_is_visible_but_inside_clamps(self):
        self._event(timezone.localtime(), DIRECTION_OUT)
        occupancy = self.camera.occupancy()
        self.assertEqual(occupancy["inside"], 0)
        self.assertEqual(occupancy["net"], -1)

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_stats_endpoint_reports_occupancy_not_lifetime(self):
        yesterday = timezone.localtime() - timedelta(days=1)
        for _ in range(30):
            self._event(yesterday, DIRECTION_IN)
        self._event(timezone.localtime(), DIRECTION_IN)

        body = self.client.get("/api/v1/stats/").json()
        self.assertEqual(body["totals"]["in"], 31)
        self.assertEqual(body["occupancy"]["inside"], 1)
        self.assertEqual(body["occupancy"]["resets_at"], "00:00")


@override_settings(COUNTER_API_TOKEN="test-token-value", COUNTER_PUBLIC_READS=True)
class ApiAuthTests(TestCase):
    """Reads open, writes closed."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        Crossing.objects.create(camera=self.camera, track_id=1,
                                  direction=DIRECTION_IN)
        self.staff = User.objects.create_user("op", password="x", is_staff=True)
        self.normal = User.objects.create_user("visitor", password="x")

    # -- reads ----------------------------------------------------------

    def test_anonymous_may_read(self):
        for path in ("/api/v1/stats/", "/api/v1/cameras/", "/api/v1/daily/"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    @override_settings(COUNTER_PUBLIC_READS=False)
    def test_reads_can_be_closed(self):
        self.assertEqual(self.client.get("/api/v1/stats/").status_code, 401)

    @override_settings(COUNTER_PUBLIC_READS=False)
    def test_staff_may_read_when_reads_are_closed(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.client.get("/api/v1/stats/").status_code, 200)

    # -- writes ---------------------------------------------------------

    def test_anonymous_cannot_reset(self):
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(Crossing.objects.count(), 1, "history must survive")

    def test_anonymous_cannot_stop_a_camera(self):
        self.assertEqual(
            self.client.post(f"/api/v1/cameras/{self.camera.pk}/stop/").status_code, 401)

    def test_non_staff_user_cannot_write(self):
        self.client.force_login(self.normal)
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(Crossing.objects.count(), 1)

    def test_staff_may_write(self):
        self.client.force_login(self.staff)
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Crossing.objects.count(), 0)

    def test_valid_token_may_write(self):
        response = self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/reset/",
            HTTP_AUTHORIZATION="Token test-token-value",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Crossing.objects.count(), 0)

    def test_wrong_token_is_rejected(self):
        response = self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/reset/",
            HTTP_AUTHORIZATION="Token not-the-token",
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(Crossing.objects.count(), 1)

    def test_malformed_authorization_header_is_rejected(self):
        response = self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/reset/",
            HTTP_AUTHORIZATION="Token",
        )
        self.assertEqual(response.status_code, 401)

    @override_settings(COUNTER_API_TOKEN="")
    def test_token_auth_refuses_when_unconfigured(self):
        """An empty setting must not make every token valid."""
        response = self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/reset/",
            HTTP_AUTHORIZATION="Token anything-at-all",
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(Crossing.objects.count(), 1)

    # -- dashboard ------------------------------------------------------

    def test_camera_controls_are_hidden_from_anonymous(self):
        response = self.client.get("/cameras/")
        self.assertNotContains(response, 'data-action="reset"')
        self.assertContains(response, "Read-only")

    def test_camera_controls_are_shown_to_staff(self):
        self.client.force_login(self.staff)
        response = self.client.get("/cameras/")
        self.assertContains(response, 'data-action="reset"')

    @override_settings(COUNTER_PUBLIC_READS=False)
    def test_stream_is_closed_when_reads_are(self):
        self.assertEqual(
            self.client.get(f"/api/v1/cameras/{self.camera.pk}/stream/").status_code, 403)


class HourlyProfileTests(TestCase):
    """Hour-of-day aggregation, in local time."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.other = Camera.objects.create(name="Side", source="1")

    def _event(self, when, direction=DIRECTION_IN, camera=None):
        Crossing.objects.create(camera=camera or self.camera, track_id=1,
                                  direction=direction, occurred_at=when)

    def _profile(self, **kwargs):
        kwargs.setdefault("start", date(2026, 3, 2))
        kwargs.setdefault("end", date(2026, 3, 2))
        return {r["hour"]: r for r in hourly_profile(**kwargs)}

    def test_hours_are_local_not_utc(self):
        # 22:30 UTC on the 1st is 01:30 on the 2nd in Kuwait.
        self._event(datetime(2026, 3, 1, 22, 30, tzinfo=dt_timezone.utc))
        profile = self._profile()
        self.assertEqual(profile[1]["in"], 1)
        self.assertEqual(profile[22]["in"], 0, "must not bucket by the UTC hour")

    def test_all_twenty_four_hours_are_present(self):
        rows = hourly_profile(start=date(2026, 3, 2), end=date(2026, 3, 2))
        self.assertEqual(len(rows), 24)
        self.assertEqual([r["hour"] for r in rows], list(range(24)))

    def test_labels_are_zero_padded(self):
        rows = hourly_profile(start=date(2026, 3, 2), end=date(2026, 3, 2))
        self.assertEqual(rows[0]["label"], "00:00")
        self.assertEqual(rows[9]["label"], "09:00")
        self.assertEqual(rows[23]["label"], "23:00")

    def test_in_and_out_separated_per_hour(self):
        morning = datetime(2026, 3, 2, 6, 0, tzinfo=dt_timezone.utc)   # 09:00 local
        for _ in range(5):
            self._event(morning)
        self._event(morning, direction=DIRECTION_OUT)
        profile = self._profile()
        self.assertEqual((profile[9]["in"], profile[9]["out"]), (5, 1))

    def test_average_divides_by_days_in_range(self):
        # 10 arrivals at 09:00 local spread over 5 days -> average 2.0/day.
        for day in range(2, 7):
            for _ in range(2):
                self._event(datetime(2026, 3, day, 6, 0, tzinfo=dt_timezone.utc))
        profile = self._profile(start=date(2026, 3, 2), end=date(2026, 3, 6))
        self.assertEqual(profile[9]["in"], 10)
        self.assertEqual(profile[9]["average_in"], 2.0)

    def test_share_is_a_percentage_of_arrivals(self):
        self._event(datetime(2026, 3, 2, 6, 0, tzinfo=dt_timezone.utc))   # 09:00
        self._event(datetime(2026, 3, 2, 6, 0, tzinfo=dt_timezone.utc))   # 09:00
        self._event(datetime(2026, 3, 2, 9, 0, tzinfo=dt_timezone.utc))   # 12:00
        self._event(datetime(2026, 3, 2, 10, 0, tzinfo=dt_timezone.utc))  # 13:00
        profile = self._profile()
        self.assertEqual(profile[9]["share"], 50.0)
        self.assertEqual(profile[12]["share"], 25.0)

    def test_camera_filter(self):
        when = datetime(2026, 3, 2, 6, 0, tzinfo=dt_timezone.utc)
        self._event(when)
        self._event(when, camera=self.other)
        self.assertEqual(self._profile()[9]["in"], 2)
        self.assertEqual(self._profile(camera=self.camera)[9]["in"], 1)

    def test_empty_range_is_all_zeros(self):
        profile = self._profile()
        self.assertEqual(sum(r["in"] for r in profile.values()), 0)
        self.assertTrue(all(r["share"] == 0.0 for r in profile.values()))


class BusiestHoursTests(TestCase):
    def _profile(self, counts):
        return [
            {"hour": hour, "label": f"{hour:02d}:00", "in": counts.get(hour, 0),
             "out": 0, "average_in": 0.0, "share": 0.0}
            for hour in range(24)
        ]

    def test_returns_busiest_first(self):
        rows = busiest_hours(self._profile({9: 40, 10: 12, 11: 25, 14: 3}))
        self.assertEqual([r["hour"] for r in rows], [9, 11, 10])

    def test_silent_hours_are_excluded(self):
        rows = busiest_hours(self._profile({9: 2}))
        self.assertEqual(len(rows), 1)

    def test_no_data_gives_empty_list(self):
        self.assertEqual(busiest_hours(self._profile({})), [])

    def test_opening_span_spans_first_to_last_arrival(self):
        span = opening_span(self._profile({7: 3, 9: 20, 17: 4}))
        self.assertEqual(span, {"first_hour": 7, "last_hour": 17})

    def test_opening_span_is_none_when_quiet(self):
        self.assertIsNone(opening_span(self._profile({})))


class WeekdayHourMatrixTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _event(self, when):
        Crossing.objects.create(camera=self.camera, track_id=1,
                                  direction=DIRECTION_IN, occurred_at=when)

    def test_shape_is_seven_days_by_twenty_four_hours(self):
        matrix = weekday_hour_matrix(start=date(2026, 3, 2), end=date(2026, 3, 8))
        self.assertEqual(len(matrix), 7)
        self.assertEqual([d["name"] for d in matrix][0], "Monday")
        self.assertTrue(all(len(d["hours"]) == 24 for d in matrix))

    def test_average_divides_by_occurrences_of_that_weekday(self):
        # 2026-03-02 and 2026-03-09 are both Mondays.
        for day in (2, 9):
            for _ in range(4):
                self._event(datetime(2026, 3, day, 6, 0, tzinfo=dt_timezone.utc))
        matrix = weekday_hour_matrix(start=date(2026, 3, 2), end=date(2026, 3, 15))
        monday = matrix[0]
        self.assertEqual(monday["occurrences"], 2)
        cell = monday["hours"][9]
        self.assertEqual(cell["in"], 8)
        self.assertEqual(cell["average_in"], 4.0, "8 arrivals over 2 Mondays")

    def test_a_weekday_absent_from_the_range_averages_zero(self):
        matrix = weekday_hour_matrix(start=date(2026, 3, 2), end=date(2026, 3, 3))
        by_name = {d["name"]: d for d in matrix}
        self.assertEqual(by_name["Friday"]["occurrences"], 0)
        self.assertTrue(all(c["average_in"] == 0.0 for c in by_name["Friday"]["hours"]))

    def test_weekdays_do_not_bleed_into_each_other(self):
        self._event(datetime(2026, 3, 2, 6, 0, tzinfo=dt_timezone.utc))   # Monday
        self._event(datetime(2026, 3, 3, 6, 0, tzinfo=dt_timezone.utc))   # Tuesday
        matrix = weekday_hour_matrix(start=date(2026, 3, 2), end=date(2026, 3, 3))
        self.assertEqual(matrix[0]["hours"][9]["in"], 1)
        self.assertEqual(matrix[1]["hours"][9]["in"], 1)
        self.assertEqual(matrix[2]["hours"][9]["in"], 0)


class HourlyApiTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        for _ in range(6):
            Crossing.objects.create(
                camera=self.camera, track_id=1, direction=DIRECTION_IN,
                occurred_at=datetime(2026, 3, 3, 6, 0, tzinfo=dt_timezone.utc),
            )

    def test_returns_twenty_four_hours_and_peaks(self):
        body = self.client.get("/api/v1/hourly/?start=2026-03-03&end=2026-03-03").json()
        self.assertEqual(len(body["hours"]), 24)
        self.assertEqual(body["busiest"][0]["hour"], 9)
        self.assertEqual(body["busiest"][0]["in"], 6)
        self.assertEqual(body["opening_span"], {"first_hour": 9, "last_hour": 9})

    def test_matrix_is_opt_in(self):
        base = "/api/v1/hourly/?start=2026-03-03&end=2026-03-03"
        self.assertNotIn("matrix", self.client.get(base).json())
        self.assertIn("matrix", self.client.get(base + "&matrix=1").json())

    def test_csv_export(self):
        response = self.client.get(
            "/api/v1/hourly/?start=2026-03-03&end=2026-03-03&format=csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        lines = response.content.decode().strip().splitlines()
        self.assertEqual(lines[0], "event,hour,in,out,average_in,share_percent")
        self.assertEqual(len(lines), 25)
        row = next(csv.reader([lines[10]]))
        self.assertEqual(row[:5], ["All events", "09:00", "6", "0", "6.0"])

    def test_bad_params_are_400(self):
        self.assertEqual(self.client.get("/api/v1/hourly/?days=lots").status_code, 400)
        self.assertEqual(self.client.get("/api/v1/hourly/?camera=9999").status_code, 400)

    def test_dashboard_includes_the_panel(self):
        self.assertContains(self.client.get("/"), "Busiest hours")


class FakeWorker:
    """Stands in for a CounterWorker without threads or a camera."""

    def __init__(self, alive=True, seconds_since_frame=1.0, error=None,
                 frames_read=100, fps=20.0, started_ago=100.0, is_file=False):
        self._alive = alive
        self.is_file = is_file
        self.seconds_since_frame = seconds_since_frame
        self.error = error
        self.frames_read = frames_read
        self.fps = fps
        self.started_at = (time_module.monotonic() - started_ago
                           if started_ago is not None else None)

    def is_alive(self):
        return self._alive


class FakeRegistry:
    def __init__(self, workers=None):
        self._workers = workers or {}

    def get(self, pk):
        return self._workers.get(pk)


@override_settings(COUNTER_STALE_FRAME_SECONDS=30)
class CameraHealthTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0", is_active=True)

    def test_delivering_frames_is_ok(self):
        row = camera_health(self.camera, FakeWorker(seconds_since_frame=2.0))
        self.assertEqual(row["state"], OK)
        self.assertTrue(row["healthy"])

    def test_frozen_stream_is_stalled_not_ok(self):
        """The failure a naive 'is it running?' check misses entirely."""
        row = camera_health(self.camera, FakeWorker(alive=True, seconds_since_frame=300.0))
        self.assertEqual(row["state"], STALLED)
        self.assertFalse(row["healthy"])
        self.assertIn("No frame for 300s", row["detail"])

    def test_just_under_threshold_is_still_ok(self):
        row = camera_health(self.camera, FakeWorker(seconds_since_frame=29.0))
        self.assertEqual(row["state"], OK)

    def test_just_over_threshold_is_stalled(self):
        row = camera_health(self.camera, FakeWorker(seconds_since_frame=31.0))
        self.assertEqual(row["state"], STALLED)

    def test_worker_with_error_is_failed(self):
        row = camera_health(
            self.camera,
            FakeWorker(alive=False, error="Could not open video source 0."),
        )
        self.assertEqual(row["state"], FAILED)
        self.assertIn("Could not open", row["detail"])

    def test_active_camera_with_no_worker_is_offline(self):
        """The post-restart case: meant to be running, silently isn't."""
        row = camera_health(self.camera, None)
        self.assertEqual(row["state"], OFFLINE)
        self.assertFalse(row["healthy"])
        self.assertIn("marked active", row["detail"])

    def test_deactivating_clears_a_stale_failure(self):
        """Turning a camera off must silence it, not alarm forever."""
        self.camera.is_active = False
        row = camera_health(
            self.camera, FakeWorker(alive=False, error="Could not open source."))
        self.assertEqual(row["state"], DISABLED)
        self.assertTrue(row["healthy"])

    def test_finished_video_file_is_not_a_fault(self):
        row = camera_health(
            self.camera,
            FakeWorker(alive=False, is_file=True, frames_read=80),
        )
        self.assertEqual(row["state"], FINISHED)
        self.assertTrue(row["healthy"])
        self.assertIn("80 frames", row["detail"])

    def test_file_that_produced_nothing_is_still_offline(self):
        row = camera_health(
            self.camera,
            FakeWorker(alive=False, is_file=True, frames_read=0),
        )
        self.assertEqual(row["state"], OFFLINE)

    def test_live_camera_that_died_is_offline_not_finished(self):
        row = camera_health(
            self.camera,
            FakeWorker(alive=False, is_file=False, frames_read=5000),
        )
        self.assertEqual(row["state"], OFFLINE)

    def test_inactive_camera_with_no_worker_is_disabled_not_a_fault(self):
        self.camera.is_active = False
        row = camera_health(self.camera, None)
        self.assertEqual(row["state"], DISABLED)
        self.assertTrue(row["healthy"], "a deliberately stopped camera is not a fault")

    def test_starting_up_is_not_immediately_stalled(self):
        row = camera_health(
            self.camera,
            FakeWorker(seconds_since_frame=None, started_ago=3.0),
        )
        self.assertEqual(row["state"], OK)
        self.assertEqual(row["detail"], "Starting up.")

    def test_never_produced_a_frame_eventually_stalls(self):
        row = camera_health(
            self.camera,
            FakeWorker(seconds_since_frame=None, started_ago=120.0),
        )
        self.assertEqual(row["state"], STALLED)
        self.assertIn("without delivering a frame", row["detail"])

    def test_last_event_age_is_reported(self):
        when = timezone.now() - timedelta(minutes=45)
        row = camera_health(self.camera, FakeWorker(), last_event_at=when)
        self.assertAlmostEqual(row["minutes_since_event"], 45.0, delta=0.5)
        self.assertIsNotNone(row["last_event_at"])

    def test_no_events_yet_is_not_an_error(self):
        row = camera_health(self.camera, FakeWorker(), last_event_at=None)
        self.assertIsNone(row["minutes_since_event"])
        self.assertTrue(row["healthy"])


class OverallStatusTests(TestCase):
    def _row(self, state):
        return {"state": state, "healthy": state in (OK, DISABLED)}

    def test_all_ok(self):
        self.assertEqual(overall_status([self._row(OK), self._row(DISABLED)]), STATUS_OK)

    def test_stalled_is_degraded(self):
        self.assertEqual(
            overall_status([self._row(OK), self._row(STALLED)]), STATUS_DEGRADED)

    def test_failed_is_down(self):
        self.assertEqual(
            overall_status([self._row(OK), self._row(FAILED)]), STATUS_DOWN)

    def test_offline_is_down(self):
        self.assertEqual(overall_status([self._row(OFFLINE)]), STATUS_DOWN)

    def test_worst_state_wins(self):
        self.assertEqual(
            overall_status([self._row(STALLED), self._row(OFFLINE)]), STATUS_DOWN)

    def test_no_cameras_is_ok(self):
        self.assertEqual(overall_status([]), STATUS_OK)


@override_settings(COUNTER_STALE_FRAME_SECONDS=30)
class HealthApiTests(TestCase):
    def setUp(self):
        self.good = Camera.objects.create(name="Good", source="0", is_active=True)
        self.bad = Camera.objects.create(name="Bad", source="1", is_active=True)

    def _patch_registry(self, workers):
        return patch("counter.views.REGISTRY", FakeRegistry(workers))

    def test_all_healthy(self):
        workers = {self.good.pk: FakeWorker(), self.bad.pk: FakeWorker()}
        with self._patch_registry(workers):
            body = self.client.get("/api/v1/health/").json()
        self.assertEqual(body["status"], STATUS_OK)
        self.assertTrue(body["healthy"])
        self.assertEqual(body["problems"], [])

    def test_offline_camera_is_reported(self):
        with self._patch_registry({self.good.pk: FakeWorker()}):
            response = self.client.get("/api/v1/health/")
        body = response.json()
        self.assertEqual(response.status_code, 200, "plain check stays 200")
        self.assertEqual(body["status"], STATUS_DOWN)
        self.assertEqual(len(body["problems"]), 1)
        self.assertEqual(body["problems"][0]["camera_name"], "Bad")

    def test_strict_mode_returns_503(self):
        with self._patch_registry({self.good.pk: FakeWorker()}):
            response = self.client.get("/api/v1/health/?strict=1")
        self.assertEqual(response.status_code, 503)

    def test_strict_mode_is_200_when_healthy(self):
        workers = {self.good.pk: FakeWorker(), self.bad.pk: FakeWorker()}
        with self._patch_registry(workers):
            self.assertEqual(
                self.client.get("/api/v1/health/?strict=1").status_code, 200)

    def test_stalled_camera_is_degraded(self):
        workers = {self.good.pk: FakeWorker(),
                   self.bad.pk: FakeWorker(seconds_since_frame=500.0)}
        with self._patch_registry(workers):
            body = self.client.get("/api/v1/health/").json()
        self.assertEqual(body["status"], STATUS_DEGRADED)

    def test_last_event_time_comes_from_the_database(self):
        Crossing.objects.create(camera=self.good, track_id=1,
                                  direction=DIRECTION_IN)
        workers = {self.good.pk: FakeWorker(), self.bad.pk: FakeWorker()}
        with self._patch_registry(workers):
            body = self.client.get("/api/v1/health/").json()
        rows = {r["camera_name"]: r for r in body["cameras"]}
        self.assertIsNotNone(rows["Good"]["last_event_at"])
        self.assertIsNone(rows["Bad"]["last_event_at"])

    def test_health_is_readable_without_auth(self):
        with self._patch_registry({}):
            self.assertEqual(self.client.get("/api/v1/health/").status_code, 200)

    def test_dashboard_has_the_alert_container(self):
        self.assertContains(self.client.get("/"), 'id="health-alert"')


@override_settings(COUNTER_STALE_FRAME_SECONDS=30)
class CheckHealthCommandTests(TestCase):
    def test_exit_zero_when_healthy(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        with patch("counter.management.commands.check_health.REGISTRY",
                   FakeRegistry({camera.pk: FakeWorker()})):
            call_command("check_health", stdout=StringIO())   # no SystemExit

    def test_exit_two_when_a_camera_is_offline(self):
        Camera.objects.create(name="Door", source="0", is_active=True)
        with patch("counter.management.commands.check_health.REGISTRY",
                   FakeRegistry({})):
            with self.assertRaises(SystemExit) as caught:
                call_command("check_health", stdout=StringIO(), stderr=StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_exit_one_when_degraded(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        worker = FakeWorker(seconds_since_frame=900.0)
        with patch("counter.management.commands.check_health.REGISTRY",
                   FakeRegistry({camera.pk: worker})):
            with self.assertRaises(SystemExit) as caught:
                call_command("check_health", stdout=StringIO(), stderr=StringIO())
        self.assertEqual(caught.exception.code, 1)

    def test_no_cameras_is_not_a_failure(self):
        call_command("check_health", stdout=StringIO())


def track_box(cx, cy, w=46, h=92):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class IouBatchTests(TestCase):
    def test_identical_boxes_are_one(self):
        box = [[0, 0, 10, 10]]
        self.assertAlmostEqual(float(iou_batch(box, box)[0][0]), 1.0, places=5)

    def test_disjoint_boxes_are_zero(self):
        self.assertEqual(float(iou_batch([[0, 0, 10, 10]], [[50, 50, 60, 60]])[0][0]), 0.0)

    def test_half_overlap(self):
        # 10x10 boxes sharing exactly half their area.
        value = float(iou_batch([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0][0])
        self.assertAlmostEqual(value, 50 / 150, places=5)

    def test_shape_and_empty_inputs(self):
        self.assertEqual(iou_batch([[0, 0, 1, 1]] * 3, [[0, 0, 1, 1]] * 4).shape, (3, 4))
        self.assertEqual(iou_batch([], [[0, 0, 1, 1]]).shape, (0, 1))
        self.assertEqual(iou_batch([[0, 0, 1, 1]], []).shape, (1, 0))

    def test_degenerate_box_does_not_divide_by_zero(self):
        self.assertEqual(float(iou_batch([[5, 5, 5, 5]], [[0, 0, 10, 10]])[0][0]), 0.0)


class LinearAssignmentTests(TestCase):
    def test_matches_the_obvious_pairing(self):
        cost = np.array([[1.0, 9.0], [9.0, 1.0]])
        self.assertEqual(sorted(linear_assignment(cost)), [(0, 0), (1, 1)])

    def test_greedy_would_lose_here(self):
        # Greedy takes (0,0)=1 and is then forced into (1,1)=100 for 101.
        # The optimum is (0,1)+(1,0) = 2 + 3 = 5.
        cost = np.array([[1.0, 2.0], [3.0, 100.0]])
        pairs = sorted(linear_assignment(cost))
        self.assertEqual(pairs, [(0, 1), (1, 0)])

    def test_rectangular_more_columns(self):
        pairs = linear_assignment(np.array([[1.0, 5.0, 9.0]]))
        self.assertEqual(pairs, [(0, 0)])

    def test_rectangular_more_rows(self):
        pairs = linear_assignment(np.array([[1.0], [5.0], [9.0]]))
        self.assertEqual(pairs, [(0, 0)])

    def test_empty(self):
        self.assertEqual(linear_assignment(np.zeros((0, 0))), [])

    def test_optimal_against_brute_force(self):
        rng = np.random.default_rng(17)
        for _ in range(40):
            rows = int(rng.integers(1, 5))
            cols = int(rng.integers(1, 5))
            cost = rng.integers(0, 15, size=(rows, cols)).astype(float)

            pairs = linear_assignment(cost)
            mine = sum(cost[r, c] for r, c in pairs)

            size = min(rows, cols)
            best = min(
                sum(cost[r, c] for r, c in zip(rs, cs))
                for rs in itertools.permutations(range(rows), size)
                for cs in itertools.permutations(range(cols), size)
            )
            self.assertAlmostEqual(mine, best, places=9)
            self.assertEqual(len(pairs), size)


class KalmanBoxTests(TestCase):
    def test_round_trips_a_box(self):
        box = (100.0, 50.0, 146.0, 142.0)
        kalman = KalmanBox(box)
        for actual, expected in zip(kalman.box, box):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_predicts_constant_velocity_forward(self):
        kalman = KalmanBox(track_box(100, 120))
        for step in range(1, 4):
            kalman.predict()
            kalman.update(track_box(100 + step * 20, 120))

        predicted = kalman.predict()
        centre_x = (predicted[0] + predicted[2]) / 2
        # Having seen +20 per frame, it should lead rather than stand still.
        self.assertGreater(centre_x, 160.0,
                           "motion model should extrapolate, not hold position")

    def test_area_never_inverts_the_box(self):
        kalman = KalmanBox(track_box(100, 120, w=10, h=10))
        for _ in range(60):
            kalman.predict()
        x1, y1, x2, y2 = kalman.box
        self.assertGreaterEqual(x2, x1)
        self.assertGreaterEqual(y2, y1)


class SortTrackerTests(TestCase):
    def _walk(self, tracker, positions, score=0.9):
        """Run the tracker, snapshotting id -> centroid at each frame.

        The tracker hands out live Track objects on purpose, so storing the
        dicts themselves would compare against their final positions later.
        """
        seen = []
        for group in positions:
            dets = [(track_box(x, y), score) for x, y in group]
            tracks = tracker.update_detections(dets)
            seen.append({tid: tuple(t.centroid) for tid, t in tracks.items()})
        return seen

    def test_one_person_keeps_one_id(self):
        tracker = SortTracker(max_disappeared=25)
        frames = self._walk(tracker, [[(60 + i * 25, 120)] for i in range(8)])
        ids = {tid for frame in frames for tid in frame}
        self.assertEqual(len(ids), 1, "a single walker must not fragment")

    def test_tentative_tracks_are_withheld(self):
        tracker = SortTracker(max_disappeared=25, min_hits=2)
        first = tracker.update_detections([(track_box(100, 120), 0.9)])
        self.assertEqual(len(first), 0, "one sighting is not yet a person")
        second = tracker.update_detections([(track_box(120, 120), 0.9)])
        self.assertEqual(len(second), 1)

    def test_crossing_paths_do_not_swap_identity(self):
        """The failure that makes a doorway miscount."""
        tracker = SortTracker(max_disappeared=25)
        path = [[(30 + i * 60, 115), (330 - i * 60, 125)] for i in range(6)]
        frames = self._walk(tracker, path)

        def nearest(snapshot, point):
            return min(snapshot.items(),
                       key=lambda kv: (kv[1][0] - point[0]) ** 2
                       + (kv[1][1] - point[1]) ** 2)[0]

        early = frames[1]
        late = frames[-1]
        a_start = nearest(early, path[1][0])
        b_start = nearest(early, path[1][1])
        a_end = nearest(late, path[-1][0])
        b_end = nearest(late, path[-1][1])

        self.assertEqual(a_start, a_end, "person A changed identity mid-walk")
        self.assertEqual(b_start, b_end, "person B changed identity mid-walk")

    def test_survives_a_brief_occlusion(self):
        tracker = SortTracker(max_disappeared=25)
        self._walk(tracker, [[(60 + i * 25, 120)] for i in range(4)])
        original = set(tracker.tracks)

        for _ in range(3):              # hidden behind a pillar
            tracker.update_detections([])

        after = tracker.update_detections([(track_box(200, 120), 0.9)])
        self.assertEqual(set(after), original,
                         "reappearing after occlusion must not create a new person")

    def test_track_expires_after_max_disappeared(self):
        tracker = SortTracker(max_disappeared=3)
        self._walk(tracker, [[(100, 120)], [(100, 120)]])
        self.assertEqual(len(tracker.tracks), 1)
        for _ in range(5):
            tracker.update_detections([])
        self.assertEqual(len(tracker.tracks), 0)

    def test_two_separate_people_get_two_ids(self):
        tracker = SortTracker(max_disappeared=25)
        frames = self._walk(tracker, [[(60, 120), (260, 120)] for _ in range(3)])
        self.assertEqual(len(frames[-1]), 2)

    def test_advance_moves_tracks_without_detections(self):
        tracker = SortTracker(max_disappeared=25)
        self._walk(tracker, [[(60, 120)], [(90, 120)], [(120, 120)]])
        before = float(next(iter(tracker.tracks.values())).centroid[0])
        tracker.advance()
        after = float(next(iter(tracker.tracks.values())).centroid[0])
        self.assertGreater(after, before, "prediction should carry motion forward")

    def test_track_objects_persist_so_side_is_remembered(self):
        """The line counter writes to track.side; it must survive the frame."""
        tracker = SortTracker(max_disappeared=25)
        self._walk(tracker, [[(60, 120)], [(90, 120)]])
        track = next(iter(tracker.tracks.values()))
        track.side = -1
        tracker.update_detections([(track_box(120, 120), 0.9)])
        self.assertEqual(next(iter(tracker.tracks.values())).side, -1)

    def test_weak_detection_recovers_a_track_in_the_second_stage(self):
        tracker = SortTracker(max_disappeared=25, high_confidence=0.5)
        self._walk(tracker, [[(60, 120)], [(90, 120)]], score=0.9)
        original = set(tracker.tracks)
        # Partly occluded: still there, but the detector is unsure.
        tracker.update_detections([(track_box(120, 120), 0.2)])
        self.assertEqual(set(tracker.tracks), original)

    def test_distance_gate_rejects_an_unrelated_box(self):
        tracker = SortTracker(max_disappeared=25)
        score = tracker.affinity([track_box(50, 50)], [track_box(2000, 2000)])
        self.assertEqual(float(score[0][0]), -1.0)

    def test_affinity_rewards_overlap_and_closeness(self):
        tracker = SortTracker()
        close = float(tracker.affinity([track_box(100, 120)], [track_box(110, 120)])[0][0])
        far = float(tracker.affinity([track_box(100, 120)], [track_box(190, 120)])[0][0])
        self.assertGreater(close, far)

    def test_fast_motion_still_matches(self):
        """Displacement near box width kills plain IoU; this must survive it."""
        tracker = SortTracker(max_disappeared=25)
        frames = self._walk(tracker, [[(60 + i * 45, 120)] for i in range(6)])
        ids = {tid for frame in frames for tid in frame}
        self.assertEqual(len(ids), 1)


class TrackerComparisonTests(TestCase):
    """Why this tracker exists, stated as a test."""

    FRAME_H, FRAME_W = 240, 320
    A = [30, 90, 150, 210, 270, 300]
    B = [290, 230, 170, 110, 50, 20]

    class _Script:
        name = "stub"
        score_is_probability = True
        min_confidence = 0.35

        def __init__(self, a_path, b_path):
            self.a_path, self.b_path, self.index = a_path, b_path, 0

        def detect(self, frame):
            k = min(self.index, len(self.a_path) - 1)
            self.index += 1
            return [(track_box(self.a_path[k], 115), 0.9),
                    (track_box(self.b_path[k], 125), 0.9)]

    def _count(self, tracker):
        counter = VisitorCounter(
            detector=self._Script(self.A, self.B),
            line=LineCounter(position=0.5, orientation=VERTICAL),
            detect_every=1, frame_width=self.FRAME_W, tracker=tracker,
        )
        for _ in range(len(self.A)):
            counter.process(np.zeros((self.FRAME_H, self.FRAME_W, 3), dtype=np.uint8))
        return counter.total_in, counter.total_out

    def test_sort_counts_a_two_way_pass_correctly(self):
        """One entering, one leaving, passing at the line: 1 in, 1 out."""
        self.assertEqual(self._count(SortTracker(max_disappeared=25)), (1, 1))

    def test_centroid_tracker_miscounts_the_same_scene(self):
        """Documents the defect being fixed: the swap cancels both crossings."""
        self.assertEqual(self._count(CentroidTracker(max_disappeared=25)), (0, 0))


class BuildTrackerTests(TestCase):
    def test_sort_is_the_default(self):
        self.assertIsInstance(build_tracker(None), SortTracker)
        self.assertIsInstance(build_tracker("sort"), SortTracker)

    def test_centroid_can_be_forced(self):
        self.assertIsInstance(build_tracker("centroid"), CentroidTracker)

    def test_two_stage_only_for_probability_detectors(self):
        class Prob:
            score_is_probability = True
            min_confidence = 0.35

        class Svm:
            score_is_probability = False
            min_confidence = 0.45

        self.assertIsNotNone(build_tracker("sort", detector=Prob()).high_confidence)
        self.assertIsNone(build_tracker("sort", detector=Svm()).high_confidence)
        self.assertIsNone(build_tracker("sort", detector=None).high_confidence)


def make_event(name="Open day", offset_hours=-1, duration_hours=4, **kwargs):
    start = timezone.now() + timedelta(hours=offset_hours)
    return Event.objects.create(
        name=name,
        starts_at=start,
        ends_at=start + timedelta(hours=duration_hours),
        **kwargs,
    )


class EventModelTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.other = Camera.objects.create(name="Side", source="1")

    def test_running_while_between_its_times(self):
        event = make_event(offset_hours=-1, duration_hours=4)
        self.assertTrue(event.is_running())
        self.assertFalse(event.has_finished())

    def test_not_running_before_it_starts(self):
        event = make_event(offset_hours=2)
        self.assertFalse(event.is_running())

    def test_not_running_after_it_ends(self):
        event = make_event(offset_hours=-10, duration_hours=2)
        self.assertFalse(event.is_running())
        self.assertTrue(event.has_finished())

    def test_deactivating_stops_it_running(self):
        event = make_event(is_active=False)
        self.assertFalse(event.is_running())

    def test_end_must_follow_start(self):
        event = Event(
            name="Backwards",
            starts_at=timezone.now(),
            ends_at=timezone.now() - timedelta(hours=1),
        )
        with self.assertRaises(ValidationError):
            event.clean()

    def test_no_cameras_means_every_camera(self):
        event = make_event()
        self.assertTrue(event.covers_camera(self.camera))
        self.assertTrue(event.covers_camera(self.other))
        self.assertTrue(event.totals() is not None)

    def test_listing_cameras_restricts_it(self):
        event = make_event()
        event.cameras.add(self.camera)
        self.assertTrue(event.covers_camera(self.camera))
        self.assertFalse(event.covers_camera(self.other))

    def test_totals_count_only_its_own_crossings(self):
        event = make_event()
        other = make_event("Other")
        for _ in range(4):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN, event=event)
        Crossing.objects.create(camera=self.camera, track_id=2,
                                direction=DIRECTION_OUT, event=event)
        Crossing.objects.create(camera=self.camera, track_id=3,
                                direction=DIRECTION_IN, event=other)
        Crossing.objects.create(camera=self.camera, track_id=4,
                                direction=DIRECTION_IN)      # unattributed

        self.assertEqual(event.totals(), {"in": 4, "out": 1, "inside": 3, "net": 3})

    def test_capacity_state_levels(self):
        event = make_event(capacity=10)
        for _ in range(5):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN, event=event)
        self.assertEqual(event.capacity_state()["level"], "ok")
        self.assertEqual(event.capacity_state()["remaining"], 5)

        for _ in range(4):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN, event=event)
        self.assertEqual(event.capacity_state()["level"], "near")

        for _ in range(3):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN, event=event)
        state = event.capacity_state()
        self.assertEqual(state["level"], "over")
        self.assertEqual(state["remaining"], 0)

    def test_no_capacity_means_no_state(self):
        self.assertIsNone(make_event().capacity_state())

    def test_deleting_an_event_keeps_its_crossings(self):
        event = make_event()
        Crossing.objects.create(camera=self.camera, track_id=1,
                                direction=DIRECTION_IN, event=event)
        event.delete()
        self.assertEqual(Crossing.objects.count(), 1,
                         "history must survive removing the event")
        self.assertIsNone(Crossing.objects.first().event_id)


class ActiveEventForTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.other = Camera.objects.create(name="Side", source="1")

    def test_none_when_nothing_scheduled(self):
        self.assertIsNone(active_event_for(self.camera))

    def test_finds_the_running_event(self):
        event = make_event()
        self.assertEqual(active_event_for(self.camera), event)

    def test_ignores_events_for_other_cameras(self):
        event = make_event()
        event.cameras.add(self.other)
        self.assertIsNone(active_event_for(self.camera))

    def test_ignores_inactive_and_finished_events(self):
        make_event("Off", is_active=False)
        make_event("Past", offset_hours=-10, duration_hours=2)
        make_event("Future", offset_hours=5)
        self.assertIsNone(active_event_for(self.camera))

    def test_most_recently_started_wins_when_nested(self):
        make_event("All-day programme", offset_hours=-5, duration_hours=10)
        session = make_event("Keynote", offset_hours=-1, duration_hours=2)
        self.assertEqual(active_event_for(self.camera), session,
                         "the inner session is the more specific answer")

    def test_camera_specific_event_is_found_among_others(self):
        make_event("Elsewhere", offset_hours=-2).cameras.add(self.other)
        mine = make_event("Mine", offset_hours=-1)
        self.assertEqual(active_event_for(self.camera), mine)


class EventOccupancyTests(TestCase):
    """Occupancy during an event counts from when the event began."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_people_inside_before_the_event_are_not_counted(self):
        now = timezone.now()
        event = Event.objects.create(
            name="Evening show",
            starts_at=now - timedelta(hours=1),
            ends_at=now + timedelta(hours=2),
        )
        # Earlier the same day, before the event started.
        for _ in range(12):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN,
                                    occurred_at=now - timedelta(hours=3))
        # During the event.
        for _ in range(3):
            Crossing.objects.create(camera=self.camera, track_id=2,
                                    direction=DIRECTION_IN,
                                    occurred_at=now - timedelta(minutes=10),
                                    event=event)

        occupancy = self.camera.occupancy()
        self.assertEqual(occupancy["inside"], 3)
        self.assertEqual(occupancy["event_name"], "Evening show")

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_falls_back_to_the_daily_reset_with_no_event(self):
        Crossing.objects.create(camera=self.camera, track_id=1,
                                direction=DIRECTION_IN)
        occupancy = self.camera.occupancy()
        self.assertEqual(occupancy["inside"], 1)
        self.assertIsNone(occupancy["event_id"])

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="00:00")
    def test_multi_day_event_still_resets_daily(self):
        now = timezone.now()
        event = Event.objects.create(
            name="Three-day expo",
            starts_at=now - timedelta(days=2),
            ends_at=now + timedelta(days=1),
        )
        # Yesterday's visitors, who went home.
        for _ in range(40):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN, event=event,
                                    occurred_at=now - timedelta(days=1))
        Crossing.objects.create(camera=self.camera, track_id=2,
                                direction=DIRECTION_IN, event=event)

        self.assertEqual(self.camera.occupancy()["inside"], 1,
                         "the later of event start and daily reset wins")
        self.assertEqual(event.totals()["in"], 41,
                         "event attendance still spans the whole run")


class EventApiTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.event = make_event("Open day", capacity=100)
        for _ in range(7):
            Crossing.objects.create(camera=self.camera, track_id=1,
                                    direction=DIRECTION_IN, event=self.event)

    def test_list(self):
        body = self.client.get("/api/v1/events/").json()
        self.assertEqual(body["count"], 1)
        row = body["events"][0]
        self.assertEqual(row["name"], "Open day")
        self.assertTrue(row["running"])
        self.assertTrue(row["covers_all_cameras"])
        self.assertEqual(row["attendance"]["in"], 7)
        self.assertEqual(row["capacity"]["capacity"], 100)

    def test_running_filter(self):
        make_event("Later", offset_hours=10)
        self.assertEqual(self.client.get("/api/v1/events/").json()["count"], 2)
        self.assertEqual(
            self.client.get("/api/v1/events/?running=1").json()["count"], 1)

    def test_camera_filter(self):
        other = Camera.objects.create(name="Side", source="1")
        self.event.cameras.add(self.camera)
        self.assertEqual(
            self.client.get(f"/api/v1/events/?camera={self.camera.pk}").json()["count"], 1)
        self.assertEqual(
            self.client.get(f"/api/v1/events/?camera={other.pk}").json()["count"], 0)

    def test_unknown_camera_is_400(self):
        self.assertEqual(self.client.get("/api/v1/events/?camera=999").status_code, 400)

    def test_detail_includes_an_hourly_profile(self):
        body = self.client.get(f"/api/v1/events/{self.event.pk}/").json()
        self.assertEqual(body["name"], "Open day")
        self.assertEqual(len(body["hours"]), 24)
        self.assertEqual(body["busiest"][0]["in"], 7)

    def test_detail_404(self):
        self.assertEqual(self.client.get("/api/v1/events/999/").status_code, 404)

    def test_current_event(self):
        body = self.client.get("/api/v1/events/current/").json()
        self.assertTrue(body["running"])
        self.assertEqual(body["events"][0]["name"], "Open day")
        self.assertGreater(body["events"][0]["seconds_remaining"], 0)

    def test_current_is_empty_when_nothing_runs(self):
        self.event.delete()
        body = self.client.get("/api/v1/events/current/").json()
        self.assertFalse(body["running"])
        self.assertEqual(body["events"], [])

    def test_reports_filter_by_event(self):
        other = make_event("Other")
        Crossing.objects.create(camera=self.camera, track_id=9,
                                direction=DIRECTION_IN, event=other)

        everything = self.client.get("/api/v1/daily/?days=1").json()
        mine = self.client.get(f"/api/v1/daily/?days=1&event={self.event.pk}").json()
        self.assertEqual(everything["summary"]["in"], 8)
        self.assertEqual(mine["summary"]["in"], 7)
        self.assertEqual(mine["event_name"], "Open day")

    def test_hourly_filters_by_event(self):
        body = self.client.get(
            f"/api/v1/hourly/?days=1&event={self.event.pk}").json()
        self.assertEqual(sum(h["in"] for h in body["hours"]), 7)
        self.assertEqual(body["event_id"], self.event.pk)

    def test_unknown_event_in_report_is_400(self):
        self.assertEqual(self.client.get("/api/v1/daily/?event=999").status_code, 400)

    def test_dashboard_has_the_event_host(self):
        self.assertContains(self.client.get("/"), 'id="event-now"')


class OccupancyScopeLabelTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="06:00")
    def test_label_names_the_daily_reset_with_no_event(self):
        body = self.client.get("/api/v1/stats/").json()
        self.assertEqual(body["occupancy"]["scope"], "since 06:00")

    def test_label_names_the_running_event(self):
        make_event("Science Fair")
        body = self.client.get("/api/v1/stats/").json()
        self.assertEqual(body["occupancy"]["scope"], "since Science Fair began")


# ---------------------------------------------------------------------------
# surviving a dropped stream
# ---------------------------------------------------------------------------

class FakeCapture:
    """Stands in for cv2.VideoCapture: n good frames, then nothing."""

    def __init__(self, frames=0, opened=True):
        self.frames = frames
        self._opened = opened
        self.served = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self.served < self.frames:
            self.served += 1
            return True, blank_frame()
        return False, None

    def get(self, prop):
        return 0.0

    def release(self):
        self.released = True


class CaptureScript:
    """Hands out captures in order, then unopenable ones for ever."""

    def __init__(self, captures):
        self.captures = list(captures)
        self.opens = 0

    def __call__(self, source):
        self.opens += 1
        if self.captures:
            return self.captures.pop(0)
        return FakeCapture(opened=False)


class ForgetTracksTests(TestCase):
    """Dropping tracks over a stream gap, while keeping the id counter."""

    def test_centroid_tracker_forgets_tracks_but_not_ids(self):
        tracker = CentroidTracker()
        tracker.update([box_at(100, 100)])
        first_id = next(iter(tracker.tracks))

        tracker.forget_tracks()
        self.assertEqual(len(tracker.tracks), 0)

        tracker.update([box_at(100, 100)])
        self.assertNotIn(first_id, tracker.tracks,
                         "a reused id would read as the same person")

    def test_sort_tracker_forgets_tracks_but_not_ids(self):
        tracker = SortTracker(min_hits=1)
        tracker.update([(track_box(100, 100), 0.9)])
        first_id = next(iter(tracker.tracks))

        tracker.forget_tracks()
        self.assertEqual(len(tracker.tracks), 0)

        tracker.update([(track_box(100, 100), 0.9)])
        self.assertNotIn(first_id, tracker.tracks)

    def test_counter_keeps_its_totals(self):
        counter = VisitorCounter(detector=StubDetector([[]]),
                                 line=LineCounter(position=0.5))
        counter.line.total_in = 7
        counter.line.total_out = 3

        counter.forget_tracks()

        self.assertEqual((counter.total_in, counter.total_out), (7, 3),
                         "a reconnect continues the session, it does not reset it")
        self.assertEqual(counter.active_tracks, 0)


class ReconnectDoesNotInventCrossingsTests(TestCase):
    """The hazard in reconnecting: a stale track meeting a new person.

    Both halves run the same walk past the line. The only difference is
    the gap in the middle, which is what must stop the crossing being
    counted -- the person on the far side is not the one who was waiting.
    """

    WALK = [40, 45, 50, 55, 60, 180, 185, 190, 195, 200]

    def _walk(self, gap_after=None):
        counter = VisitorCounter(
            detector=StubDetector([[box_at(160, y)] for y in self.WALK]),
            tracker=build_tracker("sort", max_disappeared=25),
            line=LineCounter(position=0.5),
            detect_every=1,
        )
        frame = blank_frame()
        events = []
        for index in range(len(self.WALK)):
            if gap_after is not None and index == gap_after:
                counter.forget_tracks()
            _annotated, batch = counter.process(frame)
            events.extend(batch)
        return counter, events

    def test_an_uninterrupted_walk_is_counted(self):
        counter, events = self._walk()
        self.assertEqual(counter.total_in, 1)
        self.assertEqual(len(events), 1,
                         "control case: this walk must be countable")

    def test_the_same_walk_across_a_gap_is_not_counted(self):
        counter, events = self._walk(gap_after=5)
        self.assertEqual(events, [])
        self.assertEqual((counter.total_in, counter.total_out), (0, 0))


@override_settings(COUNTER_RECONNECT_ATTEMPTS=1)
class WorkerReconnectTests(TransactionTestCase):
    """The real worker thread, over a capture that drops out."""

    def _run(self, script, timeout=20):
        camera = Camera.objects.create(name="Gate", source="0")
        worker = CounterWorker(camera)
        worker.counter.detector = StubDetector(
            [[box_at(160, y)] for y in ReconnectDoesNotInventCrossingsTests.WALK])
        worker.counter.detect_every = 1

        with patch("cv2.VideoCapture", script), \
                patch("counter.camera.MAX_READ_FAILURES", 2), \
                patch("counter.camera.RECONNECT_BACKOFF_START", 0.05), \
                patch("counter.camera.RECONNECT_BACKOFF_MAX", 0.05):
            worker.start()
            worker.join(timeout=timeout)
        self.assertFalse(worker.is_alive(), "worker never finished")
        return camera, worker

    def test_a_dropped_stream_is_reopened(self):
        script = CaptureScript([FakeCapture(frames=5), FakeCapture(frames=5)])
        _camera, worker = self._run(script)

        self.assertEqual(worker.reconnects, 1)
        self.assertEqual(worker.frames_read, 10,
                         "frames from both connections count")

    def test_reconnecting_does_not_fabricate_a_crossing(self):
        # Five frames above the line, the stream drops, five frames below.
        script = CaptureScript([FakeCapture(frames=5), FakeCapture(frames=5)])
        camera, worker = self._run(script)

        self.assertEqual(worker.reconnects, 1)
        self.assertEqual(Crossing.objects.filter(camera=camera).count(), 0)
        self.assertEqual(worker.counter.total_in, 0)

    def test_the_same_frames_without_a_drop_do_count(self):
        script = CaptureScript([FakeCapture(frames=10)])
        camera, worker = self._run(script)

        self.assertEqual(worker.reconnects, 0)
        self.assertEqual(Crossing.objects.filter(camera=camera).count(), 1,
                         "control case: the walk itself is countable")

    def test_attempts_are_capped_by_the_setting(self):
        script = CaptureScript([FakeCapture(frames=2)])
        _camera, worker = self._run(script)

        # One connection, one retry allowed, then the next failure stops it.
        self.assertEqual(script.opens, 2)
        self.assertIsNotNone(worker.error)

    @override_settings(COUNTER_RECONNECT_ATTEMPTS=0)
    def test_zero_attempts_means_keep_trying(self):
        script = CaptureScript([FakeCapture(frames=2)])
        camera = Camera.objects.create(name="Gate", source="0")
        worker = CounterWorker(camera)

        with patch("cv2.VideoCapture", script), \
                patch("counter.camera.MAX_READ_FAILURES", 1), \
                patch("counter.camera.RECONNECT_BACKOFF_START", 0.01), \
                patch("counter.camera.RECONNECT_BACKOFF_MAX", 0.01):
            worker.start()
            deadline = time_module.monotonic() + 5
            while script.opens < 4 and time_module.monotonic() < deadline:
                time_module.sleep(0.02)
            still_trying = worker.is_alive()
            worker.stop()

        self.assertTrue(still_trying, "a camera that may come back is not given up on")
        self.assertGreaterEqual(script.opens, 4)

    def test_a_file_that_ends_is_not_reconnected_to(self):
        camera = Camera.objects.create(name="Clip", source="/tmp/some-clip.mp4")
        worker = CounterWorker(camera)
        script = CaptureScript([FakeCapture(frames=3)])

        with patch("cv2.VideoCapture", script):
            worker.start()
            worker.join(timeout=10)

        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.reconnects, 0, "a clip ending is not a fault")
        self.assertEqual(script.opens, 1)
        self.assertIsNone(worker.error)


@override_settings(COUNTER_STALE_FRAME_SECONDS=30)
class ReconnectingHealthTests(TestCase):
    """A camera being reconnected to is degraded, not down, and says so."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0", is_active=True)

    def _worker(self, reconnects=0, **kwargs):
        worker = FakeWorker(**kwargs)
        worker.reconnecting = True
        worker.reconnects = reconnects
        worker.retry_in = 8.0
        return worker

    def test_reconnecting_beats_stalled(self):
        worker = self._worker(seconds_since_frame=120.0,
                              error="Camera stopped returning frames.")
        row = camera_health(self.camera, worker)

        self.assertEqual(row["state"], RECONNECTING,
                         "'stalled' hides the fact that something is being done")
        self.assertIn("Retrying in 8s", row["detail"])
        self.assertFalse(row["healthy"])

    def test_reconnecting_is_degraded_not_down(self):
        worker = self._worker(seconds_since_frame=120.0, error="Stream lost.")
        rows = [camera_health(self.camera, worker)]
        self.assertEqual(overall_status(rows), STATUS_DEGRADED)

    def test_a_recovered_camera_is_ok_again(self):
        worker = FakeWorker(seconds_since_frame=1.0)
        worker.reconnecting = False
        worker.reconnects = 3
        row = camera_health(self.camera, worker)

        self.assertEqual(row["state"], OK)
        self.assertEqual(row["reconnects"], 3)


# ---------------------------------------------------------------------------
# keeping cameras running
# ---------------------------------------------------------------------------

class RecordingRegistry:
    """A registry that records what it was asked to do."""

    def __init__(self, workers=None):
        self.workers = dict(workers or {})
        self.started = []
        self.stopped = []

    def get(self, pk):
        return self.workers.get(int(pk))

    def start(self, camera):
        self.started.append(camera.pk)
        self.workers[camera.pk] = FakeWorker()
        return self.workers[camera.pk]

    def stop(self, pk):
        self.stopped.append(int(pk))
        return self.workers.pop(int(pk), None) is not None

    def stats(self):
        return [{"camera_id": pk} for pk in self.workers]


class SupervisorReconcileTests(TestCase):
    """What should be running is what the database says, not what was clicked."""

    def setUp(self):
        self.registry = RecordingRegistry()
        self.supervisor = Supervisor(registry=self.registry, interval=1, cooldown=30)

    def test_an_active_camera_with_no_worker_is_started(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        result = self.supervisor.reconcile()

        self.assertEqual(result.started, [camera.pk])
        self.assertEqual((result.stopped, result.waiting), ([], []))

    def test_a_running_camera_is_left_alone(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        self.registry.workers[camera.pk] = FakeWorker(alive=True)

        result = self.supervisor.reconcile()

        self.assertEqual((result.started, result.stopped), ([], []),
                         "reconciling must not restart a healthy camera")

    def test_a_dead_worker_is_replaced(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        self.registry.workers[camera.pk] = FakeWorker(alive=False)

        result = self.supervisor.reconcile()

        self.assertEqual(result.started, [camera.pk],
                         "this is the post-restart case the report warns about")

    def test_a_deactivated_camera_is_stopped(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=False)
        self.registry.workers[camera.pk] = FakeWorker(alive=True)

        result = self.supervisor.reconcile()

        self.assertEqual(result.stopped, [camera.pk])
        self.assertEqual(result.started, [])

    def test_a_deleted_camera_is_stopped(self):
        self.registry.workers[404] = FakeWorker(alive=True)

        result = self.supervisor.reconcile()

        self.assertEqual(result.stopped, [404])

    def test_a_flapping_camera_waits_out_its_cooldown(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)

        first = self.supervisor.reconcile(now=1000.0)
        self.assertEqual(first.started, [camera.pk])

        # It died again immediately.
        self.registry.workers[camera.pk] = FakeWorker(alive=False)
        result = self.supervisor.reconcile(now=1005.0)

        self.assertEqual(result.started, [],
                         "restarting in a tight loop helps nobody")
        self.assertEqual(result.waiting, [camera.pk])

    def test_it_restarts_once_the_cooldown_passes(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        self.supervisor.reconcile(now=1000.0)
        self.registry.workers[camera.pk] = FakeWorker(alive=False)

        result = self.supervisor.reconcile(now=1031.0)

        self.assertEqual(result.started, [camera.pk])
        self.assertEqual(result.waiting, [])

    def test_the_first_start_is_immediate(self):
        Camera.objects.create(name="Door", source="0", is_active=True)
        result = self.supervisor.reconcile(now=0.0)
        self.assertEqual(len(result.started), 1)
        self.assertEqual(result.waiting, [])


class AutostartGuardTests(TestCase):
    """Importing the app must not open a camera unless that was asked for."""

    def _argv(self, *args):
        return patch.object(sys, "argv", ["manage.py", *args])

    @override_settings(COUNTER_AUTOSTART=False)
    def test_off_by_default(self):
        with self._argv("runserver", "--noreload"):
            self.assertFalse(autostart_enabled())

    @override_settings(COUNTER_AUTOSTART=True)
    def test_on_when_asked_for(self):
        with self._argv("runserver", "--noreload"):
            self.assertTrue(autostart_enabled())

    @override_settings(COUNTER_AUTOSTART=True)
    def test_never_during_a_management_command(self):
        for command in ("migrate", "test", "collectstatic", "run_counter",
                        "supervise", "shell"):
            with self.subTest(command=command), self._argv(command):
                self.assertFalse(autostart_enabled(),
                                 f"{command} must not grab a camera")

    @override_settings(COUNTER_AUTOSTART=True)
    def test_every_blocked_command_is_blocked(self):
        for command in NO_AUTOSTART_COMMANDS:
            with self.subTest(command=command), self._argv(command):
                self.assertFalse(autostart_enabled())

    @override_settings(COUNTER_AUTOSTART=True)
    def test_the_reloaders_parent_process_does_not_count(self):
        # runserver without --noreload runs the app twice. Only the child,
        # which carries RUN_MAIN, should open cameras.
        with self._argv("runserver"), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RUN_MAIN", None)
            self.assertFalse(autostart_enabled())

        with self._argv("runserver"), patch.dict(os.environ, {"RUN_MAIN": "true"}):
            self.assertTrue(autostart_enabled())

    @override_settings(COUNTER_AUTOSTART=True)
    def test_a_wsgi_server_counts(self):
        with patch.object(sys, "argv", ["gunicorn", "config.wsgi"]):
            self.assertTrue(autostart_enabled())


class SuperviseCommandTests(TestCase):
    def test_once_does_a_single_pass(self):
        camera = Camera.objects.create(name="Door", source="0", is_active=True)
        out = StringIO()

        registry = RecordingRegistry()
        with patch("counter.management.commands.supervise.REGISTRY", registry):
            call_command("supervise", "--once", stdout=out)

        self.assertIn(f"started camera {camera.pk}", out.getvalue())
        self.assertEqual(registry.started, [camera.pk])


# ---------------------------------------------------------------------------
# the fullest moment, not just the busiest day
# ---------------------------------------------------------------------------

def local_at(year, month, day, hour=0, minute=0):
    return timezone.make_aware(
        datetime(year, month, day, hour, minute), timezone.get_current_timezone())


class PeakOccupancyTests(TestCase):
    """Occupancy's high-water mark, which "inside now" cannot answer later."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.other = Camera.objects.create(name="Side", source="1")

    def _cross(self, when, direction=DIRECTION_IN, camera=None):
        return Crossing.objects.create(
            camera=camera or self.camera, track_id=1,
            direction=direction, occurred_at=when,
        )

    def test_peak_is_the_highest_simultaneous_count(self):
        day = date(2026, 5, 4)
        for minute, direction in [(0, DIRECTION_IN), (1, DIRECTION_IN),
                                  (2, DIRECTION_IN), (3, DIRECTION_OUT),
                                  (4, DIRECTION_OUT), (5, DIRECTION_IN)]:
            self._cross(local_at(2026, 5, 4, 10, minute), direction)

        result = peak_occupancy(start=day, end=day)

        self.assertEqual(result["peak"], 3, "three were inside at 10:02")
        self.assertEqual(result["date"], "2026-05-04")

    def test_peak_records_when_it_happened(self):
        day = date(2026, 5, 4)
        self._cross(local_at(2026, 5, 4, 9, 0))
        self._cross(local_at(2026, 5, 4, 11, 40))
        self._cross(local_at(2026, 5, 4, 12, 0), DIRECTION_OUT)

        result = peak_occupancy(start=day, end=day)

        self.assertEqual(result["peak"], 2)
        self.assertEqual(timezone.localtime(
            datetime.fromisoformat(result["at"])).strftime("%H:%M"), "11:40")

    def test_footfall_and_fullness_are_different_questions(self):
        # 400 people through the door, never more than two inside at once.
        day = date(2026, 5, 4)
        when = local_at(2026, 5, 4, 9, 0)
        for index in range(200):
            self._cross(when + timedelta(seconds=index * 10), DIRECTION_IN)
            self._cross(when + timedelta(seconds=index * 10 + 1), DIRECTION_IN)
            self._cross(when + timedelta(seconds=index * 10 + 2), DIRECTION_OUT)
            self._cross(when + timedelta(seconds=index * 10 + 3), DIRECTION_OUT)

        rows = daily_counts(start=day, end=day)

        self.assertEqual(rows[0]["in"], 400)
        self.assertEqual(rows[0]["peak"], 2,
                         "a busy doorway is not a full room")

    def test_a_missed_exit_cannot_inflate_later_days(self):
        # Ten in and no exits on the first day: the next day must start empty.
        for minute in range(10):
            self._cross(local_at(2026, 5, 4, 10, minute))
        for minute in range(2):
            self._cross(local_at(2026, 5, 5, 10, minute))

        peaks = occupancy_peaks(start=date(2026, 5, 4), end=date(2026, 5, 5))

        self.assertEqual(peaks[date(2026, 5, 4)]["peak"], 10)
        self.assertEqual(peaks[date(2026, 5, 5)]["peak"], 2,
                         "yesterday's ghosts must not be inside today")

    def test_the_running_count_is_clamped_at_zero(self):
        # Exits with no matching entries: a miscount, not negative occupancy.
        day = date(2026, 5, 4)
        self._cross(local_at(2026, 5, 4, 9, 0), DIRECTION_OUT)
        self._cross(local_at(2026, 5, 4, 9, 1), DIRECTION_OUT)
        self._cross(local_at(2026, 5, 4, 9, 2), DIRECTION_IN)

        result = peak_occupancy(start=day, end=day)

        self.assertEqual(result["peak"], 1,
                         "a negative baseline would hide the one real arrival")

    def test_late_night_utc_lands_on_the_local_date(self):
        # 22:30 UTC on the 1st is 01:30 on the 2nd in Kuwait.
        self._cross(datetime(2026, 3, 1, 22, 30, tzinfo=dt_timezone.utc))
        peaks = occupancy_peaks(start=date(2026, 3, 1), end=date(2026, 3, 2))

        self.assertNotIn(date(2026, 3, 1), peaks)
        self.assertEqual(peaks[date(2026, 3, 2)]["peak"], 1)

    def test_all_cameras_are_walked_together(self):
        # One person in each door at the same time is two people inside,
        # and each camera on its own would say one.
        day = date(2026, 5, 4)
        self._cross(local_at(2026, 5, 4, 10, 0), camera=self.camera)
        self._cross(local_at(2026, 5, 4, 10, 1), camera=self.other)

        self.assertEqual(peak_occupancy(start=day, end=day)["peak"], 2)
        self.assertEqual(
            peak_occupancy(start=day, end=day, camera=self.camera)["peak"], 1)

    def test_nothing_recorded_is_a_zero_peak(self):
        result = peak_occupancy(start=date(2026, 5, 4), end=date(2026, 5, 4))
        self.assertEqual(result, {"peak": 0, "at": None, "date": None})

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="06:00")
    def test_a_day_that_began_before_midnight_is_not_restarted(self):
        # With a 06:00 reset, 02:00 belongs to the window that opened at
        # 06:00 the day before, and those three are still inside.
        for minute in range(3):
            self._cross(local_at(2026, 5, 4, 20, minute))
        self._cross(local_at(2026, 5, 5, 2, 0))
        self._cross(local_at(2026, 5, 5, 2, 1))

        peaks = occupancy_peaks(start=date(2026, 5, 5), end=date(2026, 5, 5))

        self.assertEqual(peaks[date(2026, 5, 5)]["peak"], 5,
                         "the five inside at 02:01, not just the two arrivals")

    @override_settings(COUNTER_OCCUPANCY_RESET_AT="06:00")
    def test_the_reset_still_empties_the_count(self):
        for minute in range(3):
            self._cross(local_at(2026, 5, 4, 20, minute))
        self._cross(local_at(2026, 5, 5, 8, 0))

        peaks = occupancy_peaks(start=date(2026, 5, 5), end=date(2026, 5, 5))

        self.assertEqual(peaks[date(2026, 5, 5)]["peak"], 1,
                         "08:00 is after the 06:00 reset, so it starts empty")


class DailyPeakReportTests(TestCase):
    """The peak travels with the per-date report and its summary."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _cross(self, when, direction=DIRECTION_IN):
        Crossing.objects.create(camera=self.camera, track_id=1,
                                direction=direction, occurred_at=when)

    def test_rows_carry_the_peak_and_its_time(self):
        self._cross(local_at(2026, 5, 4, 9, 0))
        self._cross(local_at(2026, 5, 4, 9, 30))

        rows = daily_counts(start=date(2026, 5, 4), end=date(2026, 5, 4))

        self.assertEqual(rows[0]["peak"], 2)
        self.assertTrue(rows[0]["peak_at"].startswith("2026-05-04"))

    def test_a_quiet_day_reads_zero_rather_than_blank(self):
        rows = daily_counts(start=date(2026, 5, 4), end=date(2026, 5, 5))
        self.assertEqual([r["peak"] for r in rows], [0, 0])
        self.assertEqual([r["peak_at"] for r in rows], [None, None])

    def test_the_summary_reports_the_fullest_day(self):
        # Busiest by footfall and fullest are different days on purpose.
        for minute in range(6):
            self._cross(local_at(2026, 5, 4, 9, minute))
            self._cross(local_at(2026, 5, 4, 9, minute, ) + timedelta(seconds=30),
                        DIRECTION_OUT)
        for minute in range(4):
            self._cross(local_at(2026, 5, 5, 9, minute))

        rows = daily_counts(start=date(2026, 5, 4), end=date(2026, 5, 5))
        summary = summarise(rows)

        self.assertEqual(summary["busiest_date"], "2026-05-04")
        self.assertEqual(summary["busiest_in"], 6)
        self.assertEqual(summary["peak_inside"], 4)
        self.assertEqual(summary["peak_date"], "2026-05-05")

    def test_an_empty_summary_has_a_zero_peak(self):
        summary = summarise([])
        self.assertEqual(summary["peak_inside"], 0)
        self.assertIsNone(summary["peak_date"])


class EventPeakTests(TestCase):
    """An event's fullest moment, and capacity measured against it."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _cross(self, when, direction=DIRECTION_IN, event=None):
        Crossing.objects.create(camera=self.camera, track_id=1,
                                direction=direction, occurred_at=when,
                                event=event)

    def test_peak_is_the_worst_day_not_the_sum_of_days(self):
        event = Event.objects.create(
            name="Three-day expo",
            starts_at=timezone.now() - timedelta(days=3),
            ends_at=timezone.now() + timedelta(hours=1),
        )
        # Ten inside at once on each of three days, all of whom leave.
        for days_back in (3, 2, 1):
            base = timezone.localtime() - timedelta(days=days_back)
            for index in range(10):
                self._cross(base + timedelta(minutes=index), event=event)
            for index in range(10):
                self._cross(base + timedelta(minutes=30 + index),
                            DIRECTION_OUT, event=event)

        self.assertEqual(event.totals()["in"], 30, "attendance spans the run")
        self.assertEqual(event.peak_occupancy()["peak"], 10,
                         "ten at once, three times over, is not thirty at once")

    def test_capacity_remembers_a_breach_that_has_passed(self):
        event = Event.objects.create(
            name="Gallery opening",
            starts_at=timezone.now() - timedelta(hours=2),
            ends_at=timezone.now() + timedelta(hours=2),
            capacity=5,
        )
        base = timezone.localtime() - timedelta(hours=1)
        for index in range(6):
            self._cross(base + timedelta(minutes=index), event=event)
        for index in range(5):
            self._cross(base + timedelta(minutes=30 + index),
                        DIRECTION_OUT, event=event)

        state = event.capacity_state()

        self.assertEqual(state["inside"], 1)
        self.assertEqual(state["level"], "ok", "it is comfortable now")
        self.assertEqual(state["peak"], 6)
        self.assertEqual(state["peak_percent"], 120.0)
        self.assertTrue(state["breached"],
                        "'fine when you looked' is not 'never over'")

    def test_capacity_that_was_never_breached_says_so(self):
        event = Event.objects.create(
            name="Workshop",
            starts_at=timezone.now() - timedelta(hours=2),
            ends_at=timezone.now() + timedelta(hours=2),
            capacity=50,
        )
        self._cross(timezone.localtime() - timedelta(minutes=10), event=event)

        state = event.capacity_state()
        self.assertFalse(state["breached"])
        self.assertEqual(state["peak"], 1)

    def test_a_camera_reports_its_own_peak_today(self):
        now = timezone.localtime()
        for index in range(3):
            self._cross(now - timedelta(minutes=10 - index))

        self.assertEqual(self.camera.peak_today()["peak"], 3)


class PeakApiTests(TestCase):
    """The peak reaches the places an operator actually looks."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _cross(self, when, direction=DIRECTION_IN, event=None):
        Crossing.objects.create(camera=self.camera, track_id=1,
                                direction=direction, occurred_at=when,
                                event=event)

    def test_stats_reports_the_peak_so_far_today(self):
        now = timezone.localtime().replace(hour=12, minute=0)
        for index in range(4):
            self._cross(now + timedelta(minutes=index))
        self._cross(now + timedelta(minutes=10), DIRECTION_OUT)

        payload = self.client.get("/api/v1/stats/").json()

        self.assertEqual(payload["today"]["peak"]["peak"], 4)
        self.assertIsNotNone(payload["today"]["peak"]["at"])

    def test_the_running_event_reports_its_peak(self):
        event = make_event(name="Expo", offset_hours=-1, duration_hours=4,
                           capacity=3)
        base = timezone.localtime() - timedelta(minutes=30)
        for index in range(4):
            self._cross(base + timedelta(minutes=index), event=event)
        for index in range(4):
            self._cross(base + timedelta(minutes=10 + index),
                        DIRECTION_OUT, event=event)

        payload = self.client.get("/api/v1/events/current/").json()
        card = payload["events"][0]

        self.assertEqual(card["peak"]["peak"], 4)
        self.assertEqual(card["attendance"]["inside"], 0)
        self.assertTrue(card["capacity"]["breached"])

    def test_the_csv_carries_the_peak_columns(self):
        self._cross(local_at(2026, 5, 4, 9, 0))
        self._cross(local_at(2026, 5, 4, 9, 5))

        response = self.client.get(
            "/api/v1/daily/?start=2026-05-04&end=2026-05-04&format=csv")
        body = response.content.decode()
        header, first = body.splitlines()[0], body.splitlines()[1]

        self.assertEqual(header, "event,date,in,out,net,peak_inside,peak_at")
        self.assertIn(",2,", first)


class SourceClassificationTests(TestCase):
    """A stream URL is a live camera, not a file that ends.

    Getting this backwards is quiet and expensive: running out of frames
    would be filed as "the clip finished", which is a healthy state, so an
    IP camera dropping off the network at 10:20 would be reported as fine
    and never reconnected to.
    """

    def test_a_webcam_index_is_live(self):
        self.assertFalse(is_file_source(0))
        self.assertFalse(is_file_source(2))

    def test_a_path_is_a_file(self):
        self.assertTrue(is_file_source("/tmp/walk.mp4"))
        self.assertTrue(is_file_source("clips/door.avi"))

    def test_an_rtsp_camera_is_live(self):
        self.assertFalse(is_file_source("rtsp://10.0.0.8:554/stream1"))

    def test_an_http_camera_is_live(self):
        self.assertFalse(is_file_source("http://10.0.0.8:8080/video"))

    def test_the_worker_agrees(self):
        live = Camera.objects.create(name="IP cam", source="rtsp://10.0.0.8/s")
        clip = Camera.objects.create(name="Clip", source="/tmp/walk.mp4")

        self.assertFalse(CounterWorker(live).is_file)
        self.assertTrue(CounterWorker(clip).is_file)


@override_settings(COUNTER_STALE_FRAME_SECONDS=30)
class StreamFailureIsNotAFinishedClipTests(TestCase):
    """An IP camera that stops must never be reported as healthy."""

    def setUp(self):
        self.camera = Camera.objects.create(
            name="IP cam", source="rtsp://10.0.0.8/s", is_active=True)

    def test_a_dead_stream_worker_is_a_fault(self):
        worker = FakeWorker(alive=False, frames_read=5000,
                            error="Camera stopped returning frames.",
                            is_file=False)
        row = camera_health(self.camera, worker)

        self.assertEqual(row["state"], FAILED)
        self.assertFalse(row["healthy"],
                         "a stream running out is not a clip finishing")


class SupervisorFinishedClipTests(TestCase):
    """A file that played out is not a camera to revive."""

    def setUp(self):
        self.registry = RecordingRegistry()
        self.supervisor = Supervisor(registry=self.registry, cooldown=0)

    def test_a_finished_clip_is_left_finished(self):
        camera = Camera.objects.create(
            name="Demo clip", source="/tmp/walk.avi", is_active=True)
        self.registry.workers[camera.pk] = FakeWorker(
            alive=False, is_file=True, frames_read=400, error=None)

        result = self.supervisor.reconcile()

        self.assertEqual(result.started, [],
                         "replaying a clip on a timer invents visitors")
        self.assertEqual(result.finished, [camera.pk])

    def test_a_clip_that_failed_is_still_restarted(self):
        camera = Camera.objects.create(
            name="Demo clip", source="/tmp/walk.avi", is_active=True)
        self.registry.workers[camera.pk] = FakeWorker(
            alive=False, is_file=True, frames_read=0,
            error="Could not open video source")

        result = self.supervisor.reconcile()

        self.assertEqual(result.started, [camera.pk])
        self.assertEqual(result.finished, [])

    def test_a_live_camera_that_stopped_is_restarted(self):
        camera = Camera.objects.create(
            name="IP cam", source="rtsp://10.0.0.8/s", is_active=True)
        self.registry.workers[camera.pk] = FakeWorker(
            alive=False, is_file=False, frames_read=9000,
            error="Camera stopped returning frames.")

        result = self.supervisor.reconcile()

        self.assertEqual(result.started, [camera.pk],
                         "a live camera never 'finishes' on its own")


class OpenFailureHintTests(TestCase):
    """The advice must match the kind of source that failed."""

    def _error_for(self, source):
        camera = Camera.objects.create(name="X", source=source)
        worker = CounterWorker(camera)
        with patch("cv2.VideoCapture", lambda _s: FakeCapture(opened=False)):
            self.assertIsNone(worker._open())
        return worker.error

    def test_a_webcam_mentions_permissions(self):
        self.assertIn("Privacy & Security", self._error_for("0"))

    def test_a_file_mentions_the_file(self):
        message = self._error_for("/tmp/walk.mp4")
        self.assertIn("file exists", message)
        self.assertNotIn("Privacy & Security", message)

    def test_a_network_camera_mentions_the_network(self):
        message = self._error_for("rtsp://10.0.0.8/s")
        self.assertIn("reachable on the network", message)
        self.assertNotIn("Privacy & Security", message,
                         "camera permissions have nothing to do with RTSP")


# ---------------------------------------------------------------------------
# alerting
# ---------------------------------------------------------------------------

ALERTS_ON = dict(
    COUNTER_ALERTS_ENABLED=True,
    COUNTER_ALERT_AFTER_SECONDS=60,
    COUNTER_ALERT_MIN_SEVERITY=1,
    COUNTER_ALERT_WEBHOOK="",
    COUNTER_ALERT_EMAILS="",
    COUNTER_SITE_NAME="Test venue",
)


@override_settings(**ALERTS_ON)
class AlertConditionTests(TestCase):
    """What counts as worth telling somebody about."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0", is_active=True)

    def _health(self, worker):
        return health_conditions([self.camera], FakeRegistry({self.camera.pk: worker}))

    def test_a_working_camera_is_not_news(self):
        self.assertEqual(self._health(FakeWorker(seconds_since_frame=1.0)), [])

    def test_an_offline_camera_is_critical(self):
        conditions = self._health(None)

        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].level, Alert.LEVEL_CRITICAL)
        self.assertEqual(conditions[0].state, OFFLINE)
        self.assertEqual(conditions[0].key, f"health:camera:{self.camera.pk}")
        self.assertIn("Door is offline", conditions[0].message)

    def test_a_stalled_camera_is_a_warning(self):
        conditions = self._health(FakeWorker(seconds_since_frame=120.0))

        self.assertEqual(conditions[0].level, Alert.LEVEL_WARNING)
        self.assertEqual(conditions[0].state, STALLED)

    def test_a_reconnecting_camera_is_a_warning(self):
        worker = FakeWorker(seconds_since_frame=120.0, error="Stream lost.")
        worker.reconnecting = True
        worker.reconnects = 1
        worker.retry_in = 4.0

        conditions = self._health(worker)

        self.assertEqual(conditions[0].state, RECONNECTING)
        self.assertEqual(conditions[0].level, Alert.LEVEL_WARNING)

    @override_settings(COUNTER_ALERT_MIN_SEVERITY=2)
    def test_severity_two_ignores_the_merely_degraded(self):
        self.assertEqual(self._health(FakeWorker(seconds_since_frame=120.0)), [],
                         "a blinking camera should not page anyone at 2am")
        self.assertEqual(len(self._health(None)), 1,
                         "but a dead one still should")

    def test_a_deactivated_camera_is_not_a_problem(self):
        Camera.objects.filter(pk=self.camera.pk).update(is_active=False)
        self.camera.refresh_from_db()
        self.assertEqual(self._health(None), [])

    def test_a_finished_clip_is_not_a_problem(self):
        worker = FakeWorker(alive=False, is_file=True, frames_read=500)
        clip = Camera.objects.create(name="Clip", source="/tmp/x.avi")
        conditions = health_conditions([clip], FakeRegistry({clip.pk: worker}))
        self.assertEqual(conditions, [])


@override_settings(**ALERTS_ON)
class CapacityConditionTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _fill(self, event, count):
        base = timezone.localtime() - timedelta(minutes=30)
        Crossing.objects.bulk_create([
            Crossing(camera=self.camera, track_id=index, direction=DIRECTION_IN,
                     occurred_at=base + timedelta(seconds=index), event=event)
            for index in range(count)
        ])

    def _event(self, capacity=10, **kwargs):
        return make_event(name="Expo", offset_hours=-1, duration_hours=4,
                          capacity=capacity, **kwargs)

    def test_comfortable_is_not_news(self):
        self._fill(self._event(), 3)
        self.assertEqual(capacity_conditions(), [])

    def test_near_capacity_is_a_warning(self):
        self._fill(self._event(capacity=10), 9)

        conditions = capacity_conditions()

        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].state, "near")
        self.assertEqual(conditions[0].level, Alert.LEVEL_WARNING)
        self.assertIn("near capacity", conditions[0].message)
        self.assertIn("1 place left", conditions[0].message,
                      "one place, not one places")

    def test_over_capacity_is_critical(self):
        self._fill(self._event(capacity=10), 12)

        conditions = capacity_conditions()

        self.assertEqual(conditions[0].state, "over")
        self.assertEqual(conditions[0].level, Alert.LEVEL_CRITICAL)
        self.assertIn("12 inside, limit 10 (120.0%)", conditions[0].message)

    def test_an_event_that_is_not_running_is_not_watched(self):
        past = make_event(name="Yesterday", offset_hours=-30,
                          duration_hours=2, capacity=5)
        self._fill(past, 50)
        self.assertEqual(capacity_conditions(), [])

    def test_an_event_without_a_capacity_is_not_watched(self):
        self._fill(make_event(name="Open day", offset_hours=-1), 500)
        self.assertEqual(capacity_conditions(), [])


@override_settings(**ALERTS_ON)
class AlertReconcileTests(TestCase):
    """Conditions become alerts that open, escalate and resolve."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def _condition(self, state="offline", level=Alert.LEVEL_CRITICAL):
        return Condition(
            key=f"health:camera:{self.camera.pk}",
            kind=Alert.KIND_HEALTH, state=state, level=level,
            message=f"Door is {state}.", camera=self.camera, event=None,
        )

    def test_a_new_problem_opens_one_alert(self):
        opened, escalated, resolved = reconcile_alerts([self._condition()])

        self.assertEqual(len(opened), 1)
        self.assertEqual((escalated, resolved), ([], []))
        self.assertEqual(Alert.objects.count(), 1)
        self.assertTrue(Alert.objects.get().is_open)

    def test_the_same_problem_does_not_open_another(self):
        reconcile_alerts([self._condition()])
        opened, _escalated, _resolved = reconcile_alerts([self._condition()])

        self.assertEqual(opened, [],
                         "a check every 30s must not be a message every 30s")
        self.assertEqual(Alert.objects.count(), 1)

    def test_the_detail_is_kept_current_without_saying_anything(self):
        reconcile_alerts([self._condition()])
        changed = self._condition()._replace(message="Door is offline. Retrying in 8s.")
        reconcile_alerts([changed])

        alert = Alert.objects.get()
        self.assertEqual(alert.message, "Door is offline. Retrying in 8s.")
        self.assertIsNone(alert.notified_at)

    def test_getting_worse_is_not_getting_better(self):
        reconcile_alerts([self._condition(state="near",
                                          level=Alert.LEVEL_WARNING)])
        opened, escalated, resolved = reconcile_alerts([self._condition(state="over")])

        self.assertEqual(len(escalated), 1)
        self.assertEqual((opened, resolved), ([], []))

        old = Alert.objects.get(state="near")
        self.assertFalse(old.is_open)
        self.assertTrue(old.superseded,
                        "an escalation must not read as a fix")
        self.assertTrue(Alert.objects.get(state="over").is_open)

    def test_a_problem_going_away_resolves_it(self):
        reconcile_alerts([self._condition()])
        _opened, _escalated, resolved = reconcile_alerts([])

        self.assertEqual(len(resolved), 1)
        self.assertFalse(Alert.objects.get().is_open)

    def test_the_same_problem_returning_is_a_new_alert(self):
        reconcile_alerts([self._condition()])
        reconcile_alerts([])
        opened, _escalated, _resolved = reconcile_alerts([self._condition()])

        self.assertEqual(len(opened), 1)
        self.assertEqual(Alert.objects.count(), 2)
        self.assertEqual(Alert.objects.filter(resolved_at__isnull=True).count(), 1)

    def test_two_open_alerts_for_one_problem_are_impossible(self):
        reconcile_alerts([self._condition()])
        with self.assertRaises(IntegrityError):
            Alert.objects.create(
                kind=Alert.KIND_HEALTH, key=f"health:camera:{self.camera.pk}",
                state="offline", level=Alert.LEVEL_CRITICAL, message="again",
            )


@override_settings(**ALERTS_ON)
class AlertGraceTests(TestCase):
    """Nothing is sent until a problem has proved it is not a blink."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.condition = Condition(
            key="health:camera:1", kind=Alert.KIND_HEALTH, state="offline",
            level=Alert.LEVEL_CRITICAL, message="Door is offline.",
            camera=self.camera, event=None,
        )

    def test_a_brand_new_problem_waits(self):
        reconcile_alerts([self.condition])
        to_announce, to_clear = pending()

        self.assertEqual((to_announce, to_clear), ([], []))

    def test_it_is_announced_once_the_grace_passes(self):
        past = timezone.now() - timedelta(seconds=90)
        reconcile_alerts([self.condition], now=past)

        to_announce, _to_clear = pending()
        self.assertEqual(len(to_announce), 1)

    def test_a_blink_tells_nobody(self):
        # Appeared and gone again inside the grace period.
        reconcile_alerts([self.condition])
        reconcile_alerts([])

        to_announce, to_clear = pending()
        self.assertEqual((to_announce, to_clear), ([], []),
                         "two seconds of trouble is not worth a phone buzzing")
        alert = Alert.objects.get()
        self.assertFalse(alert.is_open)
        self.assertIsNone(alert.notified_at)

    def test_a_recovery_is_owed_only_if_the_problem_was_announced(self):
        past = timezone.now() - timedelta(seconds=90)
        reconcile_alerts([self.condition], now=past)
        Alert.objects.update(notified_at=timezone.now())
        reconcile_alerts([])

        _to_announce, to_clear = pending()
        self.assertEqual(len(to_clear), 1)

    def test_a_superseded_alert_owes_no_recovery(self):
        past = timezone.now() - timedelta(seconds=90)
        reconcile_alerts([self.condition], now=past)
        Alert.objects.update(notified_at=timezone.now())
        reconcile_alerts([self.condition._replace(state="worse")])

        _to_announce, to_clear = pending()
        self.assertEqual(to_clear, [],
                         "'resolved' would be a lie when it got worse")


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class WebhookRecorder:
    """Stands in for urlopen, keeping what would have gone out."""

    def __init__(self, error=None):
        self.requests = []
        self.error = error

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return FakeResponse()

    @property
    def payloads(self):
        return [json.loads(request.data.decode()) for request in self.requests]


NOWHERE = dict(ALERTS_ON, COUNTER_ALERT_AFTER_SECONDS=0)
TO_WEBHOOK = dict(NOWHERE, COUNTER_ALERT_WEBHOOK="https://hooks.example.com/abc")
TO_EMAIL = dict(
    NOWHERE,
    COUNTER_ALERT_EMAILS="ops@example.com, manager@example.com",
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)


class AlertDeliveryTests(TestCase):
    """Getting the message out, and being honest when it did not go."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.alert = Alert.objects.create(
            kind=Alert.KIND_HEALTH, key="health:camera:1", state="offline",
            level=Alert.LEVEL_CRITICAL, message="Door is offline.",
            camera=self.camera,
        )

    @override_settings(**TO_WEBHOOK)
    def test_the_webhook_gets_readable_text_and_the_detail(self):
        recorder = WebhookRecorder()
        with patch("counter.alerts.urllib.request.urlopen", recorder):
            delivered, error = deliver(self.alert)

        self.assertTrue(delivered)
        self.assertEqual(error, "")
        self.assertEqual(len(recorder.requests), 1)

        request = recorder.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")

        payload = recorder.payloads[0]
        # Slack and Teams both render "text"; the rest is for anything else.
        self.assertEqual(payload["text"],
                         "[Test venue] CRITICAL: Door is offline.")
        self.assertEqual(payload["alert"]["state"], "offline")
        self.assertEqual(payload["alert"]["camera"], "Door")
        self.assertFalse(payload["alert"]["resolved"])

    @override_settings(**dict(NOWHERE, COUNTER_ALERT_WEBHOOK="file:///etc/passwd"))
    def test_a_webhook_must_be_http(self):
        delivered, error = deliver(self.alert)

        self.assertFalse(delivered)
        self.assertIn("must be http or https", error)

    @override_settings(**TO_EMAIL)
    def test_email_reaches_every_recipient(self):
        delivered, error = deliver(self.alert)

        self.assertTrue(delivered)
        self.assertEqual(error, "")
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(sorted(mail.outbox[0].to),
                         ["manager@example.com", "ops@example.com"])
        self.assertIn("Door is offline.", mail.outbox[0].subject)

    @override_settings(**NOWHERE)
    def test_with_nowhere_to_send_it_says_so(self):
        delivered, error = deliver(self.alert)

        self.assertFalse(delivered)
        self.assertEqual(error, "No alert channel configured.")

    @override_settings(**TO_WEBHOOK)
    def test_one_channel_failing_does_not_hide_the_other(self):
        with override_settings(COUNTER_ALERT_EMAILS="ops@example.com",
                               EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend"):
            recorder = WebhookRecorder(error=urllib.error.URLError("no route"))
            with patch("counter.alerts.urllib.request.urlopen", recorder):
                delivered, error = deliver(self.alert)

        self.assertTrue(delivered, "the email still went")
        self.assertIn("webhook:", error, "but the failure is recorded")
        self.assertEqual(len(mail.outbox), 1)

    @override_settings(**NOWHERE)
    def test_the_wording_says_what_happened(self):
        self.assertEqual(notification_text(self.alert),
                         "[Test venue] CRITICAL: Door is offline.")

        self.alert.level = Alert.LEVEL_WARNING
        self.assertTrue(notification_text(self.alert).startswith(
            "[Test venue] Warning:"))

        self.alert.started_at = timezone.now() - timedelta(minutes=12)
        self.alert.resolved_at = timezone.now()
        self.assertEqual(
            notification_text(self.alert, resolved=True),
            "[Test venue] Resolved after 12m: Door is offline.")


@override_settings(**ALERTS_ON)
class SendPendingTests(TestCase):
    """What is owed goes out once, and a failure is not lost."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.condition = Condition(
            key="health:camera:1", kind=Alert.KIND_HEALTH, state="offline",
            level=Alert.LEVEL_CRITICAL, message="Door is offline.",
            camera=self.camera, event=None,
        )
        reconcile_alerts([self.condition],
                         now=timezone.now() - timedelta(seconds=90))

    @override_settings(**TO_WEBHOOK)
    def test_it_goes_out_once_and_not_again(self):
        recorder = WebhookRecorder()
        with patch("counter.alerts.urllib.request.urlopen", recorder):
            sent, failed = send_pending()
            self.assertEqual((len(sent), failed), (1, []))

            sent_again, _failed = send_pending()

        self.assertEqual(sent_again, [], "an open alert is not a repeating alarm")
        self.assertEqual(len(recorder.requests), 1)
        self.assertTrue(Alert.objects.get().delivered)

    @override_settings(**TO_WEBHOOK)
    def test_a_failed_send_is_retried_rather_than_lost(self):
        failing = WebhookRecorder(error=urllib.error.URLError("timed out"))
        with patch("counter.alerts.urllib.request.urlopen", failing):
            sent, failed = send_pending()

        self.assertEqual((sent, len(failed)), ([], 1))
        alert = Alert.objects.get()
        self.assertIsNone(alert.notified_at, "still owed")
        self.assertIn("timed out", alert.delivery_error)

        working = WebhookRecorder()
        with patch("counter.alerts.urllib.request.urlopen", working):
            sent, failed = send_pending()

        self.assertEqual((len(sent), failed), (1, []))
        self.assertTrue(Alert.objects.get().delivered)

    @override_settings(**NOWHERE)
    def test_with_nowhere_to_send_it_is_not_retried_for_ever(self):
        sent, failed = send_pending()

        self.assertEqual((len(sent), failed), (1, []))
        alert = Alert.objects.get()
        self.assertIsNotNone(alert.notified_at, "nothing left to try")
        self.assertFalse(alert.delivered,
                         "but nobody was told, and it must not claim otherwise")
        self.assertEqual(alert.delivery_error, "No alert channel configured.")

    @override_settings(**TO_WEBHOOK)
    def test_recovery_is_announced_to_whoever_heard_the_problem(self):
        recorder = WebhookRecorder()
        with patch("counter.alerts.urllib.request.urlopen", recorder):
            send_pending()
            reconcile_alerts([])
            sent, _failed = send_pending()

        self.assertEqual(len(sent), 1)
        self.assertEqual(len(recorder.requests), 2)
        self.assertIn("Resolved after", recorder.payloads[1]["text"])
        self.assertTrue(recorder.payloads[1]["alert"]["resolved"])

    @override_settings(**TO_WEBHOOK)
    def test_a_full_pass_ties_it_together(self):
        Alert.objects.all().delete()
        Camera.objects.filter(pk=self.camera.pk).update(is_active=True)
        recorder = WebhookRecorder()

        with patch("counter.alerts.urllib.request.urlopen", recorder):
            # Nothing is running, so the camera is offline.
            outcome = run_checks(registry=FakeRegistry())

        self.assertEqual(len(outcome.opened), 1)
        self.assertEqual(len(outcome.sent), 1)
        self.assertIn("is offline", recorder.payloads[0]["text"])


@override_settings(**ALERTS_ON)
class AlertApiTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.old = Alert.objects.create(
            kind=Alert.KIND_HEALTH, key="health:camera:9", state="offline",
            level=Alert.LEVEL_CRITICAL, message="Old problem", camera=self.camera,
            started_at=timezone.now() - timedelta(hours=2),
            resolved_at=timezone.now() - timedelta(hours=1),
            notified_at=timezone.now() - timedelta(hours=2),
        )
        self.live = Alert.objects.create(
            kind=Alert.KIND_CAPACITY, key="capacity:event:1", state="over",
            level=Alert.LEVEL_CRITICAL, message="Hall is over capacity",
            started_at=timezone.now() - timedelta(minutes=5),
        )

    def test_open_alerts_come_first(self):
        payload = self.client.get("/api/v1/alerts/").json()

        self.assertEqual(payload["open"], 1)
        self.assertEqual([a["message"] for a in payload["alerts"]],
                         ["Hall is over capacity", "Old problem"])
        self.assertTrue(payload["alerts"][0]["open"])
        self.assertFalse(payload["alerts"][1]["open"])

    def test_only_what_is_wrong_now(self):
        payload = self.client.get("/api/v1/alerts/?open=1").json()
        self.assertEqual(len(payload["alerts"]), 1)

    def test_filtering_by_kind(self):
        payload = self.client.get("/api/v1/alerts/?kind=capacity").json()
        self.assertEqual([a["kind"] for a in payload["alerts"]], ["capacity"])

    def test_an_unknown_kind_is_a_400(self):
        response = self.client.get("/api/v1/alerts/?kind=weather")
        self.assertEqual(response.status_code, 400)

    def test_it_says_whether_anyone_would_be_told(self):
        payload = self.client.get("/api/v1/alerts/").json()

        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["channels"], [],
                         "nothing configured, and the dashboard should say so")
        self.assertFalse(payload["alerts"][0]["delivered"])

    @override_settings(**TO_WEBHOOK)
    def test_channels_are_listed_when_configured(self):
        payload = self.client.get("/api/v1/alerts/").json()
        self.assertEqual(payload["channels"], ["webhook"])


@override_settings(**NOWHERE)
class CheckAlertsCommandTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0", is_active=True)

    def test_dry_run_reports_without_recording(self):
        out = StringIO()
        call_command("check_alerts", "--dry-run", stdout=out)

        self.assertIn("Door is offline", out.getvalue())
        self.assertEqual(Alert.objects.count(), 0,
                         "a dry run must not leave anything behind")

    def test_a_real_pass_opens_and_reports(self):
        out = StringIO()
        call_command("check_alerts", stdout=out)

        self.assertIn("opened:", out.getvalue())
        self.assertEqual(Alert.objects.count(), 1)

    def test_nothing_wrong_says_so(self):
        Camera.objects.all().delete()
        out = StringIO()
        call_command("check_alerts", "--dry-run", stdout=out)
        self.assertIn("nothing wrong", out.getvalue())

    @override_settings(**TO_WEBHOOK)
    def test_a_delivery_failure_exits_non_zero(self):
        failing = WebhookRecorder(error=urllib.error.URLError("nope"))
        with patch("counter.alerts.urllib.request.urlopen", failing):
            with self.assertRaises(SystemExit) as caught:
                call_command("check_alerts", stderr=StringIO(), stdout=StringIO())

        self.assertEqual(caught.exception.code, 1)


class SupervisorAlertHookTests(TestCase):
    """The supervisor is where alerting lives, at its own slower cadence."""

    def setUp(self):
        self.supervisor = Supervisor(registry=RecordingRegistry(), interval=1)

    @override_settings(COUNTER_ALERTS_ENABLED=False)
    def test_nothing_happens_when_alerting_is_off(self):
        with patch("counter.alerts.run_checks") as checks:
            self.assertIsNone(self.supervisor.maybe_check_alerts())
        checks.assert_not_called()

    @override_settings(**dict(ALERTS_ON, COUNTER_ALERT_INTERVAL=30))
    def test_it_checks_then_waits_its_interval(self):
        supervisor = Supervisor(registry=RecordingRegistry(), interval=1)
        with patch("counter.alerts.run_checks") as checks:
            supervisor.maybe_check_alerts(now=1000.0)
            supervisor.maybe_check_alerts(now=1010.0)
            self.assertEqual(checks.call_count, 1,
                             "reconciling every 10s must not alert every 10s")
            supervisor.maybe_check_alerts(now=1031.0)
            self.assertEqual(checks.call_count, 2)

    @override_settings(**ALERTS_ON)
    def test_a_failing_check_does_not_stop_the_supervisor(self):
        with patch("counter.alerts.run_checks", side_effect=RuntimeError("boom")):
            self.assertIsNone(self.supervisor.maybe_check_alerts())


@override_settings(**ALERTS_ON)
class AlertOrderingTests(TestCase):
    """A long-running problem must not be pushed off the list by fixed ones."""

    def test_an_old_open_alert_outranks_newer_resolved_ones(self):
        old_open = Alert.objects.create(
            kind=Alert.KIND_HEALTH, key="health:camera:1", state="offline",
            level=Alert.LEVEL_CRITICAL, message="Still broken",
            started_at=timezone.now() - timedelta(days=2),
        )
        for index in range(5):
            Alert.objects.create(
                kind=Alert.KIND_HEALTH, key=f"health:camera:{index + 2}",
                state="stalled", level=Alert.LEVEL_WARNING,
                message=f"Fixed {index}",
                started_at=timezone.now() - timedelta(minutes=index + 1),
                resolved_at=timezone.now(),
            )

        payload = self.client.get("/api/v1/alerts/?limit=3").json()

        self.assertEqual(payload["alerts"][0]["id"], old_open.pk)
        self.assertEqual(len(payload["alerts"]), 3)


# ---------------------------------------------------------------------------
# standing on the line is not walking through it
# ---------------------------------------------------------------------------

class LineHysteresisTests(TestCase):
    """One person hesitating in a doorway used to count as eleven.

    A track that hovers across the line flips sides on every sway, and each
    flip was a crossing. The guard is distance: a crossing counts once the
    track is clear of the line by a fraction of its own size, which a sway
    never manages at any speed.

    There was a second guard, a frame count, and either one would commit the
    crossing. It was measured and removed: a frame count cannot tell a slow
    sway from a walk, because a sway lingers at the far end of its arc for
    as long as a walk takes to leave.
    """

    LINE_Y = 120          # a 240px frame, line at 0.5

    def _run(self, ys, position=0.5, **kwargs):
        counter = VisitorCounter(
            detector=StubDetector([[box_at(160, y)] for y in ys]),
            tracker=build_tracker("sort", max_disappeared=25),
            line=LineCounter(position=position, **kwargs),
            detect_every=1,
        )
        frame = blank_frame()
        for _ in ys:
            counter.process(frame)
        return counter.total_in, counter.total_out

    @staticmethod
    def _sway(amplitude, times=6, line=120):
        points = [40, 60, 90]
        for _ in range(times):
            points += [line - amplitude, line + amplitude]
        return points

    def test_swaying_on_the_line_is_not_a_dozen_visitors(self):
        for amplitude in (30, 40, 50):
            with self.subTest(amplitude=amplitude):
                total = sum(self._run(self._sway(amplitude)))
                self.assertLessEqual(
                    total, 1,
                    f"swaying +/-{amplitude}px counted {total} crossings")

    def test_the_old_rule_really_did_inflate(self):
        # Guards the guard: if this ever stops failing, the scenario has
        # drifted and the test above proves nothing.
        total = sum(self._run(self._sway(40), commit_fraction=0.0))
        self.assertGreater(total, 5,
                           "counting on the first flip should inflate here")

    def test_walking_through_still_counts_at_any_speed(self):
        for step in (5, 10, 15, 20, 30, 40, 60, 80):
            with self.subTest(step=step):
                total_in, _out = self._run(list(range(10, 235, step)))
                self.assertEqual(total_in, 1, f"missed a walk at {step}px/frame")

    def test_a_real_change_of_mind_still_counts(self):
        def leg(start, end, step):
            direction = 1 if end > start else -1
            return list(range(start, end, direction * step)) + [end]

        ys = leg(20, 220, 15) + leg(220, 20, 15) + leg(20, 220, 15)
        self.assertEqual(self._run(ys), (2, 1))

    def test_distance_alone_is_enough_for_a_fast_crossing(self):
        # One frame past the line, which is too few to "hold" it, but far
        # enough that it cannot be a sway.
        counted = self._run([40, 60, 90, 220])
        self.assertEqual(counted[0], 1,
                         "a sprinter has no frames to spare")

    def test_a_slow_sway_is_not_counted_however_long_it_lasts(self):
        """The failure that the frame count let through.

        A person seated at a webcam, or standing in a doorway holding it
        open, drifts across the line and back on a slow arc. Each far end of
        that arc held the new side for many frames, so the frame-count guard
        committed a crossing every time: 45 seconds of it produced 29.
        """
        line, amplitude, period = 120, 40, 60      # 3 seconds at 20fps
        ys = [40, 60, 90]
        for frame in range(900):                   # 45 seconds
            ys.append(line + amplitude * math.sin(2 * math.pi * frame / period))
        total = sum(self._run(ys))
        self.assertEqual(total, 0,
                         f"a slow sway counted {total} crossings")

    def test_not_getting_clear_of_the_line_means_no_crossing(self):
        counted = self._run([40, 60, 90, 135], commit_fraction=10.0)
        self.assertEqual(counted, (0, 0))

    def test_stopping_just_over_the_line_is_not_counted(self):
        """A known cost of dropping the frame count, recorded on purpose.

        Somebody who steps over the line and stops short of their own body
        width -- holding a door, turning to talk -- is not counted. The
        frame count used to catch this, at the price of counting every sway.
        """
        counted = self._run([40, 60, 90, 135] + [137] * 200)
        self.assertEqual(counted, (0, 0),
                         "if this starts counting, the trade-off has changed")


class CommitDistanceTests(TestCase):
    """The distance guard scales with the person, not with the lens."""

    def test_a_bigger_person_must_travel_further(self):
        line = LineCounter(position=0.5, commit_fraction=0.5)
        near = Track(track_id=1, centroid=(0, 0), box=(0, 0, 40, 200))
        far = Track(track_id=2, centroid=(0, 0), box=(0, 0, 10, 40))

        # 40px past the line: nothing for someone filling the frame,
        # plenty for someone at the far end of a hall.
        self.assertFalse(line._is_clear_of_line(near, 40))
        self.assertTrue(line._is_clear_of_line(far, 40))

    def test_a_missing_box_is_never_clear_of_the_line(self):
        line = LineCounter(position=0.5)
        self.assertFalse(line._is_clear_of_line(
            Track(track_id=1, centroid=(0, 0), box=None), 500))

    def test_a_vertical_line_measures_width(self):
        line = LineCounter(position=0.5, orientation=VERTICAL,
                           commit_fraction=0.5)
        wide = Track(track_id=1, centroid=(0, 0), box=(0, 0, 200, 40))
        self.assertFalse(line._is_clear_of_line(wide, 40))
        self.assertTrue(line._is_clear_of_line(wide, 120))


class SuggestedLineStaysOffTheEdgeTests(TestCase):
    """A line at 0.95 leaves nowhere to confirm a crossing."""

    def test_the_suggestion_is_pulled_away_from_the_edge(self):
        # Everyone walking along the very bottom of the frame.
        centroids = [(x, 340) for x in range(20, 300, 20)]
        suggestion = suggest_line_setup(centroids, (360, 640, 3))

        self.assertIsNotNone(suggestion)
        self.assertGreaterEqual(suggestion["position"], LINE_POSITION_MIN)
        self.assertLessEqual(suggestion["position"], LINE_POSITION_MAX)

    def test_a_central_suggestion_is_left_alone(self):
        centroids = [(x, 180) for x in range(20, 620, 20)]
        suggestion = suggest_line_setup(centroids, (360, 640, 3))
        self.assertAlmostEqual(suggestion["position"], 0.5, places=1)


class MeasureCommandTests(TransactionTestCase):
    """Counting a recorded clip, so accuracy can be a number not an opinion."""

    FRAMES = 40

    def _clip(self, path):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (FRAME_W, FRAME_H))
        if not writer.isOpened():
            return False
        for index in range(self.FRAMES):
            frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
            y = int(20 + 200 * index / (self.FRAMES - 1))
            cv2.rectangle(frame, (140, y - 40), (180, y + 40), (200, 200, 200), -1)
            writer.write(frame)
        writer.release()
        return path.exists() and path.stat().st_size > 0

    def _walker(self):
        """One person walking down the frame, past the middle."""
        return StubDetector([
            [box_at(160, int(20 + 200 * i / (self.FRAMES - 1)))]
            for i in range(self.FRAMES)
        ])

    def _run(self, clip, *args):
        out = StringIO()
        with patch("counter.management.commands.measure.build_detector",
                   return_value=self._walker()):
            call_command("measure", str(clip), "--detect-every", "1",
                         "--quiet", *args, stdout=out)
        return out.getvalue()

    def test_it_counts_a_clip_without_touching_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = pathlib.Path(tmp) / "door.avi"
            if not self._clip(clip):
                self.skipTest("OpenCV build cannot write MJPG video here")

            output = self._run(clip)

            self.assertIn("counted: 1 in, 0 out", output)
            self.assertEqual(Crossing.objects.count(), 0,
                             "measuring is not a shift; it records nothing")
            self.assertEqual(Camera.objects.count(), 0)

    def test_it_reports_the_error_against_a_hand_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = pathlib.Path(tmp) / "door.avi"
            if not self._clip(clip):
                self.skipTest("OpenCV build cannot write MJPG video here")

            output = self._run(clip, "--expect-in", "2", "--expect-out", "0")

            self.assertIn("counted 1, truth 2", output)
            self.assertIn("-1", output, "the error itself must be stated")
            self.assertIn("50.0%", output)

    def test_every_crossing_is_logged_with_its_place_in_the_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = pathlib.Path(tmp) / "door.avi"
            if not self._clip(clip):
                self.skipTest("OpenCV build cannot write MJPG video here")
            report = pathlib.Path(tmp) / "found.csv"

            self._run(clip, "--csv", str(report))

            rows = list(csv.DictReader(report.open()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["direction"], DIRECTION_IN)
            # The frame and timestamp are the point: they are what lets
            # somebody open the recording at the moment in dispute.
            self.assertTrue(rows[0]["at"].startswith("00:"))
            self.assertGreater(int(rows[0]["frame"]), 0)

    def test_it_can_write_back_an_annotated_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = pathlib.Path(tmp) / "door.avi"
            if not self._clip(clip):
                self.skipTest("OpenCV build cannot write MJPG video here")
            checked = pathlib.Path(tmp) / "checked.avi"

            self._run(clip, "--record", str(checked))

            self.assertTrue(checked.exists())
            playback = cv2.VideoCapture(str(checked))
            try:
                self.assertTrue(playback.isOpened(), "recording is not readable")
                ok, frame = playback.read()
                self.assertTrue(ok)
                self.assertIsNotNone(frame)
            finally:
                playback.release()

    def test_a_missing_clip_is_a_clear_error(self):
        with self.assertRaises(CommandError) as caught:
            call_command("measure", "/no/such/clip.mp4", stdout=StringIO())
        self.assertIn("No such file", str(caught.exception))


class CountingDefaultsAgreeTests(TestCase):
    """The settings file must not quietly disagree with the measured default.

    It did once: vision was tuned after measuring, settings still held the
    rejected number, so every real camera ran on the number that had been
    rejected and only the unit tests saw the tuned one.
    """

    def test_the_commit_fraction_default_is_the_one_that_was_measured(self):
        self.assertEqual(settings.COUNTER_LINE_COMMIT_FRACTION,
                         DEFAULT_COMMIT_FRACTION)

    def test_a_worker_gets_it(self):
        camera = Camera.objects.create(name="Door", source="0")
        worker = CounterWorker(camera)
        self.assertEqual(worker.counter.line.commit_fraction,
                         DEFAULT_COMMIT_FRACTION)


# ---------------------------------------------------------------------------
# the macOS bundle: an app that owns its own camera permission
# ---------------------------------------------------------------------------

class InfoPlistTests(TestCase):
    """macOS needs a reason string, or it kills the app instead of asking."""

    def test_the_camera_reason_is_present_and_says_something(self):
        plist = build_app.info_plist()
        reason = plist["NSCameraUsageDescription"]
        self.assertIn("camera", reason.lower())
        self.assertGreater(len(reason), 40, "a one-word reason reads as spam")

    def test_the_executable_name_matches_the_one_that_gets_written(self):
        # If these drift the bundle launches nothing and macOS says it is
        # damaged, with no hint as to why.
        plist = build_app.info_plist()
        self.assertEqual(plist["CFBundleExecutable"], build_app.APP_NAME)

    def test_the_bundle_identifier_is_stable(self):
        # TCC remembers the answer against this string. Change it and the
        # camera permission is silently forgotten.
        self.assertEqual(build_app.info_plist()["CFBundleIdentifier"],
                         "com.visitorstracker.desktop")

    def test_no_icon_means_no_icon_key(self):
        self.assertNotIn("CFBundleIconFile", build_app.info_plist())

    def test_an_icon_is_named_by_its_filename(self):
        plist = build_app.info_plist(icon="/somewhere/Else/Logo.icns")
        self.assertEqual(plist["CFBundleIconFile"], "Logo.icns")


class LauncherScriptTests(TestCase):
    """The bundle's main executable."""

    def test_it_runs_the_desktop_module_with_the_given_interpreter(self):
        script = build_app.launcher_script("/p/VisitorsTracker", "/p/.venv/bin/python")
        self.assertTrue(script.startswith("#!/bin/sh"))
        self.assertIn("-m counter.desktop", script)
        self.assertIn("'/p/.venv/bin/python'", script)
        self.assertIn("cd '/p/VisitorsTracker'", script)

    def test_a_path_with_spaces_survives(self):
        # "CPA Latest FP Update" is a real directory on this machine.
        script = build_app.launcher_script(
            "/Users/me/My Projects/VT", "/Users/me/My Projects/VT/.venv/bin/python")
        self.assertIn("cd '/Users/me/My Projects/VT'", script)

    def test_a_quote_in_the_path_cannot_break_out(self):
        script = build_app.launcher_script("/tmp/it's here", "/tmp/py")
        self.assertIn("'/tmp/it'\\''s here'", script)


class BuildAppCommandTests(TestCase):
    """Building the bundle."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # A stand-in interpreter, so the test does not depend on a venv.
        self.python = pathlib.Path(self.tmp) / "python"
        self.python.write_text("#!/bin/sh\n")
        self.bundle = pathlib.Path(self.tmp) / "out" / "VisitorsTracker.app"

    def build(self, **extra):
        out = StringIO()
        call_command("build_app", output=str(pathlib.Path(self.tmp) / "out"),
                     python=str(self.python), no_sign=True, stdout=out, **extra)
        return out.getvalue()

    def test_it_produces_a_launchable_bundle(self):
        self.build()
        plist_path = self.bundle / "Contents" / "Info.plist"
        executable = self.bundle / "Contents" / "MacOS" / "VisitorsTracker"
        self.assertTrue(plist_path.exists())
        self.assertTrue(executable.exists())
        self.assertTrue(os.access(executable, os.X_OK),
                        "macOS will not launch a bundle whose binary is not +x")
        with open(plist_path, "rb") as handle:
            plist = plistlib.load(handle)
        self.assertIn("NSCameraUsageDescription", plist)
        self.assertEqual(plist["CFBundleExecutable"], "VisitorsTracker")

    def test_rebuilding_over_an_existing_bundle_works(self):
        self.build()
        marker = self.bundle / "Contents" / "stale.txt"
        marker.write_text("left over from the last build")
        self.build()
        self.assertFalse(marker.exists(),
                         "a rebuild must not leave the previous build behind")

    def test_a_missing_interpreter_is_refused(self):
        with self.assertRaises(CommandError):
            call_command("build_app", python="/nowhere/python", no_sign=True,
                         output=str(pathlib.Path(self.tmp) / "out"),
                         stdout=StringIO())

    def test_a_missing_icon_is_refused(self):
        with self.assertRaises(CommandError):
            self.build(icon="/nowhere/Logo.icns")

    def test_an_icon_is_copied_in(self):
        icon = pathlib.Path(self.tmp) / "Logo.icns"
        icon.write_bytes(b"not really an icon")
        self.build(icon=str(icon))
        self.assertTrue((self.bundle / "Contents" / "Resources" / "Logo.icns").exists())


class DesktopPortTests(TestCase):
    """Two copies of the app must not fight over the camera."""

    def test_a_bound_port_is_not_free(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            self.assertFalse(desktop.port_is_free(port))

    def test_an_unbound_port_is_free(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.assertTrue(desktop.port_is_free(port))


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b'{"status": "ok", "cameras": []}'
    status = 200

    def do_GET(self):
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


class AlreadyServingTests(TestCase):
    """Launching the app twice opens a window on the first one."""

    def serve(self, handler_cls):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_our_own_health_response_is_recognised(self):
        port = self.serve(_Handler)
        self.assertTrue(desktop.ours_is_already_serving(port=port))

    def test_somebody_else_on_the_port_is_not_us(self):
        class Other(_Handler):
            body = b"<html>a totally different app</html>"
        port = self.serve(Other)
        self.assertFalse(desktop.ours_is_already_serving(port=port))

    def test_nothing_on_the_port_is_not_us(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.assertFalse(desktop.ours_is_already_serving(port=port, timeout=0.5))


class WaitForServerTests(TestCase):
    """The window must not open on a connection error."""

    def test_it_returns_once_the_server_answers(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        self.assertTrue(desktop.wait_for_server(url, timeout=5))

    def test_a_404_still_counts_as_up(self):
        class NotFound(_Handler):
            status = 404
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NotFound)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        self.assertTrue(desktop.wait_for_server(url, timeout=5),
                        "the server answered; the path simply does not exist")

    def test_it_gives_up_on_a_dead_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        started = time_module.monotonic()
        self.assertFalse(
            desktop.wait_for_server(f"http://127.0.0.1:{port}/", timeout=1))
        self.assertLess(time_module.monotonic() - started, 10)


class DesktopServerDoesNotHangOnQuitTests(TestCase):
    """Closing the window has to end the process, live view or not.

    Django's ThreadedWSGIServer blocks on close until every request thread
    finishes. An MJPEG live view is a request that never finishes, so with
    the default the app would stay running invisibly after its window went
    away, still holding the camera.
    """

    def test_the_server_class_does_not_block_on_close(self):
        httpd, _thread = desktop.build_server(port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.assertFalse(httpd.block_on_close)
        self.assertTrue(httpd.daemon_threads)


class DesktopShutdownTests(TestCase):
    """Quitting releases the camera."""

    def test_it_stops_the_workers_and_the_server(self):
        calls = []

        class FakeServer:
            def shutdown(self):
                calls.append("shutdown")

            def server_close(self):
                calls.append("close")

        class FakeSupervisor:
            def stop(self):
                calls.append("supervisor")

        with patch.object(camera_module.REGISTRY, "stop_all",
                          lambda: calls.append("stop_all")):
            desktop.shutdown(FakeServer(), FakeSupervisor())

        self.assertEqual(calls, ["supervisor", "stop_all", "shutdown", "close"])

    def test_it_is_safe_with_nothing_running(self):
        with patch.object(camera_module.REGISTRY, "stop_all", lambda: None):
            desktop.shutdown(None, None)       # must not raise


class RestartingOnTheSamePortTests(TestCase):
    """Quitting and relaunching must not hit a minute of "port busy".

    The probe used to bind without SO_REUSEADDR while the server binds with
    it, so connections left in TIME_WAIT by the run that just ended made the
    app refuse to start on its own port.
    """

    def test_the_port_is_free_again_as_soon_as_the_server_stops(self):
        httpd, _thread = desktop.build_server(port=0)
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}/api/v1/health/"

        # Real connections, so the kernel leaves real TIME_WAIT entries.
        for _ in range(3):
            try:
                urllib.request.urlopen(url, timeout=5).read()
            except urllib.error.HTTPError:
                pass
        httpd.shutdown()
        httpd.server_close()

        self.assertTrue(
            desktop.port_is_free(port),
            "the app would have refused to restart on its own port")

    def test_the_probe_agrees_with_a_live_server(self):
        httpd, _thread = desktop.build_server(port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.assertFalse(desktop.port_is_free(httpd.server_address[1]))


class DefaultIconTests(TestCase):
    """The bundle should carry the project's icon without being asked."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.python = pathlib.Path(self.tmp) / "python"
        self.python.write_text("#!/bin/sh\n")
        self.out = pathlib.Path(self.tmp) / "out"
        self.bundle = self.out / "VisitorsTracker.app"

    def build(self, **extra):
        call_command("build_app", output=str(self.out), python=str(self.python),
                     no_sign=True, stdout=StringIO(), **extra)

    def icon_key(self):
        with open(self.bundle / "Contents" / "Info.plist", "rb") as handle:
            return plistlib.load(handle).get("CFBundleIconFile")

    def test_the_project_icon_is_picked_up_with_no_flag(self):
        shipped = pathlib.Path(settings.BASE_DIR) / build_app.DEFAULT_ICON
        if not shipped.exists():
            self.skipTest(f"{build_app.DEFAULT_ICON} is not in this checkout")
        self.build()
        self.assertEqual(self.icon_key(), "AppIcon.icns")
        self.assertTrue(
            (self.bundle / "Contents" / "Resources" / "AppIcon.icns").exists())

    def test_no_icon_builds_without_one(self):
        self.build(no_icon=True)
        self.assertIsNone(self.icon_key())

    def test_an_explicit_icon_wins_over_the_default(self):
        mine = pathlib.Path(self.tmp) / "Other.icns"
        mine.write_bytes(b"icns")
        self.build(icon=str(mine))
        self.assertEqual(self.icon_key(), "Other.icns")


class ShippedIconTests(TestCase):
    """The icon macOS is given has to suit the way macOS draws it.

    macOS 26 and later draw their own rounded container and place the app's
    icon inside it. A pre-rounded icon with a margin then renders as a
    rounded square inside a rounded square, shrunk and ringed in grey, so
    the artwork has to reach the edges and let the system do the rounding.
    """

    def test_the_master_is_square_and_reaches_the_edges(self):
        master = pathlib.Path(settings.BASE_DIR) / "assets" / "AppIcon.png"
        if not master.exists():
            self.skipTest("assets/AppIcon.png is not in this checkout")

        from PIL import Image

        image = Image.open(master).convert("RGBA")
        self.assertEqual(image.width, image.height, "an app icon is square")

        alpha = np.array(image)[:, :, 3]
        transparent = (alpha <= 250).mean()
        self.assertLess(
            transparent, 0.001,
            f"{transparent:.1%} of the icon is transparent; macOS would draw "
            "its own container around it and the result is a rounded square "
            "inside a rounded square")


# ---------------------------------------------------------------------------
# running somewhere other than a Mac
# ---------------------------------------------------------------------------

class CaptureBackendTests(TestCase):
    """OpenCV's default backend is not the right one everywhere.

    These assert the choice on platforms this project is not tested on, so
    the Windows path is at least pinned rather than assumed.
    """

    def backends(self, platform, source=0):
        with patch.object(camera_module.sys, "platform", platform):
            return camera_module.capture_backends(source)

    def test_windows_prefers_directshow_over_media_foundation(self):
        order = self.backends("win32")
        self.assertEqual(order[0], cv2.CAP_DSHOW)
        self.assertIn(cv2.CAP_MSMF, order)
        self.assertEqual(order[-1], cv2.CAP_ANY,
                         "the default must stay as a last resort")

    def test_linux_prefers_v4l2(self):
        self.assertEqual(self.backends("linux")[0], cv2.CAP_V4L2)

    def test_macos_is_left_on_the_default(self):
        self.assertEqual(self.backends("darwin"), [cv2.CAP_ANY])

    def test_a_url_never_gets_a_camera_backend(self):
        # DirectShow cannot open RTSP; naming it would break network cameras.
        for platform in ("win32", "linux", "darwin"):
            with self.subTest(platform=platform):
                self.assertEqual(
                    self.backends(platform, "rtsp://cam/stream"), [cv2.CAP_ANY])

    def test_a_file_never_gets_a_camera_backend(self):
        self.assertEqual(self.backends("win32", "/clips/door.mp4"), [cv2.CAP_ANY])


class CameraPrivacyHintTests(TestCase):
    """The advice has to name the right place on the right system."""

    def hint(self, platform):
        with patch.object(camera_module.sys, "platform", platform):
            worker = CounterWorker(Camera.objects.create(name="D", source="0"))
            return worker._open_hint()

    def test_windows_points_at_windows_settings(self):
        hint = self.hint("win32")
        self.assertIn("Privacy & security", hint)
        self.assertIn("desktop apps", hint)
        self.assertNotIn("System Settings", hint,
                         "that is the macOS pane, on Windows")

    def test_macos_points_at_system_settings(self):
        self.assertIn("System Settings > Privacy & Security > Camera",
                      self.hint("darwin"))

    def test_linux_does_not_send_people_to_a_settings_app(self):
        self.assertIn("video", self.hint("linux"))


class WindowsLauncherTests(TestCase):
    """The Windows launcher, pinned by its text.

    None of this is run on Windows -- the project is developed on a Mac --
    so these assert what gets written, not that Windows likes it.
    """

    def test_it_uses_pythonw_so_no_console_window_is_left_behind(self):
        script = build_app.windows_launcher_script(
            r"C:\VT", r"C:\VT\.venv\Scripts\pythonw.exe")
        self.assertIn("pythonw.exe", script)
        self.assertIn("-m counter.desktop", script)
        self.assertIn('start ""', script,
                      "without start the batch file blocks until the app exits")

    def test_batch_files_use_crlf(self):
        # A .bat with bare LF can fail in ways that are hard to see.
        script = build_app.windows_launcher_script(r"C:\VT", r"C:\py.exe")
        self.assertNotIn("\n", script.replace("\r\n", ""))

    def test_the_path_is_quoted_so_program_files_works(self):
        script = build_app.windows_launcher_script(
            r"C:\Program Files\VT", r"C:\Program Files\VT\pythonw.exe")
        self.assertIn(r'cd /d "C:\Program Files\VT"', script)

    def test_the_headless_launcher_turns_autostart_on(self):
        # Task Scheduler cannot set environment variables, so if the batch
        # file does not, the server starts and no camera does.
        script = build_app.windows_service_script(r"C:\VT", r"C:\VT\python.exe")
        self.assertIn("set COUNTER_AUTOSTART=1", script)
        self.assertIn("runserver", script)
        self.assertNotIn("pythonw", script,
                         "headless wants the real console interpreter")

    def test_the_shortcut_carries_the_icon(self):
        script = build_app.shortcut_script(
            r"C:\out\VT.lnk", r"C:\out\VT.bat", r"C:\VT\assets\AppIcon.ico",
            r"C:\VT")
        self.assertIn("WScript.Shell", script)
        self.assertIn(r'$link.IconLocation = "C:\VT\assets\AppIcon.ico"', script)
        self.assertIn("$link.Save()", script)

    def test_a_shortcut_without_an_icon_omits_the_line(self):
        script = build_app.shortcut_script(r"C:\a.lnk", r"C:\a.bat", None, r"C:\VT")
        self.assertNotIn("IconLocation", script)


class BuildTargetTests(TestCase):
    """Which launcher gets built, and with which icon."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = pathlib.Path(self.tmp) / "out"

    def build(self, **extra):
        out = StringIO()
        call_command("build_app", output=str(self.out), stdout=out,
                     stderr=StringIO(), **extra)
        return out.getvalue()

    def test_windows_writes_the_batch_files_not_a_bundle(self):
        self.build(target="windows", python=str(pathlib.Path(self.tmp) / "py.exe"))
        self.assertTrue((self.out / "VisitorsTracker.bat").exists())
        self.assertTrue((self.out / "VisitorsTracker-headless.bat").exists())
        self.assertTrue((self.out / "CreateShortcut.ps1").exists())
        self.assertFalse((self.out / "VisitorsTracker.app").exists())

    def test_windows_defaults_to_the_ico_not_the_icns(self):
        shipped = pathlib.Path(settings.BASE_DIR) / build_app.DEFAULT_ICONS["windows"]
        if not shipped.exists():
            self.skipTest("assets/AppIcon.ico is not in this checkout")
        self.build(target="windows", python=str(pathlib.Path(self.tmp) / "py.exe"))
        script = (self.out / "CreateShortcut.ps1").read_text()
        self.assertIn("AppIcon.ico", script)
        self.assertNotIn("AppIcon.icns", script,
                         "an .icns means nothing to Windows")

    def test_cross_generating_says_it_is_only_for_inspection(self):
        errors = StringIO()
        call_command("build_app", target="windows", output=str(self.out),
                     python=str(pathlib.Path(self.tmp) / "py.exe"),
                     stdout=StringIO(), stderr=errors)
        self.assertIn("inspection only", errors.getvalue())

    def test_a_missing_interpreter_is_only_fatal_for_this_machine(self):
        # Cross-generating cannot check a path that lives on another OS.
        self.build(target="windows", python=r"C:\nope\pythonw.exe")
        self.assertTrue((self.out / "VisitorsTracker.bat").exists())


class WindowsIconTests(TestCase):
    """Windows draws the .ico as given; it does not mask it."""

    def test_the_ico_keeps_its_own_rounded_corners(self):
        ico = pathlib.Path(settings.BASE_DIR) / "assets" / "AppIcon.ico"
        if not ico.exists():
            self.skipTest("assets/AppIcon.ico is not in this checkout")

        from PIL import Image

        image = Image.open(ico)
        image.size = (256, 256)
        corners = image.convert("RGBA")
        alpha = np.array(corners)[:, :, 3]
        # Unlike the macOS master, this one must NOT be full bleed, or it
        # shows as a square tile among rounded neighbours.
        self.assertLess(alpha[0, 0], 128, "the top-left corner should be clear")
        self.assertGreater(alpha[128, 128], 250, "the middle should be solid")


class ExportModelCommandTests(TestCase):
    """Getting the detector onto a machine that has just cloned this.

    The model is gitignored, so a fresh checkout has none and `auto`
    quietly falls back to HOG. The command that produces it used to be a
    bash script, which meant Windows had no way to get one at all.
    """

    def test_the_venv_interpreter_path_is_right_per_platform(self):
        with patch.object(export_model.sys, "platform", "win32"):
            self.assertEqual(
                export_model.venv_python(pathlib.Path("/w/venv")).name,
                "python.exe")
            self.assertIn(
                "Scripts",
                str(export_model.venv_python(pathlib.Path("/w/venv"))))
        with patch.object(export_model.sys, "platform", "darwin"):
            path = export_model.venv_python(pathlib.Path("/w/venv"))
            self.assertEqual(path.name, "python")
            self.assertIn("bin", str(path))

    def test_a_checkpoint_name_cannot_be_a_path_or_a_flag(self):
        for bad in ("../../etc/passwd", "a/b", r"a\b", "--upgrade", ""):
            with self.subTest(name=bad):
                with self.assertRaises(CommandError):
                    call_command("export_model", bad, stdout=StringIO())

    def test_a_missing_interpreter_is_refused_before_any_work(self):
        with self.assertRaises(CommandError):
            call_command("export_model", python="/nowhere/python",
                         stdout=StringIO())

    def test_the_build_environment_does_not_inherit_project_settings(self):
        # Ultralytics should not see DJANGO_SETTINGS_MODULE or COUNTER_*:
        # it is a separate program that happens to run from here.
        with patch.dict(os.environ, {"DJANGO_SETTINGS_MODULE": "config.settings",
                                     "COUNTER_AUTOSTART": "1",
                                     "PATH": "/usr/bin"}):
            env = export_model.Command.clean_env()
        self.assertNotIn("DJANGO_SETTINGS_MODULE", env)
        self.assertNotIn("COUNTER_AUTOSTART", env)
        self.assertIn("PATH", env, "it still needs a usable environment")


class CheckSetupCommandTests(TestCase):
    """One command that says whether a machine can run this.

    Written for the quiet failures. A missing model does not stop anything,
    it just counts worse; this has to be the thing that says so.
    """

    def run_check(self, **extra):
        out, err = StringIO(), StringIO()
        call_command("check_setup", stdout=out, stderr=err, **extra)
        return out.getvalue()

    def test_a_working_checkout_reports_the_real_detector(self):
        if not resolve_model_path(settings.COUNTER_ONNX_MODEL).exists():
            self.skipTest("no model exported in this checkout")
        output = self.run_check()
        self.assertIn("OnnxPersonDetector", output)
        self.assertIn("migrations", output)

    @override_settings(COUNTER_ONNX_MODEL="definitely_not_here.onnx")
    def test_a_missing_model_is_called_out_not_passed_over(self):
        output = self.run_check()
        self.assertIn("definitely_not_here.onnx", output)
        self.assertIn("export_model", output, "it must say how to fix it")
        self.assertIn("not YOLO", output,
                      "the HOG fallback is the quiet failure this is for")

    @override_settings(COUNTER_ONNX_MODEL="definitely_not_here.onnx")
    def test_it_checks_the_configured_model_not_the_default_one(self):
        # The bug this guards: resolving the default name would report a
        # healthy yolov8n.onnx while the counter loads something else.
        self.assertNotIn("yolov8n.onnx (", self.run_check())

    @override_settings(DEBUG=True)
    def test_debug_being_on_is_a_warning_scoped_to_this_command(self):
        # It reports how this process was started, not the server's
        # setting; saying otherwise is a false alarm about an app that
        # configures itself correctly.
        output = self.run_check()
        self.assertIn("DJANGO_DEBUG", output)
        self.assertIn("for this command", output)
        self.assertIn("desktop app sets it to 0", output)

    @override_settings(DEBUG=False)
    def test_debug_off_is_not_mentioned(self):
        self.assertNotIn("DJANGO_DEBUG", self.run_check())

    def test_it_names_the_cameras_and_whether_they_are_active(self):
        Camera.objects.create(name="Front door", source="0", is_active=True)
        output = self.run_check()
        self.assertIn("Front door", output)
        self.assertIn("active", output)

    def test_no_cameras_is_a_warning_with_a_way_forward(self):
        Camera.objects.all().delete()
        output = self.run_check()
        self.assertIn("none", output)
        self.assertIn("Adding a camera", output)

    def test_it_reports_the_platform_it_is_run_on(self):
        # The whole point is running it on the machine you are unsure about.
        self.assertIn(platform.system(), self.run_check())


# ---------------------------------------------------------------------------
# one process per camera
# ---------------------------------------------------------------------------

class CameraLeaseTests(TestCase):
    """Two processes counting one camera double the totals, silently.

    An RTSP stream serves both readers happily, so nothing errors; the
    numbers are just wrong. Measured before this existed: four processes
    over a clip containing 3 crossings recorded 12.
    """

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def test_a_free_camera_can_be_claimed(self):
        self.assertTrue(self.camera.claim_lease("process-a"))
        self.assertEqual(self.camera.lease_held_by(), "process-a")

    def test_a_second_process_is_refused(self):
        self.assertTrue(self.camera.claim_lease("process-a"))
        other = Camera.objects.get(pk=self.camera.pk)
        self.assertFalse(other.claim_lease("process-b"))
        self.assertEqual(other.lease_held_by(), "process-a")

    def test_the_holder_can_claim_again(self):
        # Restarting a worker in the same process must not deadlock on itself.
        self.assertTrue(self.camera.claim_lease("process-a"))
        self.assertTrue(self.camera.claim_lease("process-a"))

    def test_an_expired_lease_can_be_taken_over(self):
        # A process that is killed never releases; the camera must not be
        # stranded until somebody notices.
        past = timezone.now() - timedelta(seconds=1)
        Camera.objects.filter(pk=self.camera.pk).update(
            lease_owner="dead-process", lease_expires_at=past)
        fresh = Camera.objects.get(pk=self.camera.pk)
        self.assertEqual(fresh.lease_held_by(), "",
                         "an expired lease is held by nobody")
        self.assertTrue(fresh.claim_lease("process-b"))

    def test_renewing_keeps_it(self):
        self.camera.claim_lease("process-a", seconds=30)
        first = self.camera.lease_expires_at
        self.assertTrue(self.camera.renew_lease("process-a", seconds=90))
        self.assertGreater(self.camera.lease_expires_at, first)

    def test_renewing_a_lease_you_lost_fails(self):
        self.camera.claim_lease("process-a")
        Camera.objects.filter(pk=self.camera.pk).update(lease_owner="process-b")
        self.assertFalse(self.camera.renew_lease("process-a"))

    def test_releasing_frees_it_at_once(self):
        self.camera.claim_lease("process-a")
        self.assertTrue(self.camera.release_lease("process-a"))
        fresh = Camera.objects.get(pk=self.camera.pk)
        self.assertEqual(fresh.lease_held_by(), "")
        self.assertTrue(fresh.claim_lease("process-b"))

    def test_you_cannot_release_somebody_elses(self):
        self.camera.claim_lease("process-a")
        other = Camera.objects.get(pk=self.camera.pk)
        self.assertFalse(other.release_lease("process-b"))
        self.assertEqual(
            Camera.objects.get(pk=self.camera.pk).lease_held_by(), "process-a")


class WorkerRespectsTheLeaseTests(TestCase):
    """What a worker does when the camera is already taken."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")

    def test_a_worker_refuses_and_says_why(self):
        self.camera.claim_lease("some-other-process")
        worker = CounterWorker(self.camera)
        worker.run()                      # the thread body, inline
        self.assertIsNotNone(worker.error)
        self.assertIn("Another process", worker.error)
        self.assertIn("double", worker.error,
                      "the message has to say what goes wrong, not just refuse")
        self.assertEqual(worker.frames_read, 0)

    def test_each_process_gets_its_own_token(self):
        token = camera_module.process_token()
        self.assertEqual(token, camera_module.process_token(),
                         "it must be stable within a process")
        self.assertIn(str(os.getpid()), token)

    def test_the_token_is_not_just_host_and_pid(self):
        # pids are reused; a stale lease must not be renewable by whatever
        # process inherits that pid later.
        with patch.object(camera_module, "_PROCESS_TOKEN", None):
            first = camera_module.process_token()
        with patch.object(camera_module, "_PROCESS_TOKEN", None):
            second = camera_module.process_token()
        self.assertNotEqual(first, second)


class BackupCommandTests(TestCase):
    """A copy of the database that is safe to take while it is being written."""

    def setUp(self):
        import sqlite3

        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.into = self.tmp / "backups"

        # A real file on disk: the test database is in memory, and the whole
        # point of this command is copying a file safely.
        self.db = self.tmp / "db.sqlite3"
        con = sqlite3.connect(self.db)
        con.execute("create table counter_camera (id integer primary key, name text)")
        con.execute("insert into counter_camera (name) values ('Door')")
        con.commit()
        con.close()
        self.databases_setting = {
            "default": {"ENGINE": "django.db.backends.sqlite3",
                        "NAME": str(self.db)}}

    def backup(self, **extra):
        out = StringIO()
        with override_settings(DATABASES=self.databases_setting):
            call_command("backup_db", into=str(self.into), stdout=out, **extra)
        return out.getvalue()

    def test_the_copy_is_a_valid_database_with_the_rows_in_it(self):
        self.backup(no_compress=True)

        copies = list(self.into.glob("*.sqlite3"))
        self.assertEqual(len(copies), 1)

        import sqlite3

        con = sqlite3.connect(copies[0])
        try:
            self.assertEqual(
                con.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            rows = con.execute("select count(*) from counter_camera").fetchone()[0]
        finally:
            con.close()
        self.assertEqual(rows, 1)

    def test_an_in_memory_database_is_explained_not_just_refused(self):
        memory = {"default": {"ENGINE": "django.db.backends.sqlite3",
                              "NAME": ":memory:"}}
        with override_settings(DATABASES=memory):
            with self.assertRaises(CommandError) as caught:
                call_command("backup_db", into=str(self.into), stdout=StringIO())
        self.assertIn("in memory", str(caught.exception))

    def test_it_compresses_by_default(self):
        self.backup()
        self.assertEqual(len(list(self.into.glob("*.gz"))), 1)
        self.assertEqual(len(list(self.into.glob("*.sqlite3"))), 0,
                         "the uncompressed copy should not be left behind")

    def test_it_keeps_only_the_newest(self):
        self.into.mkdir(parents=True, exist_ok=True)
        for n in range(5):
            (self.into / f"db-2026010{n}-000000.sqlite3.gz").write_bytes(b"old")
        self.backup(keep=3)
        remaining = sorted(p.name for p in self.into.iterdir())
        self.assertEqual(len(remaining), 3)
        self.assertNotIn("db-20260100-000000.sqlite3.gz", remaining,
                         "the oldest should go first")

    def test_keep_zero_prunes_nothing(self):
        self.into.mkdir(parents=True, exist_ok=True)
        (self.into / "db-20260101-000000.sqlite3.gz").write_bytes(b"old")
        self.backup(keep=0)
        self.assertEqual(len(list(self.into.iterdir())), 2)

    def test_it_refuses_a_database_it_cannot_copy_this_way(self):
        engines = {"default": {"ENGINE": "django.db.backends.postgresql",
                               "NAME": "vt"}}
        with override_settings(DATABASES=engines):
            with self.assertRaises(CommandError) as caught:
                call_command("backup_db", into=str(self.into), stdout=StringIO())
        self.assertIn("pg_dump", str(caught.exception),
                      "it should say what to use instead")


class SpeedCheckTests(TestCase):
    """Whether a machine can keep up is a question it should answer itself.

    Written because the server this is going on is an EC2 instance whose
    size nobody could tell me, and detection is the whole cost.
    """

    class SlowDetector:
        def __init__(self, seconds):
            self.seconds = seconds

        def detect(self, frame):
            time_module.sleep(self.seconds)
            return []

    def check_with(self, seconds, detect_every=3):
        out = StringIO()
        with patch("counter.vision.build_detector",
                   return_value=self.SlowDetector(seconds)), \
                override_settings(COUNTER_DETECT_EVERY=detect_every):
            call_command("check_setup", benchmark=True, stdout=out,
                         stderr=StringIO())
        return out.getvalue()

    def test_a_fast_machine_passes(self):
        output = self.check_with(0.005)           # 5ms -> 600 fps of video
        self.assertIn("sustainable rate", output)
        self.assertNotIn("too slow", output)

    def test_a_machine_that_cannot_keep_up_fails_and_says_what_to_change(self):
        output = self.check_with(0.5)             # 500ms -> 6 fps of video
        self.assertIn("too slow", output)
        self.assertIn("COUNTER_DETECT_EVERY", output,
                      "it must name a lever, not just the verdict")

    def test_a_marginal_machine_is_warned_about(self):
        # ~0.25s per detection x 3 = about 12 fps of video: enough to run,
        # not enough to trust at a busy door.
        output = self.check_with(0.25)
        self.assertIn("between detections", output)

    def test_raising_detect_every_changes_the_verdict(self):
        # The same hardware, counted differently: the lever works.
        slow = self.check_with(0.25, detect_every=3)
        relaxed = self.check_with(0.25, detect_every=8)
        self.assertIn("between detections", slow)
        self.assertNotIn("between detections", relaxed)

    def test_the_benchmark_is_opt_in(self):
        out = StringIO()
        call_command("check_setup", stdout=out, stderr=StringIO())
        self.assertNotIn("sustainable rate", out.getvalue(),
                         "a routine check should not spend a second detecting")


class StaleLeaseTests(TestCase):
    """A killed process must not lock its own camera out on restart.

    SIGTERM and SIGKILL never run a finally block, so the lease outlives
    the process. Waiting it out means 45 seconds of a desktop app that
    will not count, right after somebody restarted it.
    """

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.host = socket.gethostname()

    def test_a_dead_local_process_is_recognised(self):
        # A pid that cannot be running: 0 is never a real process here.
        self.assertTrue(
            camera_module.lease_looks_stale(f"{self.host}:999999999:abcd1234"))

    def test_a_live_local_process_is_not(self):
        self.assertFalse(
            camera_module.lease_looks_stale(f"{self.host}:{os.getpid()}:abcd1234"))

    def test_another_machine_is_never_judged(self):
        # A pid on another host means nothing here, alive or not.
        self.assertFalse(
            camera_module.lease_looks_stale(f"some-other-box:1:abcd1234"))

    def test_a_token_in_an_unknown_shape_is_left_alone(self):
        for token in ("", None, "garbage", "a:b", "a:b:c:d"):
            with self.subTest(token=token):
                self.assertFalse(camera_module.lease_looks_stale(token))

    def test_a_worker_takes_over_from_a_dead_process(self):
        dead = f"{self.host}:999999999:abcd1234"
        self.camera.claim_lease(dead)
        worker = CounterWorker(self.camera)
        self.assertTrue(worker._claim())
        self.assertEqual(
            Camera.objects.get(pk=self.camera.pk).lease_held_by(),
            worker.lease_owner)

    def test_a_worker_does_not_take_over_from_a_live_one(self):
        self.camera.claim_lease(f"{self.host}:{os.getpid()}:other-worker")
        worker = CounterWorker(self.camera)
        self.assertFalse(worker._claim())
        self.assertIn("Another process", worker.error)

    def test_taking_over_is_conditional_on_the_holder(self):
        # If the lease changed hands between the check and the write, the
        # takeover must lose rather than stomp.
        self.camera.claim_lease("first-holder")
        Camera.objects.filter(pk=self.camera.pk).update(lease_owner="second-holder")
        self.assertFalse(self.camera.take_lease_from("first-holder", "me"))
        self.assertEqual(
            Camera.objects.get(pk=self.camera.pk).lease_owner, "second-holder")


class DesktopProductionDefaultsTests(TestCase):
    """A double-clicked app has no shell, so it configures itself."""

    def defaults(self, preset=None):
        env = dict(preset or {})
        with patch.dict(os.environ, env, clear=True):
            desktop.production_defaults()
            return dict(os.environ)

    def test_debug_is_off_by_default(self):
        self.assertEqual(self.defaults()["DJANGO_DEBUG"], "0")

    def test_it_points_at_a_key_file_rather_than_inventing_one_each_run(self):
        env = self.defaults()
        self.assertTrue(env["DJANGO_SECRET_KEY_FILE"].endswith(".secret_key"))

    def test_allowed_hosts_is_local_only(self):
        hosts = self.defaults()["DJANGO_ALLOWED_HOSTS"]
        self.assertIn("127.0.0.1", hosts)
        self.assertNotIn("*", hosts)

    def test_an_explicit_setting_still_wins(self):
        env = self.defaults({"DJANGO_DEBUG": "1",
                             "DJANGO_ALLOWED_HOSTS": "example.com"})
        self.assertEqual(env["DJANGO_DEBUG"], "1")
        self.assertEqual(env["DJANGO_ALLOWED_HOSTS"], "example.com")


class SecretKeyFileTests(TestCase):
    """The key has to survive a restart, or everyone is logged out."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp / ".secret_key"

    def test_it_is_created_once_and_reused(self):
        first = secretkey.read_or_create(self.path)
        self.assertEqual(secretkey.read_or_create(self.path), first)
        self.assertGreaterEqual(len(first), secretkey.MINIMUM_LENGTH)

    def test_it_is_not_world_readable(self):
        secretkey.read_or_create(self.path)
        mode = self.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, f"mode was {oct(mode)}")

    def test_a_truncated_file_is_replaced_rather_than_trusted(self):
        # A short key weakens every signature. Refusing to start on it was
        # the first attempt and it just made the app unlaunchable.
        self.path.write_text("short")
        key = secretkey.read_or_create(self.path)
        self.assertGreaterEqual(len(key), secretkey.MINIMUM_LENGTH)
        self.assertEqual(self.path.read_text().strip(), key)

    def test_an_empty_file_is_replaced(self):
        self.path.write_text("")
        self.assertGreaterEqual(
            len(secretkey.read_or_create(self.path)), secretkey.MINIMUM_LENGTH)

    def test_it_creates_the_directory_if_needed(self):
        nested = self.tmp / "a" / "b" / ".secret_key"
        self.assertGreaterEqual(
            len(secretkey.read_or_create(nested)), secretkey.MINIMUM_LENGTH)


class DataDirectoryTests(TestCase):
    """Frozen, the code lives somewhere that is deleted when the app exits.

    PyInstaller unpacks its bundle into a temporary directory and removes
    it on exit, so a database written beside the code would be gone after
    every run, taking the history and the secret key with it.
    """

    def test_a_checkout_keeps_everything_together(self):
        with patch.object(paths, "is_frozen", return_value=False), \
                patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.resolve_data_dir("/srv/vt"),
                             pathlib.Path("/srv/vt"))

    def test_frozen_moves_data_out_of_the_bundle(self):
        with patch.object(paths, "is_frozen", return_value=True), \
                patch.dict(os.environ, {}, clear=True):
            resolved = paths.resolve_data_dir("/tmp/_MEI12345")
        self.assertNotIn("_MEI", str(resolved),
                         "that directory is deleted when the app exits")

    def test_an_explicit_directory_wins_everywhere(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                with patch.object(paths, "is_frozen", return_value=frozen), \
                        patch.dict(os.environ, {"COUNTER_DATA_DIR": "/data/vt"}):
                    self.assertEqual(paths.resolve_data_dir("/srv/vt"),
                                     pathlib.Path("/data/vt"))

    def test_windows_uses_localappdata(self):
        with patch.object(paths.sys, "platform", "win32"), \
                patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\me\AppData\Local"}):
            resolved = paths.default_data_dir()
        self.assertIn("AppData", str(resolved))
        self.assertTrue(str(resolved).endswith("VisitorsTracker"))

    def test_macos_uses_application_support(self):
        with patch.object(paths.sys, "platform", "darwin"):
            self.assertIn("Application Support",
                          str(paths.default_data_dir()))

    def test_linux_uses_xdg(self):
        with patch.object(paths.sys, "platform", "linux"), \
                patch.dict(os.environ, {"XDG_DATA_HOME": "/home/me/.local/share"}):
            self.assertEqual(str(paths.default_data_dir()),
                             "/home/me/.local/share/VisitorsTracker")

    def test_the_database_is_configured_under_the_data_directory(self):
        # The test runner swaps NAME for an in-memory database, so the live
        # setting cannot show this. Check the wiring instead.
        configured = (pathlib.Path(settings.BASE_DIR) / "config" / "settings.py"
                      ).read_text()
        self.assertIn('"NAME": DATA_DIR / "db.sqlite3"', configured)
        self.assertTrue(pathlib.Path(settings.DATA_DIR).is_absolute())


class SpecFileTests(TestCase):
    """The spec has to carry what Django finds by scanning, not importing."""

    def setUp(self):
        self.spec = pathlib.Path(settings.BASE_DIR) / "VisitorsTracker.spec"
        if not self.spec.exists():
            self.skipTest("no spec in this checkout")
        self.text = self.spec.read_text()

    def test_it_collects_what_django_discovers_by_filesystem(self):
        # Nothing imports these, so nothing traces them either. The
        # submodule collection covers the whole counter package, which
        # includes management commands and migrations.
        self.assertIn("STRING_REFERENCED", self.text)
        for package in ("config", "counter", "corsheaders", "rest_framework"):
            self.assertIn(f'"{package}"', self.text)
        self.assertIn("counter/templates", self.text)
        self.assertIn("counter/migrations", self.text)

    def test_it_bundles_the_detector(self):
        # Otherwise the exe runs on HOG and counts badly, quietly.
        self.assertIn("ml_models", self.text)
        self.assertIn(".onnx", self.text)

    def test_it_leaves_out_the_export_only_dependencies(self):
        # torch arrives with ultralytics for export_model and would add
        # gigabytes to something that never calls it.
        for heavy in ("torch", "ultralytics"):
            self.assertIn(heavy, self.text.split("excludes=")[1][:200])

    def test_it_builds_a_windowed_app_with_the_icon(self):
        self.assertIn("console=False", self.text)
        self.assertIn("AppIcon.ico", self.text)


class FrozenEntryPointTests(TestCase):
    """PyInstaller runs its entry script as __main__, not as a module.

    counter/desktop.py uses relative imports, so pointing the build at it
    produced "attempted relative import with no known parent package" after
    the app had already started, migrated and opened its port.
    """

    def test_there_is_an_entry_script_that_only_imports(self):
        entry = pathlib.Path(settings.BASE_DIR) / "app_main.py"
        self.assertTrue(entry.exists())
        text = entry.read_text()
        self.assertIn("from counter.desktop import main", text)
        # Code only -- the docstring explains relative imports, so a plain
        # substring check matches its own explanation.
        code = [line for line in text.splitlines()
                if line.startswith(("from ", "import "))]
        self.assertFalse([line for line in code if line.startswith("from .")],
                         "an entry script runs as __main__ and has no package")

    def test_the_spec_builds_the_entry_script_not_the_module(self):
        spec = pathlib.Path(settings.BASE_DIR) / "VisitorsTracker.spec"
        if not spec.exists():
            self.skipTest("no spec in this checkout")
        text = spec.read_text()
        self.assertIn('["app_main.py"]', text)
        self.assertNotIn('["counter/desktop.py"]', text)

    def test_the_spec_puts_the_project_on_sys_path_before_collecting(self):
        # collect_submodules imports to walk, and runs while the spec is
        # evaluated -- before pathex applies. Without this it returns
        # nothing, the build succeeds, and the app dies on startup.
        spec = pathlib.Path(settings.BASE_DIR) / "VisitorsTracker.spec"
        if not spec.exists():
            self.skipTest("no spec in this checkout")
        text = spec.read_text()
        self.assertIn("sys.path.insert", text)
        self.assertLess(text.index("sys.path.insert"),
                        text.index("STRING_REFERENCED"),
                        "the path has to be set before anything is collected")

    def test_the_spec_refuses_to_build_an_empty_bundle(self):
        spec = pathlib.Path(settings.BASE_DIR) / "VisitorsTracker.spec"
        if not spec.exists():
            self.skipTest("no spec in this checkout")
        self.assertIn("raise SystemExit", spec.read_text())


class CrashLoggingTests(TestCase):
    """A windowed build has no stderr, so a crash has to be written down."""

    def test_uncaught_exceptions_are_logged_before_the_process_dies(self):
        source = pathlib.Path(settings.BASE_DIR) / "counter" / "desktop.py"
        text = source.read_text()
        self.assertIn("sys.excepthook", text)
        self.assertIn("threading.excepthook", text,
                      "a camera worker dying must not be silent either")

    def test_it_logs_where_the_data_went(self):
        # The first frozen run left an empty log file and no clue; the
        # startup line at least says which directory to look in.
        text = (pathlib.Path(settings.BASE_DIR) / "counter" / "desktop.py").read_text()
        self.assertIn("VisitorsTracker starting, data directory", text)


class FrozenManagementCommandTests(TestCase):
    """A frozen install has to be able to make its first administrator.

    The exe opened the window and nothing else, so createsuperuser was
    unreachable: the dashboard's admin link led to a login nobody could
    pass and the staff-only controls were closed for good. The Windows
    build came up with zero users and no way to add one.
    """

    def entry_source(self):
        return (pathlib.Path(settings.BASE_DIR) / "app_main.py").read_text()

    def test_arguments_run_a_command_instead_of_opening_a_window(self):
        source = self.entry_source()
        self.assertIn("execute_from_command_line", source)
        self.assertIn("len(sys.argv) > 1", source)

    def test_no_arguments_still_opens_the_window(self):
        self.assertIn("run_window()", self.entry_source())

    def test_it_gets_a_console_on_windows_one_way_or_another(self):
        # A windowed build has sys.stdin = None and input() raises
        # "lost sys.stdin". Attaching to the parent is not enough on its
        # own: PyInstaller's onefile bootloader runs the program as a
        # child, so ATTACH_PARENT_PROCESS reaches the bootloader, which is
        # windowed and has no console to lend.
        source = self.entry_source()
        self.assertIn("AttachConsole", source)
        self.assertIn("AllocConsole", source,
                      "attaching alone fails in a onefile build")
        self.assertLess(source.index("AttachConsole"), source.index("AllocConsole"),
                        "borrow the caller's console before opening a new one")

    def test_an_allocated_console_is_held_open_to_be_read(self):
        # It closes with the process, taking the result with it.
        source = self.entry_source()
        self.assertIn("Press Enter to close", source)
        self.assertIn('console == "allocated"', source)

    def test_the_command_still_runs_if_no_console_can_be_had(self):
        # --noinput commands work without one; only prompts need it.
        source = self.entry_source()
        attach = source.index("console = attach_console()")
        execute = source.index("execute_from_command_line(sys.argv)")
        self.assertLess(attach, execute)
        self.assertNotIn("if not console:\n            return", source)

    def test_a_command_actually_runs_through_the_entry_script(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, str(pathlib.Path(settings.BASE_DIR) / "app_main.py"),
             "check_setup"],
            capture_output=True, text=True, timeout=300,
            cwd=str(settings.BASE_DIR))
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertIn("Dependencies", result.stdout)

    def test_the_settings_are_configured_before_the_command_runs(self):
        # Otherwise the command fails with ImproperlyConfigured instead of
        # doing its job.
        source = self.entry_source()
        self.assertLess(source.index("production_defaults()"),
                        source.index("execute_from_command_line"))


class AdminStaticFilesTests(TestCase):
    """The admin needs its CSS, and DEBUG=0 stops Django serving it.

    The desktop app turns DEBUG off deliberately, which left the admin
    login page completely unstyled: nothing was serving /static/ at all,
    because there is no web server in front of a double-clicked app.
    """

    def test_static_is_served_when_the_app_is_the_only_server(self):
        from django.conf import settings

        if not settings.COUNTER_SERVE_STATIC:
            self.skipTest("static serving turned off in this environment")
        response = self.client.get("/static/admin/css/base.css")
        self.assertEqual(response.status_code, 200)

    def test_the_login_page_asks_for_css_that_resolves(self):
        page = self.client.get("/admin/login/").content.decode()
        hrefs = re.findall(r'href="([^"]+\.css)"', page)
        self.assertTrue(hrefs, "the admin login page should link stylesheets")
        for href in hrefs:
            with self.subTest(href=href):
                self.assertEqual(self.client.get(href).status_code, 200,
                                 f"{href} does not resolve")

    @override_settings(COUNTER_SERVE_STATIC=False)
    def test_it_can_be_turned_off_for_a_real_web_server(self):
        # nginx serves /static/ in the server deployment; Django need not.
        from importlib import reload

        import config.urls

        reload(config.urls)
        try:
            patterns = [str(p.pattern) for p in config.urls.urlpatterns]
            self.assertFalse([p for p in patterns if "static" in p])
        finally:
            reload(config.urls)

    def test_static_root_is_under_the_data_directory(self):
        from django.conf import settings

        self.assertEqual(pathlib.Path(settings.STATIC_ROOT).parent,
                         pathlib.Path(settings.DATA_DIR))


class SpecCarriesAdminStaticTests(TestCase):
    """A frozen build reads static files from the app directories."""

    def test_the_spec_collects_django_admin_data(self):
        spec = pathlib.Path(settings.BASE_DIR) / "VisitorsTracker.spec"
        if not spec.exists():
            self.skipTest("no spec in this checkout")
        text = spec.read_text()
        self.assertIn('collect_data_files("django.contrib.admin")', text)


class AdminCsvExportTests(TestCase):
    """Exporting an event's figures from the admin, by date."""

    def setUp(self):
        self.staff = User.objects.create_superuser("boss", "b@example.com", "pw")
        self.client.force_login(self.staff)
        self.camera = Camera.objects.create(name="Door", source="0")
        self.event = Event.objects.create(
            name="Trade show",
            starts_at=timezone.now() - timedelta(days=2),
            ends_at=timezone.now() + timedelta(days=1))
        for i in range(3):
            Crossing.objects.create(camera=self.camera, event=self.event,
                                    direction=DIRECTION_IN, track_id=i,
                                    occurred_at=timezone.now())

    def post(self, **extra):
        data = {"action": "export_csv",
                "_selected_action": [str(self.event.pk)], **extra}
        return self.client.post("/admin/counter/event/", data)

    def test_choosing_the_action_asks_for_dates_first(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="start"')
        self.assertContains(response, "Trade show")

    def test_it_downloads_a_csv_naming_the_event(self):
        response = self.post(apply="1")
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("Trade-show", response["Content-Disposition"])

    def test_every_row_names_its_event(self):
        # Several events in one file would otherwise be unreadable.
        rows = list(csv.reader(
            self.post(apply="1").content.decode().splitlines()))
        self.assertEqual(rows[0][0], "event")
        self.assertTrue(all(r[0] == "Trade show" for r in rows[1:]))

    def test_the_counts_are_there(self):
        rows = list(csv.reader(
            self.post(apply="1").content.decode().splitlines()))
        header, body = rows[0], rows[1:]
        total_in = sum(int(r[header.index("in")]) for r in body)
        self.assertEqual(total_in, 3)

    def test_an_explicit_window_narrows_it_and_is_named(self):
        today = timezone.localdate()
        response = self.post(apply="1", start=today.isoformat(),
                             end=today.isoformat())
        self.assertIn(today.isoformat(), response["Content-Disposition"])
        rows = list(csv.reader(response.content.decode().splitlines()))
        self.assertEqual(len(rows) - 1, 1, "one day asked for, one day back")

    def test_blank_dates_mean_the_events_own_run(self):
        rows = list(csv.reader(
            self.post(apply="1").content.decode().splitlines()))
        # The event spans three calendar days.
        self.assertGreaterEqual(len(rows) - 1, 3)

    def test_backwards_dates_are_refused_with_a_message(self):
        response = self.post(apply="1", start="2026-12-01", end="2026-01-01")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "after the end date")

    def test_a_bad_date_does_not_500(self):
        response = self.post(apply="1", start="not-a-date")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "errornote")

    def test_several_events_come_back_in_one_file(self):
        other = Event.objects.create(
            name="Open day", starts_at=timezone.now() - timedelta(days=1),
            ends_at=timezone.now())
        response = self.client.post("/admin/counter/event/", {
            "action": "export_csv", "apply": "1",
            "_selected_action": [str(self.event.pk), str(other.pk)]})
        names = {r[0] for r in csv.reader(
            response.content.decode().splitlines())} - {"event"}
        self.assertEqual(names, {"Trade show", "Open day"})
        self.assertIn("2-events", response["Content-Disposition"])

    def test_an_anonymous_visitor_cannot_export(self):
        self.client.logout()
        response = self.post(apply="1")
        self.assertNotEqual(response.get("Content-Type"), "text/csv")


class ExportFilenameTests(TestCase):
    """Event names are free text; filenames are not."""

    def test_punctuation_and_spaces_are_made_safe(self):
        self.assertEqual(exports.safe_filename("visitors", "Doors open 9-5 (east)"),
                         "visitors_Doors-open-9-5-east")

    def test_a_slash_cannot_survive(self):
        self.assertNotIn("/", exports.safe_filename("visitors", "Eid/Fitr"))

    def test_dot_runs_are_collapsed(self):
        name = exports.safe_filename("visitors", "../../etc/passwd")
        self.assertNotIn("..", name)

    def test_an_empty_name_still_gives_a_filename(self):
        self.assertEqual(exports.safe_filename(""), "export")

    def test_it_stays_a_sane_length(self):
        self.assertLessEqual(len(exports.safe_filename("x" * 400)), 120)


class ScriptedRunsMustNotHangTests(TestCase):
    """The pause that keeps a console readable must not block a script.

    An allocated console closes with the process, so an interactive
    command pauses before exiting. Run unattended, that pause waits for a
    key nobody will press: `createsuperuser --noinput` hung forever the
    first time it ran from a script.
    """

    def entry(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "app_main_under_test",
            str(pathlib.Path(settings.BASE_DIR) / "app_main.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_noinput_never_pauses(self):
        entry = self.entry()
        for flag in ("--noinput", "--no-input"):
            with self.subTest(flag=flag):
                self.assertFalse(
                    entry.wants_a_reader(["exe", "createsuperuser", flag]))

    def test_an_interactive_command_still_pauses(self):
        self.assertTrue(self.entry().wants_a_reader(["exe", "createsuperuser"]))

    def test_an_environment_variable_can_turn_it_off(self):
        entry = self.entry()
        with patch.dict(os.environ, {"VISITORSTRACKER_NO_PAUSE": "1"}):
            self.assertFalse(entry.wants_a_reader(["exe", "migrate"]))

    def test_the_pause_is_tied_to_an_allocated_console(self):
        # Borrowing the caller's console means the output stays on screen
        # after exit; there is nothing to hold open.
        source = (pathlib.Path(settings.BASE_DIR) / "app_main.py").read_text()
        self.assertIn('console == "allocated" and wants_a_reader', source)


class FirstRunAdminTests(TestCase):
    """A fresh install has to be usable without a command line.

    A camera can only be added through the Django admin -- there is no
    create endpoint -- so an install with no accounts cannot be
    configured at all. Somebody who double-clicks the app would see an
    empty dashboard and no way forward.
    """

    def setUp(self):
        self.data_dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.data_dir, True)
        User.objects.all().delete()

    def test_it_creates_an_administrator_when_there_are_none(self):
        made = firstrun.ensure_admin(self.data_dir)
        self.assertIsNotNone(made)
        user = User.objects.get(username="admin")
        self.assertTrue(user.is_superuser and user.is_staff)

    def test_the_password_works(self):
        firstrun.ensure_admin(self.data_dir)
        text = (self.data_dir / firstrun.PASSWORD_FILE).read_text()
        password = [l.split(":", 1)[1].strip() for l in text.splitlines()
                    if l.startswith("password:")][0]
        self.assertTrue(self.client.login(username="admin", password=password))

    def test_it_leaves_an_existing_install_alone(self):
        User.objects.create_superuser("boss", "b@example.com", "pw")
        self.assertIsNone(firstrun.ensure_admin(self.data_dir))
        self.assertFalse((self.data_dir / firstrun.PASSWORD_FILE).exists())

    def test_it_does_not_resurrect_a_deleted_default(self):
        # Somebody who removes the default account on purpose should not
        # find it back after a restart.
        firstrun.ensure_admin(self.data_dir)
        User.objects.create_superuser("real", "r@example.com", "pw")
        User.objects.filter(username="admin").delete()
        firstrun.ensure_admin(self.data_dir)
        self.assertFalse(User.objects.filter(username="admin").exists())

    def test_the_password_is_different_every_time(self):
        # A fixed default on every install is the oldest hole there is,
        # and this app starts a camera.
        seen = {firstrun.generate_password() for _ in range(50)}
        self.assertEqual(len(seen), 50)

    def test_the_password_avoids_lookalike_characters(self):
        # It gets copied off a screen by hand.
        joined = "".join(firstrun.generate_password() for _ in range(40))
        self.assertFalse(set("l1IO0") & set(joined))

    def test_the_file_is_not_world_readable(self):
        if sys.platform.startswith("win"):
            self.skipTest("Windows does not model permissions this way")
        firstrun.ensure_admin(self.data_dir)
        mode = (self.data_dir / firstrun.PASSWORD_FILE).stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)


class FirstRunBannerTests(TestCase):
    """The generated password has to be findable, or it helps nobody."""

    def setUp(self):
        self.data_dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.data_dir, True)

    def test_the_dashboard_shows_the_credentials_while_the_file_exists(self):
        with override_settings(DATA_DIR=self.data_dir):
            User.objects.all().delete()
            firstrun.ensure_admin(self.data_dir)
            page = self.client.get("/").content.decode()
        self.assertIn("First run", page)
        self.assertIn("admin", page)

    def test_it_disappears_once_the_file_is_deleted(self):
        with override_settings(DATA_DIR=self.data_dir):
            User.objects.all().delete()
            firstrun.ensure_admin(self.data_dir)
            (self.data_dir / firstrun.PASSWORD_FILE).unlink()
            page = self.client.get("/").content.decode()
        self.assertNotIn("First run", page)

    def test_an_established_install_never_sees_it(self):
        with override_settings(DATA_DIR=self.data_dir):
            page = self.client.get("/").content.decode()
        self.assertNotIn("First run", page)


class DashboardDateFilterTests(TestCase):
    """From/to pickers on the visitors-per-date panel."""

    def page(self):
        return self.client.get("/").content.decode()

    def test_the_controls_are_on_the_page(self):
        page = self.page()
        for marker in ('id="d-from"', 'id="d-to"', 'id="d-clear"',
                       'type="date"'):
            self.assertIn(marker, page)

    def test_clear_is_visible_from_the_start(self):
        # It used to be hidden until a date was picked, which made it a
        # control nobody could find before they needed it.
        page = self.page()
        self.assertIn('id="d-clear" disabled', page)
        self.assertNotIn('id="d-clear" hidden', page)

    def test_the_inputs_cannot_be_set_past_today(self):
        # Nothing is ever recorded in the future.
        self.assertIn("dFrom.max = todayIso", self.page())

    def test_one_date_alone_does_not_take_over(self):
        # "From the 3rd" could mean to today or to the end of the data;
        # guessing either way draws a chart nobody asked for.
        self.assertIn("from && to ? { from, to } : null", self.page())

    def test_the_effective_window_is_reflected_back(self):
        # The server swaps a reversed range and caps a long one, so the
        # inputs would otherwise show something that was never queried.
        page = self.page()
        self.assertIn("if (data.start) dFrom.value = data.start", page)
        self.assertIn("if (data.end) dTo.value = data.end", page)


class DailyRangeApiTests(TestCase):
    """What the pickers send, and what comes back."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.event = Event.objects.create(
            name="Run", starts_at=timezone.now() - timedelta(days=10),
            ends_at=timezone.now() + timedelta(days=1))

    def get(self, query):
        return self.client.get(f"/api/v1/daily/?{query}").json()

    def test_an_explicit_window_is_honoured(self):
        today = timezone.localdate()
        start = today - timedelta(days=3)
        data = self.get(f"start={start.isoformat()}&end={today.isoformat()}")
        self.assertEqual(data["start"], start.isoformat())
        self.assertEqual(data["end"], today.isoformat())
        self.assertEqual(len(data["days"]), 4)

    def test_a_reversed_window_is_swapped_not_refused(self):
        today = timezone.localdate()
        start = today - timedelta(days=3)
        data = self.get(f"start={today.isoformat()}&end={start.isoformat()}")
        self.assertEqual(data["start"], start.isoformat())
        self.assertEqual(data["end"], today.isoformat())

    def test_a_single_day_is_allowed(self):
        today = timezone.localdate().isoformat()
        data = self.get(f"start={today}&end={today}")
        self.assertEqual(len(data["days"]), 1)

    def test_an_absurd_window_is_capped_and_says_so(self):
        # The response carries the window actually used, which is what the
        # inputs are set from.
        from counter.reports import MAX_DAYS

        data = self.get("start=2000-01-01&end=2026-12-31")
        span = (date_cls.fromisoformat(data["end"])
                - date_cls.fromisoformat(data["start"])).days + 1
        self.assertLessEqual(span, MAX_DAYS)
        self.assertEqual(span, MAX_DAYS, "a huge window should hit the cap")

    def test_the_csv_honours_the_same_window(self):
        today = timezone.localdate()
        start = today - timedelta(days=2)
        response = self.client.get(
            f"/api/v1/daily/?start={start.isoformat()}"
            f"&end={today.isoformat()}&format=csv")
        rows = response.content.decode().splitlines()
        self.assertEqual(len(rows) - 1, 3)
        self.assertIn(start.isoformat(), response["Content-Disposition"])


class DesktopSaveFileTests(TestCase):
    """The window has no address bar, so a download must not navigate.

    Clicking Download CSV used to set window.location, which in a
    chrome-less webview left the operator looking at raw CSV with no Back
    button and no way to save it.
    """

    class FakeWindow:
        def __init__(self, answer):
            self.answer = answer
            self.asked = []

        def create_file_dialog(self, kind, save_filename=None):
            self.asked.append(save_filename)
            return self.answer

    def save(self, answer, filename="visitors.csv", content="a,b\n1,2\n"):
        from counter import desktop

        window = self.FakeWindow(answer)
        fake_webview = type("W", (), {"windows": [window], "SAVE_DIALOG": 30})
        with patch.dict(sys.modules, {"webview": fake_webview}):
            return desktop.DesktopApi().save_file(filename, content), window

    def test_it_writes_where_the_dialog_says(self):
        target = pathlib.Path(tempfile.mkdtemp()) / "out.csv"
        self.addCleanup(shutil.rmtree, target.parent, True)
        result, window = self.save(str(target))
        self.assertTrue(result["saved"])
        self.assertEqual(target.read_text(), "a,b\n1,2\n")
        self.assertEqual(window.asked, ["visitors.csv"])

    def test_a_sequence_from_an_older_pywebview_is_handled(self):
        target = pathlib.Path(tempfile.mkdtemp()) / "out.csv"
        self.addCleanup(shutil.rmtree, target.parent, True)
        result, _ = self.save((str(target),))
        self.assertTrue(result["saved"])
        self.assertTrue(target.exists())

    def test_cancelling_is_not_an_error(self):
        result, _ = self.save(None)
        self.assertFalse(result["saved"])
        self.assertEqual(result["reason"], "cancelled")

    def test_an_unwritable_destination_is_reported_not_raised(self):
        result, _ = self.save("/nowhere/that/exists/out.csv")
        self.assertFalse(result["saved"])
        self.assertNotEqual(result["reason"], "cancelled")

    def test_no_window_is_survivable(self):
        from counter import desktop

        fake_webview = type("W", (), {"windows": [], "SAVE_DIALOG": 30})
        with patch.dict(sys.modules, {"webview": fake_webview}):
            result = desktop.DesktopApi().save_file("x.csv", "a")
        self.assertFalse(result["saved"])

    def test_the_window_is_given_the_api(self):
        source = (pathlib.Path(settings.BASE_DIR) / "counter" / "desktop.py").read_text()
        self.assertIn("js_api=DesktopApi()", source)


class CsvDownloadDoesNotNavigateTests(TestCase):
    """Neither path may leave the page."""

    def page(self):
        return self.client.get("/").content.decode()

    def test_nothing_sets_window_location_for_a_csv(self):
        page = self.page()
        self.assertNotIn('window.location = dailyUrl', page)
        self.assertNotIn('window.location = hourlyUrl', page)

    def test_the_desktop_branch_asks_the_application_to_save(self):
        self.assertIn("window.pywebview.api.save_file", self.page())

    def test_the_browser_branch_uses_a_download_link(self):
        page = self.page()
        self.assertIn('link.download = fallbackName', page)

    def test_the_filename_comes_from_the_server_when_it_gives_one(self):
        # The server names the file after the event and the window; the
        # fallback is only for when it does not.
        self.assertIn("Content-Disposition", self.page())


class CsvNamesItsEventTests(TestCase):
    """A file off the dashboard should still say what it covers.

    Opened a week later in a spreadsheet, "16 in on the 9th" means
    nothing without knowing which event it belonged to.
    """

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.event = Event.objects.create(
            name="Trade show", starts_at=timezone.now() - timedelta(days=1),
            ends_at=timezone.now() + timedelta(days=1))
        Crossing.objects.create(camera=self.camera, event=self.event,
                                direction=DIRECTION_IN, track_id=1,
                                occurred_at=timezone.now())

    def rows(self, path):
        return list(csv.reader(
            self.client.get(path).content.decode().splitlines()))

    def test_a_filtered_daily_export_names_the_event(self):
        rows = self.rows(f"/api/v1/daily/?days=3&event={self.event.pk}&format=csv")
        self.assertEqual(rows[0][0], "event")
        self.assertTrue(all(r[0] == "Trade show" for r in rows[1:]))

    def test_an_unfiltered_export_says_so_rather_than_leaving_it_blank(self):
        # An empty cell reads as missing data; this reads as what it is.
        rows = self.rows("/api/v1/daily/?days=3&format=csv")
        self.assertTrue(all(r[0] == "All events" for r in rows[1:]))
        self.assertNotIn("", {r[0] for r in rows[1:]})

    def test_the_hourly_export_names_it_too(self):
        rows = self.rows(f"/api/v1/hourly/?days=3&event={self.event.pk}&format=csv")
        self.assertEqual(rows[0][0], "event")
        self.assertTrue(all(r[0] == "Trade show" for r in rows[1:]))

    def test_the_admin_export_and_the_api_agree_on_the_first_column(self):
        # Two exports of the same thing should not have different shapes.
        from counter.exports import DAILY_COLUMNS

        api = self.rows(f"/api/v1/daily/?days=3&event={self.event.pk}&format=csv")
        self.assertEqual(api[0], DAILY_COLUMNS)


class DashboardEventFilterTests(TestCase):
    """Choosing an event on the dashboard, and in what it exports.

    The CSV carried an event column but the dashboard never sent one, so
    every download said "All events" no matter what you were looking at.
    """

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.show = Event.objects.create(
            name="Trade show", starts_at=timezone.now() - timedelta(days=2),
            ends_at=timezone.now() + timedelta(days=1))
        self.open_day = Event.objects.create(
            name="Open day", starts_at=timezone.now() - timedelta(days=5),
            ends_at=timezone.now() - timedelta(days=4))
        Crossing.objects.create(camera=self.camera, event=self.show,
                                direction=DIRECTION_IN, track_id=1,
                                occurred_at=timezone.now())

    def page(self):
        return self.client.get("/").content.decode()

    def test_both_panels_offer_an_event_picker(self):
        page = self.page()
        self.assertIn('id="d-event"', page)
        self.assertIn('id="h-event"', page)

    def test_every_event_is_listed_with_an_all_option(self):
        page = self.page()
        self.assertIn(">All events<", page)
        self.assertIn("Trade show", page)
        self.assertIn("Open day", page)

    def test_the_newest_event_comes_first(self):
        # The one somebody wants is nearly always the most recent.
        page = self.page()
        self.assertLess(page.index("Trade show"), page.index("Open day"))

    def test_the_url_builders_carry_the_selection(self):
        page = self.page()
        self.assertIn('eventParam("d-event")', page)
        self.assertIn('eventParam("h-event")', page)

    def test_a_filtered_csv_names_that_event_on_every_row(self):
        rows = list(csv.reader(self.client.get(
            f"/api/v1/daily/?days=7&event={self.show.pk}&format=csv"
        ).content.decode().splitlines()))
        self.assertTrue(all(r[0] == "Trade show" for r in rows[1:]))

    def test_the_counts_follow_the_event_too(self):
        # Not just the label: the other event has no crossings.
        mine = self.client.get(
            f"/api/v1/daily/?days=7&event={self.show.pk}").json()
        theirs = self.client.get(
            f"/api/v1/daily/?days=7&event={self.open_day.pk}").json()
        self.assertEqual(mine["summary"]["in"], 1)
        self.assertEqual(theirs["summary"]["in"], 0)

    def test_the_hourly_export_follows_it_as_well(self):
        rows = list(csv.reader(self.client.get(
            f"/api/v1/hourly/?days=7&event={self.show.pk}&format=csv"
        ).content.decode().splitlines()))
        self.assertTrue(all(r[0] == "Trade show" for r in rows[1:]))

    def test_an_unknown_event_is_refused_not_ignored(self):
        # Silently reporting on everything would be a wrong answer that
        # looks like a right one.
        response = self.client.get("/api/v1/daily/?days=7&event=99999")
        self.assertIn(response.status_code, (400, 404))


class LiveCamerasOnTheirOwnPageTests(TestCase):
    """The streams moved off the dashboard.

    Every camera on the dashboard streamed MJPEG continuously, whether or
    not anybody had scrolled down to look. On their own page they only
    run while somebody is watching them.
    """

    def setUp(self):
        self.camera = Camera.objects.create(name="Front door", source="0")

    def test_the_dashboard_carries_no_streams(self):
        page = self.client.get("/").content.decode()
        self.assertNotIn('class="camera"', page)
        self.assertNotIn('data-role="viewport"', page)

    def test_the_cameras_page_carries_them(self):
        page = self.client.get("/cameras/").content.decode()
        self.assertIn('class="camera"', page)
        self.assertIn('data-role="viewport"', page)

    def test_both_pages_can_reach_the_other(self):
        # The desktop window has no Back button, so a page with no way
        # out is a dead end.
        for path in ("/", "/cameras/"):
            with self.subTest(path=path):
                page = self.client.get(path).content.decode()
                self.assertIn('href="/"', page)
                self.assertIn('href="/cameras/"', page)

    def test_each_page_marks_where_you_are(self):
        self.assertIn('href="/" class="on"',
                      self.client.get("/").content.decode())
        self.assertIn('href="/cameras/" class="on"',
                      self.client.get("/cameras/").content.decode())

    def test_the_cameras_page_asks_only_for_cameras(self):
        # It used to call /api/v1/stats/ and fill dashboard tiles that are
        # not there, which threw on the first one and left every card
        # stuck on "checking...".
        page = self.client.get("/cameras/").content.decode()
        self.assertIn('fetch("/api/v1/cameras/")', page)
        self.assertNotIn('getElementById("t-in")', page)

    def test_the_dashboard_still_updates_its_tiles(self):
        # That logic lived in the function that moved.
        page = self.client.get("/").content.decode()
        self.assertIn("refreshTotals", page)
        self.assertIn('getElementById("t-in")', page)


class CrossingProvenanceTests(TestCase):
    """What was true of a track when it crossed, kept rather than dropped.

    All of this was already computed by the detector and the tracker and
    thrown away at the line. Without it a disagreement with a hand count
    says only that the figure is wrong.
    """

    def counter_over(self, ys, orientation=HORIZONTAL, scores=None):
        frames = [[box_at(160, y)] for y in ys]
        detector = StubDetector(frames, scores=scores) if scores else StubDetector(frames)
        counter = VisitorCounter(
            detector=detector,
            tracker=build_tracker("sort", max_disappeared=25),
            line=LineCounter(position=0.5, orientation=orientation),
            detect_every=1,
        )
        frame = blank_frame()
        events = []
        for _ in ys:
            events.extend(counter.process(frame)[1])
        return events

    def test_a_crossing_reports_where_along_the_line_it_happened(self):
        events = self.counter_over(list(range(10, 235, 15)))
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(events[0].offset, 0.0)
        self.assertLessEqual(events[0].offset, 1.0)

    def test_a_crossing_reports_its_speed(self):
        slow = self.counter_over(list(range(10, 235, 8)))
        fast = self.counter_over(list(range(10, 235, 40)))
        self.assertGreater(fast[0].speed, slow[0].speed,
                           "a sprint should not look like a stroll")

    def test_a_crossing_reports_the_box_it_was_seen_in(self):
        events = self.counter_over(list(range(10, 235, 15)))
        self.assertGreater(events[0].width, 0)
        self.assertGreater(events[0].height, 0)

    def test_the_detector_confidence_survives_the_tracker(self):
        # It used to be dropped at update_detections, so every crossing
        # looked equally certain.
        events = self.counter_over(list(range(10, 235, 15)), scores=[0.83])
        self.assertAlmostEqual(events[0].confidence, 0.83, places=2)

    def test_the_worker_stores_all_of_it(self):
        camera = Camera.objects.create(name="Door", source="0")
        worker = CounterWorker(camera)
        worker._persist([CrossingEvent(
            track_id=7, direction=DIRECTION_IN, confidence=0.91,
            offset=0.33, speed=4.2, width=40.0, height=120.0)])
        row = Crossing.objects.get()
        self.assertEqual(row.track_id, 7)
        self.assertAlmostEqual(row.confidence, 0.91)
        self.assertAlmostEqual(row.line_offset, 0.33)
        self.assertAlmostEqual(row.speed, 4.2)
        self.assertAlmostEqual(row.box_width, 40.0)
        self.assertAlmostEqual(row.box_height, 120.0)

    def test_older_rows_without_it_are_fine(self):
        # Everything recorded before this existed has nulls, and nothing
        # should trip over them.
        camera = Camera.objects.create(name="Door", source="0")
        Crossing.objects.create(camera=camera, direction=DIRECTION_IN,
                                track_id=1, occurred_at=timezone.now())
        response = self.client.get("/api/v1/crossings/?days=7")
        self.assertEqual(response.status_code, 200)
        rows = list(csv.reader(response.content.decode().splitlines()))
        self.assertEqual(rows[1][5], "", "a missing confidence is blank, not 0")


class CrossingsCsvTests(TestCase):
    """One row per crossing, for comparing against a hand count."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Door", source="0")
        self.event = Event.objects.create(
            name="Trade show", starts_at=timezone.now() - timedelta(days=1),
            ends_at=timezone.now() + timedelta(days=1))
        for i in range(3):
            Crossing.objects.create(
                camera=self.camera, event=self.event, direction=DIRECTION_IN,
                track_id=i, occurred_at=timezone.now(), confidence=0.6 + i / 10,
                line_offset=0.5, speed=4.0, box_width=40.0, box_height=120.0)

    def test_it_gives_a_row_per_crossing_not_per_day(self):
        rows = list(csv.reader(self.client.get(
            "/api/v1/crossings/?days=7").content.decode().splitlines()))
        self.assertEqual(len(rows) - 1, 3)

    def test_every_measurement_is_there(self):
        rows = list(csv.reader(self.client.get(
            "/api/v1/crossings/?days=7").content.decode().splitlines()))
        for column in ("confidence", "line_offset", "speed",
                       "box_width", "box_height"):
            self.assertIn(column, rows[0])

    def test_it_filters_by_event_like_the_others(self):
        other = Event.objects.create(
            name="Quiet day", starts_at=timezone.now() - timedelta(days=1),
            ends_at=timezone.now())
        rows = list(csv.reader(self.client.get(
            f"/api/v1/crossings/?days=7&event={other.pk}"
        ).content.decode().splitlines()))
        self.assertEqual(len(rows) - 1, 0)


class EnableDisableCameraTests(TestCase):
    """Turning a camera off and keeping it off.

    Stop alone does not hold where a supervisor is running: it restarts
    any camera that is active and not alive, so a stopped camera comes
    back within a cooldown. Disabling is the durable switch.
    """

    def setUp(self):
        self.staff = User.objects.create_superuser("boss", "b@example.com", "pw")
        self.camera = Camera.objects.create(name="Door", source="0", is_active=True)

    def post(self, action):
        return self.client.post(f"/api/v1/cameras/{self.camera.pk}/{action}/")

    def test_disable_marks_it_inactive(self):
        self.client.force_login(self.staff)
        response = self.post("disable")
        self.assertEqual(response.status_code, 200)
        self.camera.refresh_from_db()
        self.assertFalse(self.camera.is_active)

    def test_enable_marks_it_active(self):
        self.camera.is_active = False
        self.camera.save()
        self.client.force_login(self.staff)
        self.post("enable")
        self.camera.refresh_from_db()
        self.assertTrue(self.camera.is_active)

    def test_disable_also_stops_it_rather_than_waiting_for_a_pass(self):
        self.client.force_login(self.staff)
        with patch.object(camera_module.REGISTRY, "stop",
                          return_value=True) as stop:
            self.post("disable")
        stop.assert_called_once_with(self.camera.pk)

    def test_enabling_twice_is_harmless(self):
        self.client.force_login(self.staff)
        self.assertEqual(self.post("enable").status_code, 200)
        self.assertEqual(self.post("enable").status_code, 200)
        self.camera.refresh_from_db()
        self.assertTrue(self.camera.is_active)

    def test_an_anonymous_visitor_cannot_turn_cameras_off(self):
        response = self.post("disable")
        self.assertEqual(response.status_code, 401)
        self.camera.refresh_from_db()
        self.assertTrue(self.camera.is_active)

    def test_a_missing_camera_is_a_404(self):
        self.client.force_login(self.staff)
        self.assertEqual(
            self.client.post("/api/v1/cameras/99999/disable/").status_code, 404)


class StopIsHonestAboutComingBackTests(TestCase):
    """Stop must not look permanent when it is not."""

    def setUp(self):
        self.staff = User.objects.create_superuser("boss", "b@example.com", "pw")
        self.camera = Camera.objects.create(name="Door", source="0", is_active=True)
        self.client.force_login(self.staff)

    def stop(self):
        return self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/stop/").json()

    def test_it_warns_when_a_supervisor_will_restart_it(self):
        with patch("counter.views.current_supervisor", return_value=object()):
            self.assertTrue(self.stop()["will_restart"])

    def test_it_does_not_warn_without_a_supervisor(self):
        with patch("counter.views.current_supervisor", return_value=None):
            self.assertFalse(self.stop()["will_restart"])

    def test_a_disabled_camera_is_not_going_to_restart(self):
        self.camera.is_active = False
        self.camera.save()
        with patch("counter.views.current_supervisor", return_value=object()):
            self.assertFalse(self.stop()["will_restart"])


class CameraPageControlsTests(TestCase):
    """The controls an operator actually sees."""

    def setUp(self):
        self.staff = User.objects.create_superuser("boss", "b@example.com", "pw")
        Camera.objects.create(name="Door", source="0")

    def page(self):
        self.client.force_login(self.staff)
        return self.client.get("/cameras/").content.decode()

    def test_enable_and_disable_are_offered(self):
        page = self.page()
        self.assertIn('data-action="enable"', page)
        self.assertIn('data-action="disable"', page)

    def test_a_disabled_camera_says_so(self):
        self.assertIn("it will not run until you enable it", self.page())

    def test_start_is_refused_for_a_disabled_camera(self):
        # The supervisor would stop it again on its next pass.
        self.assertIn("startButton.disabled = data.running || !active",
                      self.page())

    def test_anonymous_visitors_get_no_switches(self):
        # The selector strings appear in the script on every page, so
        # check the markup: no buttons rendered, and the notice instead.
        page = self.client.get("/cameras/").content.decode()
        self.assertNotIn('<button data-action="disable"', page)
        self.assertNotIn('<button class="primary" data-action="start"', page)
        self.assertIn("Read-only", page)
