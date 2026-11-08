"""Tests for tracking, line counting and the API.

A stub detector stands in for HOG/ONNX so nothing here needs a camera,
a model file, or any particular machine speed.
"""

import itertools
import pathlib
import tempfile
import time as time_module
from io import StringIO
from unittest.mock import patch
from datetime import date, datetime, time, timedelta, timezone as dt_timezone
from unittest import skipUnless

import cv2
import numpy as np
from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone

from .camera import CounterWorker
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
    STALLED,
    STATUS_DEGRADED,
    STATUS_DOWN,
    STATUS_OK,
    camera_health,
    overall_status,
)
from .models import Camera, CountEvent
from .reports import (
    MAX_DAYS,
    busiest_hours,
    daily_counts,
    hourly_profile,
    opening_span,
    weekday_hour_matrix,
    local_today,
    parse_date,
    resolve_range,
    summarise,
)
from .vision import (
    DEFAULT_HOG_CONFIDENCE,
    DEFAULT_ONNX_CONFIDENCE,
    PERSON_CLASS_ID,
    DIRECTION_IN,
    DIRECTION_OUT,
    HORIZONTAL,
    VERTICAL,
    CentroidTracker,
    HogPersonDetector,
    LineCounter,
    OnnxPersonDetector,
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

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def detect(self, frame):
        boxes = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [(box, 1.0) for box in boxes]


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
        self.assertEqual([d for _tid, d in events], [DIRECTION_IN])
        self.assertEqual((counter.total_in, counter.total_out), (1, 0))

    def test_upward_crossing_counts_out(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL)
        events = self._run(counter, [200, 180, 140, 100, 80, 40])
        self.assertEqual([d for _tid, d in events], [DIRECTION_OUT])
        self.assertEqual((counter.total_in, counter.total_out), (0, 1))

    def test_round_trip_counts_both_directions(self):
        counter = LineCounter(position=0.5, orientation=HORIZONTAL)
        events = self._run(counter, [40, 100, 180, 200, 180, 100, 40])
        self.assertEqual([d for _tid, d in events], [DIRECTION_IN, DIRECTION_OUT])
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
        self.assertEqual([d for _tid, d in events], [DIRECTION_OUT])

    def test_vertical_line_uses_x_axis(self):
        counter = LineCounter(position=0.5, orientation=VERTICAL)
        tracker = CentroidTracker(max_disappeared=10)
        events = []
        for x in (40, 100, 150, 200, 260):
            tracks = tracker.update([box_at(x, 120)])
            events.extend(counter.update(tracks, (FRAME_H, FRAME_W, 3)))
        self.assertEqual([d for _tid, d in events], [DIRECTION_IN])

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
        self.assertEqual({d for _tid, d in total_events}, {DIRECTION_IN})
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
            CountEvent.objects.create(camera=camera, track_id=1, direction=direction)
        self.assertEqual(camera.totals(), {"in": 3, "out": 1})
        self.assertEqual(camera.occupancy()["inside"], 2)

    def test_inside_clamps_at_zero(self):
        camera = Camera.objects.create(name="Door", source="0")
        CountEvent.objects.create(camera=camera, track_id=1, direction=DIRECTION_OUT)
        occupancy = camera.occupancy()
        self.assertEqual(occupancy["inside"], 0)
        # The raw net stays visible so a reversed camera is diagnosable.
        self.assertEqual(occupancy["net"], -1)


class ApiTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Front door", source="0")
        CountEvent.objects.create(camera=self.camera, track_id=1, direction=DIRECTION_IN)
        CountEvent.objects.create(camera=self.camera, track_id=2, direction=DIRECTION_IN)
        CountEvent.objects.create(camera=self.camera, track_id=3, direction=DIRECTION_OUT)

    def test_dashboard_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Front door")

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

    def test_events_endpoint_respects_limit(self):
        response = self.client.get(f"/api/v1/cameras/{self.camera.pk}/events/?limit=2")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 2)

    def test_reset_deletes_events(self):
        self.client.force_login(
            User.objects.create_user("op", password="x", is_staff=True))
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 3)
        self.assertEqual(CountEvent.objects.count(), 0)

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

            events = list(CountEvent.objects.filter(camera=camera))
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
        self.assertEqual(CountEvent.objects.count(), 0)


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
        return CountEvent.objects.create(
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
            CountEvent.objects.create(
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
        self.assertEqual(lines[0], "date,in,out,net")
        self.assertEqual(lines[1], "2026-03-03,4,0,4")
        self.assertEqual(lines[2], "2026-03-04,0,0,0")

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
        CountEvent.objects.create(camera=self.camera, track_id=1,
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
        CountEvent.objects.create(camera=self.camera, track_id=1,
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
        self.assertEqual(CountEvent.objects.count(), 1, "history must survive")

    def test_anonymous_cannot_stop_a_camera(self):
        self.assertEqual(
            self.client.post(f"/api/v1/cameras/{self.camera.pk}/stop/").status_code, 401)

    def test_non_staff_user_cannot_write(self):
        self.client.force_login(self.normal)
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(CountEvent.objects.count(), 1)

    def test_staff_may_write(self):
        self.client.force_login(self.staff)
        response = self.client.post(f"/api/v1/cameras/{self.camera.pk}/reset/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(CountEvent.objects.count(), 0)

    def test_valid_token_may_write(self):
        response = self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/reset/",
            HTTP_AUTHORIZATION="Token test-token-value",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(CountEvent.objects.count(), 0)

    def test_wrong_token_is_rejected(self):
        response = self.client.post(
            f"/api/v1/cameras/{self.camera.pk}/reset/",
            HTTP_AUTHORIZATION="Token not-the-token",
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(CountEvent.objects.count(), 1)

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
        self.assertEqual(CountEvent.objects.count(), 1)

    # -- dashboard ------------------------------------------------------

    def test_dashboard_hides_controls_from_anonymous(self):
        response = self.client.get("/")
        self.assertNotContains(response, 'data-action="reset"')
        self.assertContains(response, "Read-only")

    def test_dashboard_shows_controls_to_staff(self):
        self.client.force_login(self.staff)
        response = self.client.get("/")
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
        CountEvent.objects.create(camera=camera or self.camera, track_id=1,
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
        CountEvent.objects.create(camera=self.camera, track_id=1,
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
            CountEvent.objects.create(
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
        self.assertEqual(lines[0], "hour,in,out,average_in,share_percent")
        self.assertEqual(len(lines), 25)
        self.assertTrue(lines[10].startswith("09:00,6,0,6.0,100.0"))

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
        CountEvent.objects.create(camera=self.good, track_id=1,
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
