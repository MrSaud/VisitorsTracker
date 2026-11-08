"""Tests for tracking, line counting and the API.

A stub detector stands in for HOG/ONNX so nothing here needs a camera,
a model file, or any particular machine speed.
"""

import pathlib
import tempfile

import cv2
import numpy as np
from django.test import TestCase, TransactionTestCase

from .camera import CounterWorker
from .models import Camera, CountEvent
from .vision import (
    DIRECTION_IN,
    DIRECTION_OUT,
    HORIZONTAL,
    VERTICAL,
    CentroidTracker,
    HogPersonDetector,
    LineCounter,
    VisitorCounter,
    non_max_suppression,
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
        self.assertEqual(camera.totals(), {"in": 3, "out": 1, "inside": 2})

    def test_inside_clamps_at_zero(self):
        camera = Camera.objects.create(name="Door", source="0")
        CountEvent.objects.create(camera=camera, track_id=1, direction=DIRECTION_OUT)
        self.assertEqual(camera.totals()["inside"], 0)


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
        self.assertEqual(body["totals"], {"in": 2, "out": 1, "inside": 1})
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
            self.assertEqual(camera.totals(), {"in": 1, "out": 0, "inside": 1})

    def test_unopenable_source_reports_an_error_instead_of_crashing(self):
        camera = Camera.objects.create(name="Missing", source="/no/such/clip.mp4")
        worker = CounterWorker(camera)
        worker.start()
        worker.join(timeout=15)

        self.assertFalse(worker.is_alive())
        self.assertIsNotNone(worker.error)
        self.assertIn("Could not open video source", worker.error)
        self.assertEqual(CountEvent.objects.count(), 0)
