"""Tests for tracking, line counting and the API.

A stub detector stands in for HOG/ONNX so nothing here needs a camera,
a model file, or any particular machine speed.
"""

import pathlib
import tempfile
from unittest import skipUnless

import cv2
import numpy as np
from django.test import TestCase, TransactionTestCase

from .camera import CounterWorker
from .models import Camera, CountEvent
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
