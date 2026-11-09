"""Count a recorded clip and compare the result against the truth.

Until you can say "within 3%", you cannot put a number in front of a
client. This runs the real counter over a file -- same detector, same
tracker, same line logic as a live camera -- and tells you what it got.

    python manage.py measure door.mp4 --expect-in 143 --expect-out 139

Nothing is written to the database: this is a measurement, not a shift.
Every crossing is logged with the time in the clip it happened, so a
disagreement can be looked at rather than argued about:

    python manage.py measure door.mp4 --record checked.mp4 --csv found.csv

Then open checked.mp4 at the timestamp of the crossing you doubt.
"""

import csv
import pathlib
import time

import cv2
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from counter.tracking import build_tracker
from counter.vision import (
    HORIZONTAL,
    VERTICAL,
    LineCounter,
    VisitorCounter,
    build_detector,
)

FOURCC = {".mp4": "mp4v", ".m4v": "mp4v", ".avi": "MJPG", ".mov": "mp4v"}


def clock(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


class Command(BaseCommand):
    help = "Run the counter over a video file and report what it counted."

    def add_arguments(self, parser):
        parser.add_argument("clip", help="Video file to count.")
        parser.add_argument("--expect-in", type=int, dest="expect_in",
                            help="How many entries the clip really contains.")
        parser.add_argument("--expect-out", type=int, dest="expect_out",
                            help="How many exits the clip really contains.")
        parser.add_argument("--line", type=float, default=0.5,
                            help="Line position as a fraction of the frame (default 0.5).")
        parser.add_argument("--orientation", choices=[HORIZONTAL, VERTICAL],
                            default=HORIZONTAL,
                            help="Which way the line runs (default horizontal).")
        parser.add_argument("--invert", action="store_true",
                            help="Swap which direction counts as in.")
        parser.add_argument("--detect-every", type=int, dest="detect_every",
                            default=settings.COUNTER_DETECT_EVERY,
                            help="Detect on every Nth frame.")
        parser.add_argument("--record", help="Write the annotated video here.")
        parser.add_argument("--csv", dest="csv_path",
                            help="Write the crossings here (default beside the clip).")
        parser.add_argument("--quiet", action="store_true",
                            help="No progress line.")

    def handle(self, *args, **options):
        clip = pathlib.Path(options["clip"]).expanduser()
        if not clip.exists():
            raise CommandError(f"No such file: {clip}")

        capture = cv2.VideoCapture(str(clip))
        if not capture.isOpened():
            raise CommandError(
                f"OpenCV could not open {clip}. Is it a format it can read?")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        counter = self._build_counter(options)
        writer = None
        crossings = []
        frame_index = 0
        started = time.monotonic()

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                annotated, events = counter.process(frame)
                at = frame_index / source_fps if source_fps else 0.0
                for track_id, direction in events:
                    crossings.append({
                        "frame": frame_index,
                        "at": clock(at),
                        "seconds": round(at, 2),
                        "direction": direction,
                        "track_id": track_id,
                    })

                if options["record"]:
                    writer = writer or self._writer(
                        options["record"], annotated, source_fps)
                    if writer is not None:
                        writer.write(annotated)

                frame_index += 1
                if not options["quiet"] and frame_index % 50 == 0:
                    self._progress(frame_index, total_frames, counter)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if not options["quiet"]:
            self.stdout.write("")
        self._report(clip, options, counter, crossings, frame_index,
                     source_fps, time.monotonic() - started)

    # -- pieces -----------------------------------------------------------

    def _build_counter(self, options):
        detector = build_detector(
            settings.COUNTER_DETECTOR,
            settings.COUNTER_MIN_CONFIDENCE,
            settings.COUNTER_ONNX_MODEL,
        )
        return VisitorCounter(
            detector=detector,
            tracker=build_tracker(
                settings.COUNTER_TRACKER,
                max_disappeared=settings.COUNTER_MAX_DISAPPEARED,
                detector=detector,
            ),
            line=LineCounter(
                position=options["line"],
                orientation=options["orientation"],
                invert=options["invert"],
                confirm_frames=settings.COUNTER_LINE_CONFIRM_FRAMES,
            ),
            detect_every=options["detect_every"],
            max_disappeared=settings.COUNTER_MAX_DISAPPEARED,
            frame_width=settings.COUNTER_FRAME_WIDTH,
        )

    def _writer(self, path, frame, source_fps):
        target = pathlib.Path(path).expanduser()
        fourcc = cv2.VideoWriter_fourcc(*FOURCC.get(target.suffix.lower(), "mp4v"))
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(target), fourcc, source_fps or 20.0, (width, height))
        if not writer.isOpened():
            self.stderr.write(self.style.WARNING(
                f"Could not write {target}; carrying on without the recording."))
            return None
        return writer

    def _progress(self, frame_index, total_frames, counter):
        share = f"{frame_index}/{total_frames}" if total_frames else str(frame_index)
        self.stdout.write(
            f"  {share} frames · in {counter.total_in} · out {counter.total_out}",
            ending="\r")
        self.stdout.flush()

    def _report(self, clip, options, counter, crossings, frames, source_fps, elapsed):
        duration = frames / source_fps if source_fps else 0.0
        self.stdout.write(self.style.SUCCESS(f"{clip.name}"))
        self.stdout.write(
            f"  {frames} frames, {clock(duration)} at {source_fps:.0f} fps "
            f"· detector {counter.detector.name} "
            f"· processed in {elapsed:.1f}s"
        )
        self.stdout.write(
            f"  line {options['orientation']} at {options['line']}"
            + (" (inverted)" if options["invert"] else "")
        )
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"  counted: {counter.total_in} in, {counter.total_out} out"))

        self._compare(options, counter)

        path = self._write_csv(clip, options, crossings)
        if path is not None:
            self.stdout.write(f"  {len(crossings)} crossing(s) written to {path}")
        if options["record"]:
            self.stdout.write(f"  annotated video written to {options['record']}")
            self.stdout.write(
                "  open it at the time of any crossing you doubt.")

    def _compare(self, options, counter):
        expected_in = options["expect_in"]
        expected_out = options["expect_out"]
        if expected_in is None and expected_out is None:
            self.stdout.write(
                "  (pass --expect-in/--expect-out with a hand count to see the error)")
            return

        self.stdout.write("")
        for label, got, expected in (("in", counter.total_in, expected_in),
                                     ("out", counter.total_out, expected_out)):
            if expected is None:
                continue
            error = got - expected
            share = (abs(error) / expected * 100) if expected else 0.0
            line = f"  {label:>3}: counted {got}, truth {expected} → {error:+d}"
            if expected:
                line += f" ({share:.1f}%)"
            style = self.style.SUCCESS if share <= 3 and expected else self.style.WARNING
            self.stdout.write(style(line))

    def _write_csv(self, clip, options, crossings):
        if options["csv_path"] is None and not crossings:
            return None
        target = pathlib.Path(
            options["csv_path"] or clip.with_suffix(".crossings.csv")).expanduser()
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["frame", "at", "seconds", "direction", "track_id"])
            writer.writeheader()
            writer.writerows(crossings)
        return target
