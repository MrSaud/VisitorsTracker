"""Diagnose a camera: is anyone detected, and where should the line go?

Answers the two questions behind "it never counts":

1. Is the detector seeing a person at all? If not, the problem is framing
   or distance, and no line setting will help.
2. Which way do people travel across this view? A horizontal line counts
   nothing if people move left-to-right.
"""

import time

import cv2
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from counter.models import Camera
from counter.vision import (
    HORIZONTAL,
    VERTICAL,
    build_detector,
    suggest_line_setup,
)


class Command(BaseCommand):
    help = "Watch a camera and report detection quality plus a suggested counting line."

    def add_arguments(self, parser):
        parser.add_argument("--source", help="Webcam index, file path or URL. Default 0.")
        parser.add_argument("--camera", type=int, help="Probe an existing camera by id.")
        parser.add_argument("--seconds", type=float, default=20.0,
                            help="How long to watch. Default 20.")
        parser.add_argument("--apply", action="store_true",
                            help="Write the suggested line to the camera (needs --camera).")

    def handle(self, *args, **options):
        camera = None
        if options["camera"]:
            try:
                camera = Camera.objects.get(pk=options["camera"])
            except Camera.DoesNotExist:
                raise CommandError(f"No camera with id {options['camera']}")
            source = camera.resolved_source()
        else:
            raw = options["source"] or "0"
            source = int(raw) if str(raw).isdigit() else raw

        if options["apply"] and camera is None:
            raise CommandError("--apply needs --camera <id>")

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise CommandError(
                f"Could not open {source!r}.\n"
                "  - webcam: grant camera access to your terminal app in\n"
                "    System Settings > Privacy & Security > Camera, then fully quit\n"
                "    and reopen it (macOS only re-reads this at launch)\n"
                "  - if an iPhone Continuity Camera took index 0, try --source 1"
            )

        detector = build_detector(
            settings.COUNTER_DETECTOR,
            settings.COUNTER_MIN_CONFIDENCE,
            settings.COUNTER_ONNX_MODEL,
        )
        self.stdout.write(f"Watching {source!r} for {options['seconds']:.0f}s "
                          f"using the {detector.name} detector.")
        self.stdout.write("Walk through the view the way visitors actually would.\n")

        centroids = []
        frames = frames_with_person = 0
        box_heights = []
        confidences = []
        frame_shape = None
        started = time.monotonic()
        last_print = 0.0

        try:
            while time.monotonic() - started < options["seconds"]:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                height, width = frame.shape[:2]
                if width > settings.COUNTER_FRAME_WIDTH:
                    scale = settings.COUNTER_FRAME_WIDTH / float(width)
                    frame = cv2.resize(frame, (settings.COUNTER_FRAME_WIDTH,
                                               int(round(height * scale))))
                frame_shape = frame.shape
                frames += 1

                detections = detector.detect(frame)
                if detections:
                    frames_with_person += 1
                    # Track the largest box: the nearest/clearest person.
                    box, score = max(detections, key=lambda d: (d[0][3] - d[0][1]))
                    x1, y1, x2, y2 = box
                    centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                    box_heights.append((y2 - y1) / float(frame.shape[0]))
                    confidences.append(score)

                now = time.monotonic() - started
                if now - last_print >= 0.5:
                    last_print = now
                    if detections:
                        cx, cy = centroids[-1]
                        msg = (f"  {now:5.1f}s  {len(detections)} person(s)  "
                               f"centroid x={cx:4.0f} y={cy:4.0f}  "
                               f"fills {box_heights[-1]*100:3.0f}% of frame height")
                    else:
                        msg = f"  {now:5.1f}s  nobody detected"
                    self.stdout.write(msg)
        except KeyboardInterrupt:
            self.stdout.write("\ninterrupted")
        finally:
            capture.release()

        self._report(frames, frames_with_person, box_heights, confidences,
                     centroids, frame_shape, camera, options["apply"])

    # -- reporting --------------------------------------------------------

    def _report(self, frames, hits, box_heights, confidences, centroids,
                frame_shape, camera, apply_changes):
        self.stdout.write("\n" + "=" * 58)
        if frames == 0:
            self.stderr.write(self.style.ERROR("No frames were read at all."))
            return

        rate = hits / frames * 100
        self.stdout.write(f"frames examined      : {frames}")
        self.stdout.write(f"frames with a person : {hits} ({rate:.0f}%)")

        if hits == 0:
            self.stderr.write(self.style.ERROR(
                "\nNobody was detected, so counting cannot work yet.\n"
                "This is a framing problem, not a line problem:\n"
                "  - stand further back. The detector wants most of a body,\n"
                "    roughly head-to-knees; a head-and-shoulders webcam shot\n"
                "    is the single most common cause\n"
                "  - make sure you are lit from the front, not backlit by a window\n"
                "  - check the view is not so wide that people are tiny"
            ))
            return

        avg_fill = sum(box_heights) / len(box_heights) * 100
        avg_conf = sum(confidences) / len(confidences)
        self.stdout.write(f"average confidence   : {avg_conf:.2f}")
        self.stdout.write(f"body fills on average: {avg_fill:.0f}% of frame height")

        if avg_fill > 85:
            self.stdout.write(self.style.WARNING(
                "  -> very close to the camera. Back up so whole people fit."))
        elif avg_fill < 15:
            self.stdout.write(self.style.WARNING(
                "  -> people are small in frame. Move the camera closer or zoom in."))
        else:
            self.stdout.write(self.style.SUCCESS("  -> good framing for detection."))

        if rate < 50:
            self.stdout.write(self.style.WARNING(
                f"  -> only seen in {rate:.0f}% of frames; tracking may drop people.\n"
                "     Improve lighting or framing before tuning the line."))

        suggestion = suggest_line_setup(centroids, frame_shape)
        self.stdout.write("")
        if not suggestion or not suggestion["confident"]:
            self.stdout.write(self.style.WARNING(
                "Could not suggest a line: not enough movement across the view.\n"
                "Re-run and walk fully across the frame the way a visitor would."))
            return

        orientation = suggestion["orientation"]
        position = suggestion["position"]
        axis_word = "left-to-right" if orientation == VERTICAL else "up-and-down"
        self.stdout.write(self.style.SUCCESS("Suggested counting line"))
        self.stdout.write(f"  movement was mostly  : {axis_word}")
        self.stdout.write(f"  line orientation     : {orientation}")
        self.stdout.write(f"  line position        : {position}")

        toward = "increasing" if suggestion["travel"] == "increasing" else "decreasing"
        in_dir = ("rightwards" if orientation == VERTICAL else "downwards")
        if toward == "increasing":
            self.stdout.write(f"  you moved {in_dir}, which currently counts as IN")
        else:
            opposite = "leftwards" if orientation == VERTICAL else "upwards"
            self.stdout.write(self.style.WARNING(
                f"  you moved {opposite}, which currently counts as OUT.\n"
                "  If that direction is people ENTERING, tick 'invert direction'."))

        if camera is None:
            self.stdout.write(
                "\nSet these in the admin, or re-run with --camera <id> --apply")
            return

        if not apply_changes:
            self.stdout.write(
                f"\nApply with: python manage.py camera_probe "
                f"--camera {camera.pk} --seconds 20 --apply")
            return

        camera.line_orientation = orientation
        camera.line_position = position
        camera.save(update_fields=["line_orientation", "line_position"])
        self.stdout.write(self.style.SUCCESS(
            f"\nSaved to camera {camera.pk} ({camera.name})."))
