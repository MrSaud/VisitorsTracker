"""Run visitor counting from the command line, without the web server."""

import signal
import time

from django.core.management.base import BaseCommand, CommandError

from counter.camera import REGISTRY
from counter.models import Camera


class Command(BaseCommand):
    help = "Run the people counter for one camera, or for every active camera."

    def add_arguments(self, parser):
        parser.add_argument(
            "--camera", type=int, action="append", dest="cameras",
            help="Camera id to run. Repeat for several. Defaults to all active cameras.",
        )
        parser.add_argument(
            "--source",
            help="Run an ad-hoc source (webcam index, file path or URL) without a DB camera.",
        )
        parser.add_argument(
            "--name", default="ad-hoc",
            help="Name used for the camera created by --source.",
        )
        parser.add_argument(
            "--seconds", type=float, default=0,
            help="Stop automatically after this many seconds (0 = run until Ctrl-C).",
        )
        parser.add_argument(
            "--quiet", action="store_true", help="Do not print the periodic summary.",
        )

    def handle(self, *args, **options):
        cameras = self._resolve_cameras(options)
        if not cameras:
            raise CommandError(
                "No cameras to run. Add one in the admin, or pass --source 0."
            )

        workers = [REGISTRY.start(camera) for camera in cameras]
        time.sleep(1.5)

        alive = []
        for worker in workers:
            if worker.error:
                self.stderr.write(self.style.ERROR(
                    f"[{worker.camera_name}] {worker.error}"
                ))
            else:
                alive.append(worker)
                self.stdout.write(self.style.SUCCESS(
                    f"[{worker.camera_name}] running, detector={worker.detector_name}"
                ))

        if not alive:
            REGISTRY.stop_all()
            raise CommandError("No camera could be opened.")

        stopping = {"flag": False}

        def _handle_signal(signum, frame):
            stopping["flag"] = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        started = time.monotonic()
        try:
            while not stopping["flag"]:
                time.sleep(1.0)
                if not options["quiet"]:
                    self._print_summary(alive)
                if not any(w.is_alive() for w in alive):
                    self.stdout.write("All sources finished.")
                    break
                if options["seconds"] and time.monotonic() - started >= options["seconds"]:
                    break
        finally:
            REGISTRY.stop_all()

        self.stdout.write("")
        for worker in alive:
            counts = worker.counter
            self.stdout.write(self.style.SUCCESS(
                f"[{worker.camera_name}] final: in={counts.total_in} "
                f"out={counts.total_out} inside={counts.occupancy} "
                f"frames={worker.frames_read} fps={worker.fps:.1f}"
            ))

    # -- helpers ----------------------------------------------------------

    def _resolve_cameras(self, options):
        if options.get("source"):
            camera, _created = Camera.objects.get_or_create(
                source=options["source"],
                defaults={"name": options["name"]},
            )
            return [camera]

        if options.get("cameras"):
            cameras = list(Camera.objects.filter(pk__in=options["cameras"]))
            found = {c.pk for c in cameras}
            for missing in set(options["cameras"]) - found:
                self.stderr.write(self.style.WARNING(f"No camera with id {missing}"))
            return cameras

        return list(Camera.objects.filter(is_active=True))

    def _print_summary(self, workers):
        parts = []
        for worker in workers:
            counter = worker.counter
            parts.append(
                f"{worker.camera_name}: in={counter.total_in} out={counter.total_out} "
                f"inside={counter.occupancy} tracks={counter.active_tracks} "
                f"fps={worker.fps:.1f}"
            )
        self.stdout.write("  |  ".join(parts), ending="\r")
        self.stdout.flush()
