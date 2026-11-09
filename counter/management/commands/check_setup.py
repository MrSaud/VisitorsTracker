"""Report whether this machine can actually run the counter.

Written because the failures that matter here are quiet ones. A missing
detector model does not stop anything: the counter falls back to HOG and
carries on counting badly, with nothing on screen to say so. A camera that
cannot be opened shows as an error on one dashboard tile that nobody is
looking at yet. This says all of it at once, on whichever platform it is
run, so "is this machine ready" has an answer before the camera goes up.
"""

from __future__ import annotations

import platform
import socket
import sys
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection

OK, WARN, FAIL = "ok", "warn", "fail"


class Command(BaseCommand):
    help = "Check that this machine can run the counter, and say what is missing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--cameras", action="store_true",
            help="Also try to open each configured camera. This takes the "
                 "device, so do not run it while the counter is running.")

    def handle(self, *args, **options):
        self.failures = 0
        self.warnings = 0

        self.stdout.write(self.style.MIGRATE_HEADING("Machine"))
        self.report(OK, "platform", f"{platform.system()} {platform.release()} "
                                    f"({platform.machine()})")
        self.report(OK, "python", sys.version.split()[0])

        self.stdout.write(self.style.MIGRATE_HEADING("\nDependencies"))
        self.check_imports()

        self.stdout.write(self.style.MIGRATE_HEADING("\nDetector"))
        self.check_detector()

        self.stdout.write(self.style.MIGRATE_HEADING("\nDatabase"))
        self.check_database()

        self.stdout.write(self.style.MIGRATE_HEADING("\nServer"))
        self.check_port()

        self.stdout.write(self.style.MIGRATE_HEADING("\nCameras"))
        self.check_cameras(open_them=options["cameras"])

        self.stdout.write("")
        if self.failures:
            self.stdout.write(self.style.ERROR(
                f"{self.failures} thing(s) would stop this working."))
        elif self.warnings:
            self.stdout.write(self.style.WARNING(
                f"Usable, with {self.warnings} thing(s) worth fixing."))
        else:
            self.stdout.write(self.style.SUCCESS("Ready."))

    # -- reporting --------------------------------------------------------

    def report(self, level, label, detail, fix=None):
        marks = {OK: ("  ok  ", self.style.SUCCESS),
                 WARN: (" warn ", self.style.WARNING),
                 FAIL: (" FAIL ", self.style.ERROR)}
        mark, style = marks[level]
        self.stdout.write(f"{style(mark)} {label:22} {detail}")
        if fix:
            self.stdout.write(f"         {' ' * 22} -> {fix}")
        if level == FAIL:
            self.failures += 1
        elif level == WARN:
            self.warnings += 1

    # -- checks -----------------------------------------------------------

    def check_imports(self):
        for module, label in (("cv2", "opencv"), ("numpy", "numpy"),
                              ("django", "django"), ("rest_framework", "drf")):
            try:
                imported = __import__(module)
                version = getattr(imported, "__version__", "?")
                self.report(OK, label, version)
            except ImportError as error:
                self.report(FAIL, label, str(error),
                            "pip install -r requirements.txt")

        try:
            __import__("webview")
            self.report(OK, "pywebview", "installed (desktop window available)")
        except ImportError:
            self.report(WARN, "pywebview", "not installed",
                        "pip install -r requirements-desktop.txt "
                        "(only needed for the desktop window)")

    def check_detector(self):
        from counter.vision import build_detector, resolve_model_path

        # The configured model, not the default one. Checking the default
        # would report "fine" for the very case this exists to catch:
        # COUNTER_ONNX_MODEL pointing at a file that is not there.
        model_name = settings.COUNTER_ONNX_MODEL
        model = resolve_model_path(model_name)
        if model.exists():
            size = model.stat().st_size / (1024 * 1024)
            self.report(OK, "model file", f"{model.name} ({size:.1f}MB)")
        else:
            self.report(FAIL, "model file", f"missing: {model}",
                        "python manage.py export_model")

        try:
            detector = build_detector(
                settings.COUNTER_DETECTOR,
                min_confidence=settings.COUNTER_MIN_CONFIDENCE,
                model_name=model_name)
        except Exception as error:          # pragma: no cover - defensive
            self.report(FAIL, "detector", f"could not be built: {error}")
            return

        name = type(detector).__name__
        if "Hog" in name or "HOG" in name:
            # The fallback that costs accuracy and announces nothing.
            self.report(
                WARN, "detector in use", f"{name} -- not YOLO",
                "python manage.py export_model, then check again")
        else:
            self.report(OK, "detector in use", name)

    def check_database(self):
        from django.db.migrations.executor import MigrationExecutor

        try:
            connection.ensure_connection()
        except Exception as error:
            self.report(FAIL, "database", f"unreachable: {error}")
            return

        if connection.vendor == "sqlite":
            path = Path(settings.DATABASES["default"]["NAME"])
            writable = path.exists() and os_access(path)
            self.report(OK if writable else FAIL, "database",
                        f"{path.name} ({'writable' if writable else 'NOT writable'})",
                        None if writable else f"check permissions on {path}")
        else:
            self.report(OK, "database", connection.vendor)

        plan = MigrationExecutor(connection).migration_plan(
            MigrationExecutor(connection).loader.graph.leaf_nodes())
        if plan:
            self.report(FAIL, "migrations", f"{len(plan)} not applied",
                        "python manage.py migrate")
        else:
            self.report(OK, "migrations", "up to date")

    def check_port(self):
        from counter.desktop import DEFAULT_PORT, ours_is_already_serving, port_is_free

        if ours_is_already_serving(port=DEFAULT_PORT):
            self.report(OK, f"port {DEFAULT_PORT}", "VisitorsTracker is running")
        elif port_is_free(DEFAULT_PORT):
            self.report(OK, f"port {DEFAULT_PORT}", "free")
        else:
            self.report(WARN, f"port {DEFAULT_PORT}", "busy with something else",
                        "stop it, or set COUNTER_DESKTOP_PORT")

        if settings.DEBUG:
            self.report(WARN, "DJANGO_DEBUG", "on",
                        "set DJANGO_DEBUG=0 before anyone else can reach this: "
                        "tracebacks expose settings")

    def check_cameras(self, open_them=False):
        from counter.models import Camera

        cameras = list(Camera.objects.all())
        if not cameras:
            self.report(WARN, "configured", "none",
                        "add one in the admin, or see 'Adding a camera'")
            return

        for camera in cameras:
            state = "active" if camera.is_active else "inactive"
            self.report(OK, camera.name[:22], f"{camera.source!r} ({state})")

        if not open_them:
            self.stdout.write(
                "         pass --cameras to actually open them "
                "(not while the counter is running)")
            return

        import cv2
        from counter.camera import capture_backends, camera_privacy_location

        for camera in cameras:
            source = camera.resolved_source()
            opened = False
            for backend in capture_backends(source):
                capture = (cv2.VideoCapture(source) if backend == cv2.CAP_ANY
                           else cv2.VideoCapture(source, backend))
                opened = capture.isOpened()
                capture.release()
                if opened:
                    break
            if opened:
                self.report(OK, f"open {camera.name[:16]}", "opened")
            else:
                self.report(
                    FAIL, f"open {camera.name[:16]}", f"cannot open {source!r}",
                    f"camera access in {camera_privacy_location()}"
                    if isinstance(source, int) else "check the URL or file path")


def os_access(path) -> bool:
    import os

    return os.access(path, os.W_OK)
