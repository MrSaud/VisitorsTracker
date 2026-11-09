"""Fetch and convert the YOLO detector to ONNX, on any platform.

The model is not in the repository: it is 12MB of weights that git would
carry forever, and it is reproducible from a published checkpoint. So it
has to be built on each machine, and this replaces a bash script that
could only ever run on macOS and Linux.

Ultralytics is not a dependency of this project. It pulls in torch, which
is an order of magnitude larger than everything else here combined, and it
is needed once to produce a file that is then read by OpenCV's own ONNX
runtime. So it is installed into a throwaway environment and discarded.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

DEFAULT_MODEL = "yolov8n"
BUILD_REQUIREMENTS = ("ultralytics", "onnx", "onnxslim", "onnxruntime")

EXPORT_SNIPPET = """
import sys
from ultralytics import YOLO
name = sys.argv[1]
print(YOLO(name + ".pt").export(format="onnx", opset=12, imgsz=640,
                                simplify=False))
"""


def venv_python(venv_dir) -> Path:
    """Where a virtualenv keeps its interpreter on this platform."""
    venv_dir = Path(venv_dir)
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


class Command(BaseCommand):
    help = "Download a YOLO checkpoint and export it to ONNX."

    def add_arguments(self, parser):
        parser.add_argument(
            "model", nargs="?", default=DEFAULT_MODEL,
            help=f"Checkpoint to export (default: {DEFAULT_MODEL}). "
                 "yolov8s is more accurate and slower.")
        parser.add_argument(
            "--output", default=None,
            help="Where to put the .onnx (default: counter/ml_models/).")
        parser.add_argument(
            "--python", default=None,
            help="An interpreter that already has ultralytics, instead of "
                 "building a throwaway environment. Saves a large download "
                 "when you have one.")

    def handle(self, *args, **options):
        model = options["model"].strip()
        if not model or "/" in model or "\\" in model or model.startswith("-"):
            raise CommandError(f"{model!r} is not a checkpoint name.")

        destination = Path(options["output"] or
                           Path(settings.BASE_DIR) / "counter" / "ml_models")
        destination.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            python = self.prepare(work, options["python"])

            self.stdout.write(f"Exporting {model} to ONNX...")
            # YOLO_CONFIG_DIR keeps ultralytics from writing settings into
            # the user's home directory for a one-off build.
            result = subprocess.run(
                [str(python), "-c", EXPORT_SNIPPET, model],
                cwd=work, capture_output=True, text=True,
                env={**self.clean_env(), "YOLO_CONFIG_DIR": str(work / "cfg")})
            if result.returncode != 0:
                raise CommandError(
                    f"Export failed:\n{result.stderr.strip() or result.stdout}")

            produced = work / f"{model}.onnx"
            if not produced.exists():
                raise CommandError(
                    f"Export reported success but {produced.name} is not "
                    f"there. Output was:\n{result.stdout.strip()}")

            target = destination / f"{model}.onnx"
            shutil.copy2(produced, target)

        size = target.stat().st_size / (1024 * 1024)
        self.stdout.write(self.style.SUCCESS(f"Wrote {target} ({size:.1f}MB)"))
        if model != DEFAULT_MODEL:
            self.stdout.write(
                f"Set COUNTER_ONNX_MODEL={model}.onnx to use it.")

    def prepare(self, work, supplied) -> Path:
        """The interpreter to export with."""
        if supplied:
            python = Path(supplied)
            if not python.exists():
                raise CommandError(f"No interpreter at {python}.")
            self.stdout.write(f"Using {python}")
            return python

        self.stdout.write(
            "Building a throwaway environment (ultralytics pulls in torch, "
            "so this is a large download the first time)...")
        subprocess.run([sys.executable, "-m", "venv", str(work / "venv")],
                       check=True, capture_output=True)
        python = venv_python(work / "venv")
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "-q", *BUILD_REQUIREMENTS],
            capture_output=True, text=True)
        if install.returncode != 0:
            raise CommandError(
                f"Could not install the build tools:\n{install.stderr.strip()}")
        return python

    @staticmethod
    def clean_env():
        """A plain environment, so this project's settings do not leak in."""
        import os

        return {k: v for k, v in os.environ.items()
                if not k.startswith(("DJANGO_", "COUNTER_"))}
