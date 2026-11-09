# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a standalone VisitorsTracker.exe.

    pip install -r requirements.txt -r requirements-desktop.txt pyinstaller
    python manage.py export_model          # the model is bundled, not fetched
    pyinstaller VisitorsTracker.spec

Build this ON WINDOWS. PyInstaller bundles the interpreter and binaries of
the machine it runs on; it does not cross-compile, so a Mac produces a Mac
app and nothing else.

Three things PyInstaller cannot work out for a Django project, because
Django finds them by scanning the filesystem rather than importing them:

* management commands -- discovered by listing
  counter/management/commands/, so nothing imports them and nothing is
  traced. Collected explicitly below.
* migrations -- same, listed from counter/migrations/.
* templates -- never imported at all, just read. Carried as data.

The detector model goes in too: a frozen app has no writable place to put
one, and export_model builds a throwaway environment with torch in it,
which is not something to do on a customer's machine.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT = Path(SPECPATH)

# collect_submodules imports the package to walk it, and it runs now --
# while the spec is being evaluated -- not inside Analysis, so pathex has
# not been applied yet. Without this it quietly returns an empty list and
# the build succeeds while bundling none of the modules below.
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

datas = [
    (str(PROJECT / "counter" / "templates"), "counter/templates"),
    (str(PROJECT / "counter" / "migrations"), "counter/migrations"),
]

# The detector. Without it the app runs on HOG and counts badly, quietly.
model_dir = PROJECT / "counter" / "ml_models"
for model in model_dir.glob("*.onnx"):
    datas.append((str(model), "counter/ml_models"))

# Django is configured almost entirely by strings: INSTALLED_APPS,
# MIDDLEWARE, the database ENGINE, the DRF authentication and permission
# classes, the password validators. Nothing imports any of them, so
# PyInstaller's analysis cannot see them and the build succeeds without
# them. Each one is a crash on startup, discovered one at a time.
STRING_REFERENCED = ("config", "counter", "corsheaders", "rest_framework")

hiddenimports = (
    [name for package in STRING_REFERENCED
     for name in collect_submodules(package)]
    + [
        "django.contrib.admin.apps",
        "django.contrib.auth.apps",
        "django.contrib.contenttypes.apps",
        "django.contrib.sessions.apps",
        "django.contrib.messages.apps",
        "django.contrib.staticfiles.apps",
        "django.db.backends.sqlite3",
        "rest_framework.apps",
        "corsheaders.apps",
        "counter.apps",
    ]
)

for package in STRING_REFERENCED:
    if not collect_submodules(package):
        raise SystemExit(
            f"collect_submodules found nothing for {package!r}. The build "
            f"would succeed and then die with ModuleNotFoundError. Run "
            f"pyinstaller from {PROJECT} with its virtualenv active.")

a = Analysis(
    # app_main.py, not counter/desktop.py: the entry script runs as
    # __main__, and counter.desktop uses relative imports.
    ["app_main.py"],
    pathex=[str(PROJECT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Torch arrives with ultralytics during export_model and is not needed
    # at run time; left in, it adds gigabytes.
    excludes=["torch", "torchvision", "ultralytics", "matplotlib",
              "tkinter", "PySide6", "PyQt5", "PyQt6", "IPython", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VisitorsTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window behind the app. Errors before the window opens go
    # nowhere, so the app writes its own log into the data directory.
    console=False,
    disable_windowed_traceback=False,
    icon=str(PROJECT / "assets" / "AppIcon.ico"),
)
