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

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT = Path(SPECPATH)
APP_VERSION = "1.0.0"

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

# The admin's CSS. Django serves static files itself only while DEBUG is
# on, and this app turns DEBUG off, so it serves them through the
# staticfiles finders instead -- which read from the app directories, and
# those have to be in the bundle. Without this the admin loads with no
# styling at all.
datas += collect_data_files("django.contrib.admin")
datas += collect_data_files("django.contrib.auth")
datas += collect_data_files("rest_framework")

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

# Windows gets one file: the installer wants a single executable, and the
# unpack-to-temp cost is paid once at launch.
#
# macOS gets a directory inside the .app instead. Onefile there unpacks its
# libraries to a temporary folder outside the bundle, where the bundle's
# signature does not reach -- PyInstaller warns that this "clashes with
# macOS's security" and makes it an error in version 7. Measured here, it
# also cost 35 to 65 seconds per launch unpacking 80MB every time.
ONEFILE = sys.platform.startswith("win")

exe = EXE(
    pyz,
    a.scripts,
    *([a.binaries, a.datas] if ONEFILE else []),
    [],
    exclude_binaries=not ONEFILE,
    name="VisitorsTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window behind the app. Errors before the window opens go
    # nowhere, so the app writes its own log into the data directory.
    console=False,
    disable_windowed_traceback=False,
    icon=str(PROJECT / "assets" / ("AppIcon.ico" if sys.platform.startswith("win")
                                   else "AppIcon.icns")),
)

if not ONEFILE:
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="VisitorsTracker",
    )

# macOS wants an .app, and not only for appearance: the camera grant
# belongs to an application, so a bare executable has no identity to hold
# one and cannot open a webcam at all. The thin bundle from
# `manage.py build_app` borrows the project's virtualenv and only works on
# the machine that built it; this one carries its own Python and can be
# handed to somebody else.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="VisitorsTracker.app",
        icon=str(PROJECT / "assets" / "AppIcon.icns"),
        bundle_identifier="com.visitorstracker.desktop",
        version=APP_VERSION,
        info_plist={
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            # Without this macOS terminates the process the moment it
            # touches the camera, rather than asking.
            "NSCameraUsageDescription": (
                "VisitorsTracker counts people crossing a line in the "
                "camera's view. Video is processed on this machine and is "
                "not recorded or uploaded."
            ),
            # It serves its own dashboard on 127.0.0.1; nothing else.
            "LSApplicationCategoryType": "public.app-category.utilities",
            # One window, and closing it quits; no reason to linger in the
            # Dock with nothing to show.
            "LSMultipleInstancesProhibited": True,
        },
    )
