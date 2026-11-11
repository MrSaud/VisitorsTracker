#!/bin/bash
# Freeze VisitorsTracker into a self-contained .app and wrap it in a DMG.
#
# The counterpart of installer/VisitorsTracker.iss on Windows. What comes out
# needs no Python, no virtualenv and no project folder on the receiving Mac.
#
# PyInstaller does not cross-compile: run this on a Mac, and on the same
# architecture you intend to ship to. OpenCV publishes no universal wheel, so
# an Apple Silicon build is arm64-only and will not run on an Intel Mac.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
OUT="${OUT:-$PROJECT/dist}"
WORK="${WORK:-$PROJECT/build}"
APP="$OUT/VisitorsTracker.app"
VERSION="$("$PYTHON" - <<'PY'
import re, pathlib
spec = pathlib.Path("VisitorsTracker.spec").read_text()
print(re.search(r'APP_VERSION = "([^"]+)"', spec).group(1))
PY
)"
DMG="$OUT/VisitorsTracker-$VERSION-$(uname -m).dmg"

[ "$(uname -s)" = "Darwin" ] || { echo "This builds the Mac app; run it on a Mac." >&2; exit 1; }
[ -x "$PYTHON" ] || { echo "No interpreter at $PYTHON. Set PYTHON=..." >&2; exit 1; }

echo "==> detector weights"
# Exporting pulls ultralytics and torch, a download of a gigabyte or so, so
# this asks rather than doing it behind your back.
[ -f counter/ml_models/yolov8n.onnx ] || {
  echo "counter/ml_models/yolov8n.onnx is missing. Run:" >&2
  echo "    $PYTHON manage.py export_model" >&2
  exit 1
}

echo "==> freezing (a few minutes)"
"$PYTHON" -m PyInstaller --noconfirm --distpath "$OUT" --workpath "$WORK" \
    VisitorsTracker.spec

[ -d "$APP" ] || { echo "PyInstaller produced no .app at $APP" >&2; exit 1; }

echo "==> signing"
# Ad-hoc: enough for the app to hold a camera grant of its own, which is why
# it has to be a bundle at all. Not enough for Gatekeeper on another Mac --
# that wants an Apple Developer identity and notarisation.
codesign --force --deep --sign - \
         --identifier com.visitorstracker.desktop "$APP"
codesign --verify --deep --strict "$APP"

echo "==> checking it carries its own camera permission string"
/usr/libexec/PlistBuddy -c "Print :NSCameraUsageDescription" \
    "$APP/Contents/Info.plist" > /dev/null

echo "==> disk image"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp "$PROJECT/installer/README-macos.txt" "$STAGE/Read me first.txt"
rm -f "$DMG"
hdiutil create -quiet -volname "VisitorsTracker $VERSION" -srcfolder "$STAGE" \
        -ov -format UDZO "$DMG"

echo
echo "Built $DMG ($(du -h "$DMG" | cut -f1), $(uname -m) only)"
echo
echo "On the Mac you give it to, macOS will refuse it on first open because"
echo "it is not notarised. Right-click the app > Open > Open, once."
