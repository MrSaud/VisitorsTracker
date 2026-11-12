#!/bin/bash
# Freeze VisitorsTracker into a self-contained .app and wrap it in a DMG.
#
# The counterpart of installer/VisitorsTracker.iss on Windows. What comes out
# needs no Python, no virtualenv and no project folder on the receiving Mac.
#
# PyInstaller does not cross-compile, and OpenCV publishes no universal2
# wheel, so macOS needs one build per chip. Set PYTHON to a virtualenv of the
# architecture you are building for; on Apple Silicon an x86_64 interpreter
# under `arch -x86_64` produces the Intel app, which Rosetta can then run and
# test on the same machine.
#
# Do not point PYTHON at Homebrew's interpreter. Homebrew builds for the host,
# so the app would silently refuse to start on any older macOS. The check
# below reports the real floor and warns when that has happened.
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
DMG=""   # named after the built binary's architecture, once it exists

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

ARCH="$(lipo -archs "$APP/Contents/MacOS/VisitorsTracker" | awk '{print $1}')"
DMG="$OUT/VisitorsTracker-$VERSION-$ARCH.dmg"

echo "==> deployment target"
# Every bundled library carries the macOS version it was built against, and
# the app can only run on the highest of them. Guessing this wrong is worse
# than useless: the plist said 11.0 while Homebrew's Python made the real
# floor 26.0, so the app would have been offered to Macs it cannot start on.
# Two load commands carry this. Binaries built against a modern SDK use
# LC_BUILD_VERSION/minos; ones built against an older SDK -- which is exactly
# what the compatible wheels are -- use LC_VERSION_MIN_MACOSX/version. Reading
# only the first finds nothing in the wheels that matter most here.
MIN="$(find "$APP/Contents" -type f \( -name '*.so' -o -name '*.dylib' \
             -o -name 'Python' -o -name 'VisitorsTracker' \) -print0 \
       | xargs -0 -n 20 otool -l 2>/dev/null \
       | awk '/LC_BUILD_VERSION/      {mode = "build";   next}
              /LC_VERSION_MIN_MACOSX/ {mode = "version"; next}
              mode == "build"   && /minos/   {print $2; mode = ""}
              mode == "version" && /version/ {print $2; mode = ""}' \
       | sort -t. -k1,1n -k2,2n | tail -1)"
[ -n "$MIN" ] || { echo "Could not read a deployment target from the bundle" >&2; exit 1; }
/usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $MIN" "$APP/Contents/Info.plist"
echo "    $ARCH, runs on macOS $MIN and newer"

HOST_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
MIN_MAJOR="${MIN%%.*}"
if [ "$MIN_MAJOR" -ge "$HOST_MAJOR" ]; then
  echo
  echo "    WARNING: that floor is this machine's own macOS version, which"
  echo "    means the interpreter was built for this host -- Homebrew does"
  echo "    that. The app will not start on any older Mac. Build against a"
  echo "    relocatable interpreter instead and set PYTHON to it:"
  echo "      https://github.com/astral-sh/python-build-standalone"
  echo
fi

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
# The readme is shared by both builds, so say at the top which one this is.
# A person holding the wrong image otherwise finds out by double-clicking it.
{
  case "$ARCH" in
    arm64)  echo "This build is for Apple Silicon Macs (M1 and later)."
            OTHER="VisitorsTracker-$VERSION-x86_64.dmg (Intel)" ;;
    x86_64) echo "This build is for Intel Macs."
            OTHER="VisitorsTracker-$VERSION-arm64.dmg (Apple Silicon)" ;;
    *)      echo "This build is for $ARCH Macs."
            OTHER="" ;;
  esac
  echo "It needs macOS $MIN or newer."
  [ -n "$OTHER" ] && echo "For the other kind of Mac, use $OTHER."
  echo
  echo "To check which Mac you have: Apple menu > About This Mac."
  echo
  cat "$PROJECT/installer/README-macos.txt"
} > "$STAGE/Read me first.txt"
rm -f "$DMG"
hdiutil create -quiet -volname "VisitorsTracker $VERSION" -srcfolder "$STAGE" \
        -ov -format UDZO "$DMG"

echo
echo "Built $DMG ($(du -h "$DMG" | cut -f1))"
echo "  $ARCH Macs only, macOS $MIN or newer"
echo
echo "On the Mac you give it to, macOS will refuse it on first open because"
echo "it is not notarised. The way through, on every version:"
echo "  copy it to /Applications, double-click, Cancel the warning, then"
echo "  System Settings > Privacy & Security > Open Anyway."
echo "The old right-click > Open shortcut was removed in macOS 15."
