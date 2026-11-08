#!/usr/bin/env bash
# Export a YOLOv8 person detector to ONNX for VisitorsTracker.
#
# The export needs torch and ultralytics, which are large and are NOT runtime
# dependencies: the app reads the .onnx through cv2.dnn. So this builds a
# throwaway venv, exports, copies the model in, and leaves the project venv
# untouched.
#
#   ./scripts/export_yolo.sh            # yolov8n (fastest, default)
#   ./scripts/export_yolo.sh yolov8s    # small - more accurate, slower
#   ./scripts/export_yolo.sh yolov8m    # medium

set -euo pipefail

MODEL="${1:-yolov8n}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$PROJECT_DIR/counter/ml_models"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"

echo "==> building export environment in $WORK"
"$PYTHON" -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --upgrade pip -q
"$WORK/venv/bin/python" -m pip install -q ultralytics onnx onnxslim onnxruntime

echo "==> exporting $MODEL to ONNX"
cd "$WORK"
YOLO_CONFIG_DIR="$WORK/cfg" "$WORK/venv/bin/python" - "$MODEL" <<'PY'
import sys
from ultralytics import YOLO

name = sys.argv[1]
path = YOLO(f"{name}.pt").export(format="onnx", opset=12, imgsz=640, simplify=False)
print(path)
PY

mkdir -p "$DEST"
cp "$WORK/$MODEL.onnx" "$DEST/$MODEL.onnx"
echo "==> wrote $DEST/$MODEL.onnx"

if [ "$MODEL" != "yolov8n" ]; then
  echo
  echo "Point the app at it with:"
  echo "    export COUNTER_ONNX_MODEL=$MODEL.onnx"
fi
echo "Done. The detector picks it up automatically (COUNTER_DETECTOR=auto)."
