# Detector models

`COUNTER_DETECTOR` defaults to `auto`:

- a YOLOv8 ONNX model in this folder is used when present,
- otherwise OpenCV's built-in HOG pedestrian detector runs, so the project
  works immediately after `pip install` with nothing in here.

## Exporting a model

```bash
python manage.py export_model          # yolov8n, the default
python manage.py export_model yolov8s  # more accurate, slower
```

The script builds a temporary venv for `ultralytics` + `torch` and throws it
away afterwards. Those are **export-time only** — at runtime the model is read
through `cv2.dnn`, so the project venv stays small.

For a non-default model, point the app at it:

```bash
export COUNTER_ONNX_MODEL=yolov8s.onnx
```

## Notes

- `.onnx` and `.pt` files are gitignored; they are build artefacts, so export
  them per machine rather than committing ~12MB of weights.
- A missing or corrupt model never takes the cameras down: the loader logs a
  warning and falls back to HOG.
- `people.jpg`, if present, is a local fixture for the detection test and is
  also gitignored. `manage.py export_model` does not fetch it.
