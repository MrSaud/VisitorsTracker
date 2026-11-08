# VisitorsTracker

Camera-based visitor counting in Django. A background worker reads each
camera, detects people, tracks them across frames and counts them as they
cross a virtual line — giving you **in**, **out** and **inside now**.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open http://127.0.0.1:8000/ for the dashboard.

## Adding a camera

Add one in the admin at `/admin/counter/camera/add/`, or from the shell:

```bash
python manage.py run_counter --source 0 --name "Front door"
```

`source` accepts:

| Value | Meaning |
|---|---|
| `0`, `1`, … | Local webcam index |
| `/path/to/clip.mp4` | Video file (useful for testing) |
| `rtsp://user:pass@host/stream` | IP camera |

### Camera permission on macOS

The first time a webcam is opened, macOS asks the process for camera access.
If frames never arrive, grant access to the app running the server — Terminal,
iTerm or your editor — under **System Settings > Privacy & Security > Camera**,
then restart the server. The worker reports this as an error on the dashboard
rather than failing silently.

## How counting works

1. **Detect** — a YOLOv8 ONNX model when one is present, otherwise OpenCV's
   built-in HOG pedestrian detector. See [Detector](#detector) below.
2. **Track** — a nearest-centroid tracker assigns a stable id to each person,
   so one visitor is counted once rather than once per frame.
3. **Count** — each track's centroid is compared against the counting line.
   A side change outside a small dead band emits one `CountEvent`, with
   `direction` of `in` or `out`.

The dead band (`LineCounter.margin`) is what stops someone loitering on the
line from racking up counts. Set `invert_direction` on a camera if in and out
come out backwards for its mounting angle.

Detection runs on every *N*th frame (`COUNTER_DETECT_EVERY`, default 3) and
tracking fills the gaps, which is what makes CPU-only operation workable.

## Detector

`COUNTER_DETECTOR=auto` (the default) uses a YOLOv8 ONNX model when one exists
in `counter/ml_models/`, and falls back to HOG otherwise. Export one with:

```bash
./scripts/export_yolo.sh            # yolov8n
./scripts/export_yolo.sh yolov8s    # more accurate, slower
```

`ultralytics` and `torch` are needed only to *export*; the script builds a
throwaway venv for them and discards it. At runtime the model is read through
`cv2.dnn`, so they are not in `requirements.txt`.

Measured on the project's test photo (four people, Apple silicon, 640x480
frames):

| Detector | People found | Per frame | Effective video fps at `DETECT_EVERY=3` |
|---|---|---|---|
| HOG | 1 of 4 | ~12 ms | ~250 |
| YOLOv8n | 4 of 4 | ~41 ms | ~74 |

HOG is roughly 3x faster but misses most people at an angle; YOLOv8n is the
right default for a real entrance, and still has headroom well beyond camera
frame rates. A missing or corrupt model never stops the cameras — the loader
logs a warning and uses HOG.

## Tuning

All of these are environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `COUNTER_DETECTOR` | `auto` | `auto`, `onnx` or `hog` |
| `COUNTER_ONNX_MODEL` | `yolov8n.onnx` | Model file in `counter/ml_models/` |
| `COUNTER_MIN_CONFIDENCE` | *(per backend)* | Detection threshold. Unset, HOG uses 0.45 and YOLO 0.35 — HOG scores are SVM decision values, YOLO scores are probabilities, so one shared number suits neither |
| `COUNTER_DETECT_EVERY` | `3` | Detect on every Nth frame |
| `COUNTER_MAX_DISAPPEARED` | `25` | Frames a track survives unseen |
| `COUNTER_FRAME_WIDTH` | `640` | Frames are downscaled to this width before detection |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/stats/` | Totals across all cameras, plus per-camera detail |
| `GET` | `/api/v1/cameras/` | List cameras |
| `GET` | `/api/v1/cameras/<id>/` | One camera |
| `POST` | `/api/v1/cameras/<id>/start/` | Start its worker |
| `POST` | `/api/v1/cameras/<id>/stop/` | Stop its worker |
| `GET` | `/api/v1/cameras/<id>/events/?limit=50` | Recent crossings |
| `POST` | `/api/v1/cameras/<id>/reset/` | Delete recorded crossings |
| `GET` | `/api/v1/cameras/<id>/stream/` | Annotated MJPEG live stream |
| `GET` | `/api/v1/cameras/<id>/snapshot/` | Latest annotated frame as JPEG |

## "It never counts anything"

Run the probe. It separates the two causes, from your real camera:

```bash
python manage.py camera_probe --source 0 --seconds 20
```

Walk through the view the way a visitor would. It reports how often a person
was detected, how much of the frame they filled, which way they travelled,
and the counting line that suits that movement:

```
frames with a person : 80 (100%)
body fills on average: 44% of frame height
  -> good framing for detection.

Suggested counting line
  movement was mostly  : left-to-right
  line orientation     : vertical
  line position        : 0.38
  you moved rightwards, which currently counts as IN
```

Apply it straight to a camera:

```bash
python manage.py camera_probe --camera 1 --seconds 20 --apply
```

**0% detected** is a framing problem and no line setting will fix it — see
the camera placement rules below. **Detected but never counted** means the
line is in the wrong place or the wrong orientation.

## Camera placement

The detector is trained on whole people, so framing decides whether anything
works at all:

| | |
|---|---|
| **Distance** | People should fill roughly 30-60% of frame height — head to knees at least. A head-and-shoulders webcam shot is the most common reason nothing is detected |
| **Height** | Chest height or above, angled slightly down. Eye-level straight-on is fine; floor-level is not |
| **Angle** | Let people cross the frame, not walk straight at the lens. A body moving across gives the tracker a clear path over the line |
| **Lighting** | Front-lit. A camera pointed at a bright window turns everyone into a silhouette |
| **Line** | Place it where people genuinely cross, away from where they stop and loiter |

Direction convention: with a **horizontal** line, moving *down* the frame is
IN; with a **vertical** line, moving *right* is IN. If that is backwards for
your mounting, tick **invert direction** on the camera.

## Headless counting

To count without the web server:

```bash
python manage.py run_counter --all
python manage.py run_counter --camera 1 --seconds 60
python manage.py run_counter --source clip.mp4 --name "Test clip"
```

## Tests

```bash
python manage.py test
```

53 tests covering tracker identity, line-crossing direction, the dead band,
the YOLOv8 output decoding, line-setup suggestions and the API. Most use a stub detector, so they need
no camera and no model file; the tests that exercise the real model skip
automatically when it has not been exported.

## Notes

- Workers live in the web server process, so `runserver --noreload` (or a
  single Gunicorn worker) keeps one set of cameras. Across multiple Gunicorn
  workers each process would open its own capture; run counting as a separate
  `run_counter` process in that case.
- `db.sqlite3` is gitignored — run `migrate` after cloning.
