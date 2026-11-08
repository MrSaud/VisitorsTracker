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

Tracks are positional and short-lived, so the same person returning later is
a new track and counts again: **`in` is entry crossings, not unique people**.

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

## Occupancy and the daily reset

`inside` is **not** all-time `in - out`. Every counter misses some crossings,
and a missed exit is never corrected, so an all-time figure drifts upward
forever:

| Missed exits | After 1 week | After 30 days |
|---|---|---|
| 1% | reads 21 | reads 90 |
| 3% | reads 63 | reads 270 |
| 5% | reads 105 | reads 450 |

…at 300 visitors a day, with the building empty. Within a month the headline
number would be pure accumulated error.

So occupancy is counted from the most recent daily reset, set by
`COUNTER_OCCUPANCY_RESET_AT` (default `00:00`). Point it at a time the
building is reliably empty; the error can then never exceed one day's worth.

- `totals` stays lifetime footfall, which is what you want for reporting.
- `occupancy.inside` is clamped at zero, but `occupancy.net` is reported
  unclamped alongside it. A persistently negative net means exits outnumber
  entries, which indicates a reversed camera rather than an empty room.
- Anyone still inside across the reset is not carried over. If your site
  genuinely operates overnight, set the reset to its quietest hour.

## Authentication

Reads are open by default so a wall display needs no credential. Anything
that changes state — starting or stopping a camera, and especially
`reset`, which deletes history — requires an operator.

Two ways to be one:

```bash
# 1. A staff session, for humans. The dashboard shows controls only to staff.
python manage.py createsuperuser

# 2. A shared token, for machines.
export COUNTER_API_TOKEN='a-long-random-value'
curl -X POST -H 'Authorization: Token a-long-random-value' \
     http://127.0.0.1:8000/api/v1/cameras/1/stop/
```

| Variable | Default | Purpose |
|---|---|---|
| `COUNTER_API_TOKEN` | *(unset)* | Shared token for machine clients. Unset means no machine client can write; staff sessions still can |
| `COUNTER_PUBLIC_READS` | `1` | Set to `0` to require authentication for reads and the video stream too |

An unset `COUNTER_API_TOKEN` rejects every token rather than accepting any,
so a missing environment variable fails closed.

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
| `COUNTER_OCCUPANCY_RESET_AT` | `00:00` | Local time occupancy returns to zero |
| `COUNTER_STALE_FRAME_SECONDS` | `30` | No frame for this long means stalled |
| `COUNTER_API_TOKEN` | *(unset)* | Shared token for machine write access |
| `COUNTER_PUBLIC_READS` | `1` | `0` requires auth for reads as well |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health/` | Whether cameras are working — see [Health](#health) |
| `GET` | `/api/v1/stats/` | Totals across all cameras, plus per-camera detail |
| `GET` | `/api/v1/daily/` | Crossings per date — see [Visitors per date](#visitors-per-date) |
| `GET` | `/api/v1/hourly/` | Arrivals by hour of day — see [Busiest hours](#busiest-hours) |
| `GET` | `/api/v1/cameras/` | List cameras |
| `GET` | `/api/v1/cameras/<id>/` | One camera |
| `POST` | `/api/v1/cameras/<id>/start/` | Start its worker |
| `POST` | `/api/v1/cameras/<id>/stop/` | Stop its worker |
| `GET` | `/api/v1/cameras/<id>/events/?limit=50` | Recent crossings |
| `POST` | `/api/v1/cameras/<id>/reset/` | Delete recorded crossings |
| `GET` | `/api/v1/cameras/<id>/stream/` | Annotated MJPEG live stream |
| `GET` | `/api/v1/cameras/<id>/snapshot/` | Latest annotated frame as JPEG |

## Visitors per date

The dashboard's **Visitors per date** panel charts daily crossings over 7, 30
or 90 days, with a totals strip, a busiest-day figure and a CSV download. The
same data is available directly:

```bash
curl 'http://127.0.0.1:8000/api/v1/daily/?days=7'
curl 'http://127.0.0.1:8000/api/v1/daily/?start=2026-03-01&end=2026-03-31'
curl 'http://127.0.0.1:8000/api/v1/daily/?camera=1&days=30'
curl -OJ 'http://127.0.0.1:8000/api/v1/daily/?days=30&format=csv'
```

| Parameter | Meaning |
|---|---|
| `days` | Window size, default 30, capped at 730. Combines with `start` or `end` |
| `start` / `end` | Explicit `YYYY-MM-DD` window, inclusive at both ends |
| `camera` | Restrict to one camera id |
| `format=csv` | Download as CSV instead of JSON |

Response:

```json
{
  "start": "2026-11-02", "end": "2026-11-08",
  "summary": {"in": 190, "out": 157, "net": 33,
              "busiest_date": "2026-11-02", "busiest_in": 42,
              "days": 7, "daily_average_in": 27.14},
  "days": [{"date": "2026-11-02", "in": 42, "out": 34, "net": 8}]
}
```

Two things worth knowing:

- **Dates are local**, using the project's `TIME_ZONE` (`Asia/Kuwait` by
  default). Events are stored as UTC instants, so a visitor arriving at 01:30
  local belongs to that day, not to the previous UTC one.
- **Quiet days are included as zeros** rather than omitted, so a chart or
  spreadsheet does not silently skip them.

Remember that `in` counts *entry crossings*, not distinct people — someone
entering three times adds three. See the note on identity under
[How counting works](#how-counting-works).

## Busiest hours

Per-date answers *how many*. This answers *when* — which hours need staff on
the desk. The dashboard's **Busiest hours** panel shows the top three hours,
the span arrivals actually fall in, a 24-hour bar chart and a weekday-by-hour
heatmap.

```bash
curl 'http://127.0.0.1:8000/api/v1/hourly/?days=30'
curl 'http://127.0.0.1:8000/api/v1/hourly/?days=30&matrix=1'
curl -OJ 'http://127.0.0.1:8000/api/v1/hourly/?days=30&format=csv'
```

Takes the same window parameters as `/daily/` (`days`, `start`, `end`,
`camera`, `format=csv`), plus `matrix=1` for the weekday breakdown.

```json
{
  "busiest": [{"hour": 10, "label": "10:00", "in": 476, "average_in": 15.87, "share": 27.9}],
  "opening_span": {"first_hour": 8, "last_hour": 14},
  "hours": [{"hour": 0, "label": "00:00", "in": 0, "out": 0, "average_in": 0.0, "share": 0.0}]
}
```

**Which average to trust.** `average_in` on the profile divides by *every*
day in the range, closed days included. Over a range spanning weekends that
understates a working-day peak — a 30-day window covering 8 weekend days
showed 15.87 at the 10:00 peak where working days actually average ~21.

`matrix=1` fixes that: each cell divides by how many times that weekday
actually occurred, so a Tuesday average is not diluted by Fridays. Use the
profile for the shape of a day, and the matrix for staffing a given weekday.

Hours are local, like dates — an arrival at 01:30 Kuwait time is hour 1, not
the UTC hour 22.

## Health

The dangerous failure here is the quiet one. A camera that dies on a Thursday
night raises nothing — the counts simply stop, and Sunday's report is wrong
without looking wrong. A frozen stream is worse: the worker stays alive and
"running" while delivering no frames at all.

So health is judged on evidence of work, not on whether something was
started. The dashboard shows a banner when anything is wrong.

```bash
curl 'http://127.0.0.1:8000/api/v1/health/'
python manage.py check_health
```

| State | Meaning | Healthy? |
|---|---|---|
| `ok` | Alive and delivering frames | ✅ |
| `stalled` | Alive, but no frame for `COUNTER_STALE_FRAME_SECONDS` (default 30) | ❌ |
| `failed` | Stopped with an error — the reason is reported | ❌ |
| `offline` | Marked active, but nothing is running. The post-restart case | ❌ |
| `disabled` | Not active, so nothing is expected | ✅ |
| `finished` | A video file played to its end | ✅ |

Those roll up into one word: `ok`, `degraded` (something stalled) or `down`
(something failed or is offline).

Two deliberate choices, both to avoid alarms nobody will trust:

- **Deactivating a camera clears a stale failure.** Turning one off means
  stop expecting it to run; an error from an earlier run must not alarm
  forever.
- **A finished video file is not a fault.** Live cameras never end on their
  own, so this cannot mask a real failure.

### For monitoring

`/api/v1/health/` returns 200 regardless so a dashboard can poll it. Pass
`?strict=1` to get **503** when anything is wrong, which uptime checks can
alert on.

`check_health` is the cron-friendly form, exiting **0** healthy, **1**
degraded, **2** down:

```bash
*/5 * * * * cd /path/to/VisitorsTracker && .venv/bin/python manage.py check_health --quiet
```

Note it only sees workers in **its own process**, so run it against the
server's own health endpoint if counting runs elsewhere.

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

159 tests covering tracker identity, line-crossing direction, the dead band,
the YOLOv8 output decoding, line-setup suggestions, local-date and local-hour
bucketing, occupancy reset behaviour, camera health states, authentication
and the API. Most use a stub detector, so they need
no camera and no model file; the tests that exercise the real model skip
automatically when it has not been exported.

## Notes

- Workers live in the web server process, so `runserver --noreload` (or a
  single Gunicorn worker) keeps one set of cameras. Across multiple Gunicorn
  workers each process would open its own capture; run counting as a separate
  `run_counter` process in that case.
- `db.sqlite3` is gitignored — run `migrate` after cloning.
