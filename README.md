# VisitorsTracker

Camera-based visitor counting in Django. A background worker reads each
camera, detects people, tracks them across frames and counts them as they
cross a virtual line — giving you **in**, **out** and **inside now**.

Counting can be scoped to a scheduled **event**: create one in the admin and
crossings are attributed to it for as long as it runs. See [Events](#events).

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

macOS grants camera access to the **application that owns the process**, not
to Python. So the question is never "can this script use the camera", it is
"is the app that launched it allowed to". The worker reports a refusal as an
error on the dashboard rather than failing silently:

```
Could not open video source 0. Check that the process running this has camera
access in System Settings > Privacy & Security > Camera, and that no other app
is holding the device.
```

Grant access to the launching app under **System Settings > Privacy &
Security > Camera**, then restart the server.

**The trap:** that pane lists only apps that have already asked, and it has
no button to add one. An editor or assistant whose request macOS rejected
outright never appears, so there is nothing to switch on and no way to put it
there. A server started from inside one can never open a webcam.

Terminal is normally in the list already, so launch it from Terminal instead:

```bash
open -a Terminal scripts/run_macos_webcam.command
```

That script starts the server with `COUNTER_AUTOSTART=1` under Terminal, which
owns the permission. None of this applies to RTSP or HTTP cameras — only local
capture devices sit behind that prompt.

## How counting works

1. **Detect** — a YOLOv8 ONNX model when one is present, otherwise OpenCV's
   built-in HOG pedestrian detector. See [Detector](#detector) below.
2. **Track** — a SORT-style tracker assigns a stable id to each person, so
   one visitor is counted once rather than once per frame. See
   [Tracking](#tracking).
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

## Tracking

Detection finds people in a single frame; tracking is what links them
between frames, and it decides whether a visitor is counted once, twice or
not at all.

The default (`COUNTER_TRACKER=sort`) combines three things:

- **Motion.** Each track runs a constant-velocity Kalman filter, so it is
  matched against where it is *predicted* to be, not where it last was.
- **Shape.** Matching uses box overlap, not just centre distance, so a near
  person and a far one are not confused.
- **Optimality.** Assignment is solved globally (Hungarian), so one
  confident early match cannot force a cascade of wrong ones.

Where the detector reports calibrated probabilities (YOLO), association runs
in two stages: confident detections first, then a second pass that lets
unmatched tracks claim the weak detections a partly occluded person
produces, instead of dropping them.

### Why not nearest-centroid

The old tracker matched on centre distance alone. When two people pass each
other — routine at a doorway — each detection is briefly nearest the *other*
person's last position, and the identities swap. A swap at the counting line
cancels both crossings.

Measured on a two-way pass, one person entering and one leaving:

| Tracker | Counted |
|---|---|
| Truth | 1 in, 1 out |
| Nearest-centroid | **0 in, 0 out** |
| SORT | 1 in, 1 out |

And on a clip of three real people crossing the line once:

| Tracker | Counted | Identities created |
|---|---|---|
| Truth | 3 in | 3 |
| Nearest-centroid | **4 in** | 4 |
| SORT | 3 in | 3 |

`COUNTER_TRACKER=centroid` restores the old behaviour if ever needed.

### Cost

Association only, excluding detection:

| People in frame | Centroid | SORT |
|---|---|---|
| 3 | 0.03 ms | 0.16 ms |
| 8 | 0.08 ms | 0.34 ms |
| 20 | 0.23 ms | 0.96 ms |

Against ~41 ms for YOLOv8n detection, tracking is a rounding error.

### A note on matching

Pure IoU association fails once a person moves about their own box width
between detections — overlap collapses towards zero — which happens at any
realistic walking pace with `COUNTER_DETECT_EVERY` above 1. So overlap is
combined with centre proximity measured in box-diagonals, hard-gated at
twice the diagonal so the distance term cannot pair distant strangers.

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

## Peak occupancy

`inside` answers "how many are in the room **now**", which is no use the
morning after. A hall can breach its capacity at 11:40 and read comfortably
by the time anyone looks — the figure that matters afterwards is the
high-water mark.

Every report carries it. From a seeded week:

| Date | In | Out | Peak |
|---|---|---|---|
| 2026-11-02 | 30 | 30 | **13** |
| 2026-11-08 | 46 | 44 | **46** |

Similar footfall, completely different crowding: on the 2nd people trickled
through, on the 8th they all came at once. Footfall alone cannot tell those
apart.

It is walked from the crossing log — plus one for an entry, minus one for an
exit — with two deliberate rules:

- The running count **resets at the daily boundary** and is clamped at zero,
  exactly as `occupancy` is, so a missed exit cannot inflate every later day.
- Peaks are bucketed by the date the crossing happened on, so the column
  lines up with the `in`/`out` beside it even when the reset is not midnight.

A multi-day event is therefore the **worst day**, not the sum of its
mornings: ten inside at once on each of three days is a peak of 10, not 30.

Where it shows up:

- `stats.today.peak` — the `Peak today` tile.
- `events[].peak` and `capacity.peak` / `peak_percent` / `breached` — the
  event card shows a red mark on the capacity bar at the level reached, so
  an almost-empty bar cannot imply the room was never full.
- `daily[].peak` and `summary.peak_inside` — the per-date table and CSV.

`capacity.breached` is worth alerting on. "Fine when you looked" is not the
same as "never over".

## Staying up

Counting that needs somebody to press Start is counting that will be missing
after a power cut. Three separate things can stop it, and each has its own
answer.

### The source drops

RTSP streams drop for ordinary reasons: a network blip, a switch hiccup, the
camera rebooting its firmware. The worker reconnects on its own, waiting 1s,
2s, 4s … up to 30s between attempts, and **never gives up** by default
(`COUNTER_RECONNECT_ATTEMPTS=0`). Set a number to stop trying after that many
consecutive failures.

Across the gap the counting session continues — `in`/`out` keep their values
— but **every track is thrown away**. This matters: the scene moved on while
we were away, and a surviving track would be matched to whoever now stands
nearest, recording a crossing from one side to the other that nobody made.
Tests cover both halves: the same walk past the line counts 1 when the stream
is continuous and 0 when the stream dropped in the middle.

While reconnecting, health reports `reconnecting` (degraded, not down) with
the reason and the time until the next attempt.

> A **URL source is live, a path is a file.** Running out of frames is a
> normal finish for a clip and a fault for a camera, so getting this backwards
> would file a dead IP camera as healthy and never reconnect to it.

### The process goes away

Workers live in the process that started them, so a restart takes all of them
with it, and `/api/v1/health/` reports `offline` — correct, but after the
fact. With autostart the server brings up every active camera itself:

```bash
COUNTER_AUTOSTART=1 .venv/bin/python manage.py runserver 0.0.0.0:8000 --noreload
```

A supervisor thread then re-checks every `COUNTER_SUPERVISE_INTERVAL` seconds
(default 10) and reconciles what is running against what the database says:

- an active camera with no live worker is started;
- a deactivated or deleted camera is stopped;
- a camera that keeps dying is left alone for 30s between restarts, so a
  broken configuration cannot spin;
- a **video file that played to its end is left finished** — replaying a demo
  clip on a timer would invent visitors for ever.

Autostart is off by default, and never runs under `migrate`, `test`,
`run_counter` and the rest, nor in the reloader's parent process. Importing
the app should not take hold of a camera.

To keep counting in a process of its own instead:

```bash
.venv/bin/python manage.py supervise
```

Pick **one** home for counting — the server with autostart, or `supervise`.
Workers are per-process, so the dashboard's Start button acts on the server's
workers, and two processes would fight over the device.

### The machine reboots

`deploy/com.visitorstracker.counter.plist` is a launchd agent that starts the
server at login with autostart on, and restarts it if it exits:

```bash
sed 's|__PROJECT__|'"$PWD"'|g' deploy/com.visitorstracker.counter.plist \
  > ~/Library/LaunchAgents/com.visitorstracker.counter.plist
launchctl load -w ~/Library/LaunchAgents/com.visitorstracker.counter.plist
```

It must be a **LaunchAgent, not a LaunchDaemon**: macOS grants camera access
per user, and a daemon running as root gets no camera at all.

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
| `COUNTER_RECONNECT_ATTEMPTS` | `0` | Consecutive reconnection attempts before giving up on a live camera. `0` means never give up |
| `COUNTER_AUTOSTART` | `0` | `1` starts every active camera with the server and keeps them running |
| `COUNTER_SUPERVISE_INTERVAL` | `10` | Seconds between supervisor checks |
| `COUNTER_TRACKER` | `sort` | `sort` or `centroid` |
| `COUNTER_API_TOKEN` | *(unset)* | Shared token for machine write access |
| `COUNTER_PUBLIC_READS` | `1` | `0` requires auth for reads as well |
| `COUNTER_ALERTS_ENABLED` | `0` | `1` turns alerting on |
| `COUNTER_ALERT_WEBHOOK` | *(unset)* | Slack/Teams-compatible incoming webhook |
| `COUNTER_ALERT_EMAILS` | *(unset)* | Comma-separated recipients; needs the `DJANGO_EMAIL_*` settings |
| `COUNTER_ALERT_AFTER_SECONDS` | `60` | How long a problem must persist before anyone is told |
| `COUNTER_ALERT_MIN_SEVERITY` | `1` | `1` alerts on degraded too, `2` only on down |
| `COUNTER_ALERT_INTERVAL` | `30` | Seconds between alert checks |
| `COUNTER_SITE_NAME` | `VisitorsTracker` | Named in every message, for people with two venues |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/events/` | Scheduled events, with attendance |
| `GET` | `/api/v1/events/current/` | What is running right now |
| `GET` | `/api/v1/events/<id>/` | One event, with its hourly profile |
| `GET` | `/api/v1/health/` | Whether cameras are working — see [Health](#health) |
| `GET` | `/api/v1/alerts/` | Alert history, open first — see [Alerts](#alerts) |
| `GET` | `/api/v1/stats/` | Totals across all cameras, plus per-camera detail |
| `GET` | `/api/v1/daily/` | Crossings per date and daily peaks — see [Visitors per date](#visitors-per-date) |
| `GET` | `/api/v1/hourly/` | Arrivals by hour of day — see [Busiest hours](#busiest-hours) |
| `GET` | `/api/v1/cameras/` | List cameras |
| `GET` | `/api/v1/cameras/<id>/` | One camera |
| `POST` | `/api/v1/cameras/<id>/start/` | Start its worker |
| `POST` | `/api/v1/cameras/<id>/stop/` | Stop its worker |
| `GET` | `/api/v1/cameras/<id>/crossings/?limit=50` | Recent crossings |
| `POST` | `/api/v1/cameras/<id>/reset/` | Delete recorded crossings |
| `GET` | `/api/v1/cameras/<id>/stream/` | Annotated MJPEG live stream |
| `GET` | `/api/v1/cameras/<id>/snapshot/` | Latest annotated frame as JPEG |

## Events

An **event** is something being counted: an exhibition, a conference, an open
day. Create one at `/admin/counter/event/add/` with a name, a start and an
end. While it runs, every crossing is stamped with it, so attendance can be
reported per event rather than only per date.

| Field | Meaning |
|---|---|
| `starts_at` / `ends_at` | When counting is attributed to this event |
| `cameras` | **Leave empty to cover every camera.** Set it to scope the event to particular entrances |
| `capacity` | Optional. The dashboard warns as occupancy approaches it, and turns red once over |
| `is_active` | Untick to stop attributing crossings without deleting the event |

While an event runs, the dashboard shows a card with live attendance, who is
still inside, and a capacity bar. Reports accept `?event=<id>`:

```bash
curl 'http://127.0.0.1:8000/api/v1/events/current/'
curl 'http://127.0.0.1:8000/api/v1/hourly/?event=3'
curl -OJ 'http://127.0.0.1:8000/api/v1/daily/?event=3&format=csv'
```

### How it behaves

- **Occupancy counts from the event start.** Anyone inside beforehand is not
  part of it. A multi-day event still gets the daily reset, because the later
  of the two boundaries wins and the venue empties overnight.
- **Overlapping events: the most recently started wins.** Scheduling a
  session inside a longer programme is normal, and the inner one is the more
  specific answer.
- **Deleting an event keeps its crossings.** They simply become
  unattributed, so history is never lost.
- **Crossings outside any event are still recorded**, with no event attached,
  and still appear in the date and hour reports.

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
              "days": 7, "daily_average_in": 27.14,
              "peak_inside": 46, "peak_date": "2026-11-08",
              "peak_at": "2026-11-08T20:13:09+03:00"},
  "days": [{"date": "2026-11-02", "in": 42, "out": 34, "net": 8,
            "peak": 13, "peak_at": "2026-11-02T09:24:00+03:00"}]
}
```

Two things worth knowing:

- **Dates are local**, using the project's `TIME_ZONE` (`Asia/Kuwait` by
  default). Events are stored as UTC instants, so a visitor arriving at 01:30
  local belongs to that day, not to the previous UTC one.
- **Quiet days are included as zeros** rather than omitted, so a chart or
  spreadsheet does not silently skip them.
- **`peak` is the fullest moment of that day**, not its footfall — see
  [Peak occupancy](#peak-occupancy). `busiest_date` and `peak_date` answer
  different questions and are often different days.

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
| `reconnecting` | Lost the source and trying to get it back — see [Staying up](#staying-up) | ❌ |
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

## Alerts

A dashboard only works while somebody is looking at it. Two things are worth
interrupting a person for: the camera that has stopped counting, because the
data is being lost as it happens, and the hall filling past its capacity,
because that is a safety limit.

```bash
COUNTER_ALERTS_ENABLED=1 \
COUNTER_ALERT_WEBHOOK='https://hooks.slack.com/services/…' \
COUNTER_ALERT_EMAILS='ops@example.com' \
.venv/bin/python manage.py runserver 0.0.0.0:8000 --noreload
```

The webhook payload carries a `text` field, which Slack and Teams both
render, with the structured alert beside it for anything else:

```json
{
  "text": "[Expo centre] CRITICAL: Main entrance is offline. …",
  "alert": {"kind": "health", "state": "offline", "level": "critical",
            "camera": "Main entrance", "resolved": false, "started_at": "…"}
}
```

### Conditions, not messages

An alert **opens** when something goes wrong, stays open while it is still
wrong, and **resolves** when it is put right. That is what makes "the hall
was over capacity from 11:40 to 11:52" answerable afterwards, and it is what
stops a check every thirty seconds becoming a message every thirty seconds.
Three rules follow, and they are the whole design:

- **Nothing is sent inside the grace period.** A camera that blinks for two
  seconds is not worth a phone buzzing at midnight. A condition that clears
  before `COUNTER_ALERT_AFTER_SECONDS` (default 60) is recorded and resolved
  having told nobody.
- **Getting worse is not getting fixed.** "Near capacity" becoming "over
  capacity" closes one alert and opens another, and **no "resolved" message
  goes out for the first** — it is marked superseded instead.
- **Recovery is news too.** Whoever was told it broke is told when it is
  working again, with how long it was down.

### What raises one

| Condition | Level |
|---|---|
| Camera `failed` or `offline` | critical |
| Camera `stalled` or `reconnecting` | warning |
| Running event at ≥ 90% of capacity | warning |
| Running event at ≥ 100% of capacity | critical |

Set `COUNTER_ALERT_MIN_SEVERITY=2` to be told only about cameras that are
down, not ones that are merely degraded. `ok`, `disabled` and `finished`
never raise anything.

### Delivery is recorded, not assumed

An alert nobody received is worse than no alert, because a screen saying
"alerted" implies somebody was told.

- A **failed** webhook or email leaves the alert owed and retries on the
  next pass; the error is stored and shown.
- With **no channel configured** the alert is still recorded and shown on
  the dashboard, marked `No alert channel configured.` — the API's
  `delivered` is false, and nothing pretends otherwise.

### Where it runs

Inside the supervisor, at `COUNTER_ALERT_INTERVAL` (default 30s), because
that is the process that owns the workers and camera health can only be
judged where they are. For a look from the outside:

```bash
python manage.py check_alerts --dry-run   # what is wrong, sends nothing
python manage.py check_alerts             # open, resolve and notify
```

From cron it sees no workers, so capacity alerts work and every camera reads
as offline. Keep camera alerting in the counting process.

`/api/v1/alerts/` lists them — `?open=1`, `?kind=health|capacity`, `?limit=`
— open first so a long-running problem cannot be pushed off the end by a
dozen newer ones that are already fixed. The dashboard's **Alerts** panel
shows the same thing, with what is still unsent.

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

`run_counter` runs what you name and exits when it ends. For an unattended
deployment use `supervise`, which also restarts what stops — see
[Staying up](#staying-up).

## Tests

```bash
python manage.py test
```

335 tests covering tracker identity through crossings and occlusion,
event scoping and attribution, reconnection after a dropped stream,
supervisor reconciliation, peak occupancy, alert grace periods,
escalation and delivery,
assignment optimality against brute force, line-crossing direction, the dead
band,
the YOLOv8 output decoding, line-setup suggestions, local-date and local-hour
bucketing, occupancy reset behaviour, camera health states, authentication
and the API. Most use a stub detector, so they need
no camera and no model file; the tests that exercise the real model skip
automatically when it has not been exported.

## Notes

- Workers live in the web server process, so `runserver --noreload` (or a
  single Gunicorn worker) keeps one set of cameras. Across multiple Gunicorn
  workers each process would open its own capture and the supervisor would
  run in each of them; use one worker, or move counting to `supervise`.
- `logs/` is where the launchd agent writes; it is gitignored.
- `db.sqlite3` is gitignored — run `migrate` after cloning.
