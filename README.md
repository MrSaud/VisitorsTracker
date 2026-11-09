# VisitorsTracker

Camera-based visitor counting in Django. A background worker reads each
camera, detects people, tracks them across frames and counts them as they
cross a virtual line — giving you **in**, **out** and **inside now**.

Counting can be scoped to a scheduled **event**: create one in the admin and
crossings are attributed to it for as long as it runs. See [Events](#events).

## Setup

macOS and Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Then, the same on both:

```bash
pip install -r requirements.txt
python manage.py export_model
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open http://127.0.0.1:8000/ for the dashboard.

`export_model` downloads and converts the detector weights, which are not in
the repository. Skip it and the counter still runs, but on the built-in HOG
detector instead of YOLO — noticeably worse, with nothing on screen to say
why. `manage.py check_setup` reports which one you ended up with.

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

### The app bundle

Borrowing Terminal's permission works, but it is a workaround: the camera is
granted to a terminal emulator, the window has to stay open, and there is
nothing to double-click. `build_app` wraps the same server in a real macOS
application so it has an identity of its own:

```bash
pip install -r requirements-desktop.txt
python manage.py build_app
open dist/VisitorsTracker.app
```

What that gives:

- **Its own entry in Privacy & Security > Camera**, named VisitorsTracker. The
  bundle carries `NSCameraUsageDescription`, which is what macOS shows in the
  prompt — and without which it terminates the process instead of asking.
- **A window**, via `pywebview`, rendering the same dashboard. No browser tab,
  no Terminal.
- **The same server**, on `127.0.0.1:8000`. The window is a view onto a real
  HTTP server, not a replacement for it, so a second browser on the same
  machine and the REST API both still work. It is *not* reachable from
  another device unless you bind `0.0.0.0` — see
  [Viewing it from another machine](#viewing-it-from-another-machine).

It carries `assets/AppIcon.icns` unless you pass `--icon` or `--no-icon`. To
replace the artwork:

```bash
python scripts/make_icon.py my-artwork.png
python manage.py build_app
```

That script does one thing worth knowing about. macOS 26 and later draw their
own rounded container and place the app's icon inside it, so a pre-rounded
icon with a margin — which is what icon artwork normally looks like — renders
as a rounded square *inside* a rounded square, shrunk and ringed in grey. The
icon therefore has to bleed to the edges and let the system do the rounding.
Since the source artwork's own corners are transparent, the script paints the
background into them by inpainting from the surrounding pixels, rather than
cropping and losing a slice of the design at every edge.

It is ad-hoc signed (`codesign --sign -`) so TCC has a stable identity to
remember the answer against; pass `--no-sign` to skip that. The path to the
project and to its interpreter are baked in at build time, so the bundle works
from the Dock or `/Applications` — move the project and rebuild.

Double-clicking it twice is handled by macOS, which activates the running copy
rather than starting a second one. The app's own guard is for the other case:
when a server is *already* running from `manage.py runserver`, the Terminal
launcher or the LaunchAgent, the app opens a window onto that one instead of
failing on a busy port. Two servers would fight over the camera, because a
capture device opens once.

Closing the window releases the camera and stops the server. The bundle's WSGI
server deliberately does not block on close — an MJPEG live view is a request
that never finishes, and the default would leave the app running invisibly with
the camera still held.

**The permission really does land on the bundle**, which was worth checking
rather than assuming. Homebrew's `bin/python3.12` is a stub that executes
`Python.framework/.../Python.app/Contents/MacOS/Python`, so the running
process's binary lives *outside* the bundle — the grant could plausibly have
gone to that shared interpreter instead, and been shared with every Python GUI
app on the machine. It does not: macOS attributes it to the application that
was launched. Confirmed on macOS 27.2 with Homebrew python 3.12.12 —
**Privacy & Security > Camera** lists *VisitorsTracker*, not *Python*.

If a different interpreter layout ever shows *Python* in that pane, putting a
framework interpreter inside `Contents/MacOS/` would pin the grant to the
bundle. There is no way to read the attribution programmatically: the TCC
database needs Full Disk Access, so that pane is the only place it is visible.

### Viewing it from another machine

The dashboard is a web page, so **any** device with a browser can display it —
a Windows PC, a phone, a tablet — with nothing installed on that device. That
has always been true and needs no port of anything.

What is *not* portable that way is the counting. `cv2.VideoCapture(0)` opens a
capture device attached to the machine it runs on, so the counting process has
to run on whichever computer the camera is plugged into. That machine's
operating system is the only one that matters.

So:

| Camera is plugged into | What you need |
|---|---|
| A Mac, viewed from Windows | Nothing Windows-specific. Bind `0.0.0.0` and open the Mac's address in a browser |
| A Windows PC | The Windows launcher below — the counting runs there |

By default the server listens on `127.0.0.1:8000`, which is the machine
itself and nothing else. To let other devices reach it:

```bash
python manage.py runserver 0.0.0.0:8000 --noreload
```

Think before you do. **Reads are unauthenticated** — anyone who can reach the
host can see the counts and the camera's live view. Writes need a staff
session or the service token, so nobody can stop a camera or wipe history,
but the video is open. Set `ALLOWED_HOSTS` to the real hostname and turn off
`CORS_ALLOW_ALL_ORIGINS` at the same time.

### Windows

The counting engine is the same on both platforms — there are no
`sys.platform` branches in it and no POSIX paths — so Windows needs only a
launcher and an icon of its own:

```bat
pip install -r requirements.txt -r requirements-desktop.txt
python manage.py export_model
python manage.py migrate
python manage.py build_app
powershell -ExecutionPolicy Bypass -File dist\CreateShortcut.ps1
```

`export_model` is not optional. The detector weights are gitignored, so a
fresh clone has none and `COUNTER_DETECTOR=auto` quietly falls back to HOG —
the cameras run and the counts get much worse, with nothing on screen to say
why. Every dependency has a Windows wheel, including the pinned OpenCV
(`cp37-abi3-win_amd64`).

That writes `dist\VisitorsTracker.bat` and a `.lnk` shortcut carrying
`assets\AppIcon.ico`. Drag the shortcut to the desktop or taskbar.

Three differences from the Mac worth knowing:

- **The launcher runs `pythonw.exe`, not `python.exe`.** With `python.exe` a
  console window sits behind the app for as long as it runs, and closing it
  kills the server.
- **The icon is a different shape.** macOS 26+ masks app icons, so the
  `.icns` is full bleed; Windows draws the `.ico` exactly as given, so that
  one keeps its own rounded corners. `scripts/make_icon.py` writes both from
  one source image.
- **Camera permission is a system-wide toggle**, not a per-app grant. If the
  camera will not open, check **Settings > Privacy & security > Camera** and
  that *"Let desktop apps access your camera"* is on. None of the macOS
  bundle-identity machinery applies.

OpenCV's default capture backend on Windows is Media Foundation, which can
take seconds to open a webcam and fails outright on some. The worker tries
**DirectShow** first and keeps the default as a fallback. Network cameras are
untouched by this — a URL always goes through FFmpeg.

For unattended running, `deploy/visitorstracker-task.xml` is the Task
Scheduler counterpart to the macOS LaunchAgent:

```bat
schtasks /create /tn VisitorsTracker /xml deploy\visitorstracker-task.xml
```

Edit the paths in it first. It points at `VisitorsTracker-headless.bat`, which
is what sets `COUNTER_AUTOSTART` — a scheduled task's `<Exec>` has nowhere to
put environment variables, and without it the server starts but no camera
does.

Whichever machine the camera ends up on, `python manage.py check_setup` on
that machine says whether it is ready — dependencies, which detector is
actually loaded, migrations, the port, `DEBUG`, and the cameras. With
`--cameras` it tries to open each one, which is the step that proves the
permission is granted. Run it there before trusting a single count.

### A standalone .exe

For a machine with no Python on it, `VisitorsTracker.spec` builds one
executable with the interpreter, OpenCV and the detector inside it.

**This has to be built on Windows.** PyInstaller bundles the interpreter and
binaries of the machine it runs on; it does not cross-compile, so a Mac
produces a Mac app and nothing else.

```bat
pip install -r requirements.txt -r requirements-desktop.txt pyinstaller
python manage.py export_model
pyinstaller VisitorsTracker.spec
```

The result is `dist\VisitorsTracker.exe`, several hundred MB, which runs by
double-click with nothing installed.

Three things the spec has to be told, because Django finds them by scanning
the filesystem rather than importing them, so PyInstaller never sees them:
**management commands**, **migrations** and **templates**. The detector model
goes in too — a frozen app has nowhere to write one, and `export_model`
builds a throwaway environment with torch in it, which is not something to
do on a customer's machine.

The data moves as well. Frozen, the code is unpacked into a temporary
directory that is **deleted when the app exits**, so the database and secret
key live in the platform's application-data directory instead
(`%LOCALAPPDATA%\VisitorsTracker` on Windows). `COUNTER_DATA_DIR` overrides
it. In a checkout nothing changes: data stays beside the code.

With no console window, a failure before the window opens would be
invisible, so a frozen build writes `visitorstracker.log` into that same
directory.

Expect Windows SmartScreen to warn about an unsigned executable the first
time. Signing it needs a code-signing certificate, which costs money and is
issued to a registered business.

**None of the Windows path has been run on Windows.** This project is
developed on a Mac. The generated files are pinned by tests and the backend
choice is asserted, but "asserted" is not "works" — expect to iron something
out on first run. `python manage.py build_app --target windows` on a Mac
generates the files for inspection, with this machine's paths in them.

## Deploying on a Linux server

The server can only count cameras it can reach, so this shape assumes
**network cameras** (RTSP/HTTP). A USB webcam plugged into a laptop is not
visible to a server in another building; that needs the counting to run on
the laptop, with the server doing dashboards only — which this does not
support today.

```bash
sudo useradd --system --home /opt/visitorstracker visitorstracker
sudo git clone <repo> /opt/visitorstracker && cd /opt/visitorstracker
sudo -u visitorstracker python3.12 -m venv .venv
sudo -u visitorstracker .venv/bin/pip install -r requirements.txt -r requirements-server.txt
sudo -u visitorstracker .venv/bin/python manage.py export_model

sudo cp deploy/visitorstracker.env.example /etc/visitorstracker.env
sudo chmod 600 /etc/visitorstracker.env     # it holds the secret key
sudoedit /etc/visitorstracker.env

sudo cp deploy/visitorstracker.service deploy/visitorstracker-backup.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now visitorstracker visitorstracker-backup.timer

sudo cp deploy/nginx-visitorstracker.conf /etc/nginx/conf.d/
sudo nginx -t && sudo systemctl reload nginx
```

Then, on the server:

```bash
python manage.py check_setup --benchmark
```

### Will the machine keep up?

Detection is the entire cost; tracking is noise beside it. It runs on every
*N*th frame (`COUNTER_DETECT_EVERY`) and tracking fills the gaps, so the
video rate a machine can sustain is roughly
`(detections per second) x COUNTER_DETECT_EVERY`. `--benchmark` measures it:

```
Speed
  ok   detection              43 ms (23.1/s)
  ok   detect_every           3 (tracking fills the gaps)
  ok   sustainable rate       69 fps of video
```

That is an M1 Mac, with plenty of headroom over a 20 fps camera. A small
burstable cloud instance will be far slower, and a machine that cannot keep
up does not fail — it falls behind, and people cross the line between
detections, which shows up as undercounting rather than as an error. Below
15 fps sustainable the check warns; below 8 it fails and names the levers:
a bigger machine, a higher `COUNTER_DETECT_EVERY`, or a lower
`COUNTER_FRAME_WIDTH`.

Run it on the server **before** mounting a camera.

### Running the desktop app in production

The Mac app and the Windows launcher configure themselves, because a
double-clicked app has no shell to export anything in:

```
DJANGO_DEBUG=0
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_SECRET_KEY_FILE=<project>/.secret_key
```

Each is a default, so a server or a developer that sets the variable still
wins. The key file is created on first launch with mode 600 and reused after
that: a fresh key every restart would log the operator out each time, and it
is gitignored, so it never leaves the machine. It is replaced rather than
trusted if it is ever truncated — a key under 50 characters weakens every
signature made with it.

`DEBUG=0` matters even on a localhost-only app: with it on, an error serves
a traceback containing your settings to anyone who can reach the port, and
Django keeps the last 9000 queries in memory for a process meant to run for
weeks.

### Burstable instances are the wrong family

A t2/t3/t4g instance earns CPU credits at a fixed rate and spends them when
it works. Counting is not bursty — it is one steady CPU-bound load, forever.
Do the arithmetic before choosing one:

```
t3.medium earns                       24 credits/hour  (2 vCPU x 20% baseline)
one camera at 20fps, detect_every 3    6.7 detections/second
  at 100ms per detection             = 0.67 vCPU       = 40 credits/hour
```

That is a deficit whatever the per-detection figure turns out to be, and the
credit balance is finite: when it runs out the instance is throttled to its
baseline, counting collapses, and the only symptom is a frame rate quietly
falling until people cross the line between detections. In `unlimited` mode
it does not throttle — it bills surplus credits instead, forever.

Use a compute-optimised instance (c-family) sized from
`check_setup --benchmark` on the box itself, and remember that whatever else
the machine already does is competing for the same cores.

### One worker. Not for load — for correctness

`deploy/gunicorn.conf.py` sets `workers = 1`, and raising it breaks counting.

Camera workers are threads inside the web process, so each gunicorn worker
process would start its own counter for the same camera. An RTSP stream
serves them all happily, so nothing errors — the counts simply multiply.
Measured, before cameras took a lease: **four processes over a clip
containing 3 crossings recorded 12.**

Cameras now take a **lease** (`lease_owner`, `lease_expires_at`), so a second
process is refused rather than counting alongside the first. That stops the
corruption, but extra workers still would not work: only the process holding
the lease has a live camera, so the live view, the start/stop buttons and the
health tiles would answer differently depending on which worker got the
request. One process with many threads is the right shape here. The lease
expires after 45 seconds, so a process that is killed does not strand its
camera.

### Settings refuse to start wrong

With `DJANGO_DEBUG=0`, the app will not boot without `DJANGO_SECRET_KEY` or
`DJANGO_ALLOWED_HOSTS`. A shared key in a public repository signs session
cookies for every install that has one, so a silent fallback is worse than
not starting. `DJANGO_BEHIND_TLS_PROXY=1` turns on secure cookies and lets
Django see the real scheme; do not set it unless something really is
terminating TLS, or a client could claim HTTPS itself. With that config
`manage.py check --deploy` reports only the HSTS-preload warning, which is a
deliberate opt-in.

### Backups

```bash
python manage.py backup_db --keep 14
```

`visitorstracker-backup.timer` runs it nightly at 00:30, after the daily
occupancy figure rolls over. It uses SQLite's own backup API rather than
copying the file, because `cp` on a database that the counter is writing to
can produce a copy that is subtly corrupt. Verified against the live
database with a camera running: `PRAGMA integrity_check` returns `ok`.

## How counting works

1. **Detect** — a YOLOv8 ONNX model when one is present, otherwise OpenCV's
   built-in HOG pedestrian detector. See [Detector](#detector) below.
2. **Track** — a SORT-style tracker assigns a stable id to each person, so
   one visitor is counted once rather than once per frame. See
   [Tracking](#tracking).
3. **Count** — each track's centroid is compared against the counting line.
   A side change that survives the checks below emits one `Crossing`, with
   `direction` of `in` or `out`.

Tracks are positional and short-lived, so the same person returning later is
a new track and counts again: **`in` is entry crossings, not unique people**.

Set `invert_direction` on a camera if in and out come out backwards for its
mounting angle.

### Loitering

Somebody standing in the doorway shifting their weight crosses the line over
and over. Counting each flip is wrong, and it is wrong in the direction that
flatters you — footfall inflates while occupancy stays roughly right, so the
headline number is the one that lies. Measured on a 240px frame:

| One person swaying on the line | Counted |
|---|---|
| ±20px | 1 ✅ |
| ±30px | **4 in, 3 out** ❌ |
| ±40px | **6 in, 5 out** ❌ |

The dead band alone does not fix this: it only decides when a side is
*claimed*, and a sway of 30px clears it easily. What separates the two cases
is **how far past the line they actually get**:

| | Reaches past the line |
|---|---|
| Swaying, even ±50px | **32px** |
| Walking through, 10–60px/frame | **70–110px** |

That gap is the real signal, and unlike a frame count it does not care how
fast somebody moves. So a side change is believed once the track is more than
`COUNTER_LINE_COMMIT_FRACTION` of its own box past the line (default 0.6) — a
fraction of the person, not a pixel count, so it holds whether they are close
to the lens or down the hall.

This is also why a suggested line is kept between 0.2 and 0.8 of the frame:
at 0.9 a close subject has no room left to get clear of the line.

### Why not a frame count

There used to be a second guard — the track had to hold the new side for
`COUNTER_LINE_CONFIRM_FRAMES` frames — and *either* guard committed the
crossing. It was removed after measuring it, because a frame count cannot
tell a slow sway from a walk: a sway lingers at the far end of its arc for as
long as a walk takes to leave. Forty-five seconds of one person seated at a
webcam, line through the middle of the frame:

| Rule | False crossings (truth: 0) |
|---|---|
| Distance **or** 2 frames held | **29** |
| Distance **or** 10 frames held | **29** |
| Distance alone | **0** |

Raising the threshold does not help. On real footage the distance guard alone
counted exactly the same crossings as the pair of them, 0.1s later each.

The removal has a price, which is deliberate and tested: somebody who steps
over the line and stops short of their own body width — holding a door,
turning to talk — is no longer counted. The frame count used to catch that,
at the cost of counting every sway as a visitor.

Detection runs on every *N*th frame (`COUNTER_DETECT_EVERY`, default 3) and
tracking fills the gaps, which is what makes CPU-only operation workable.

## Detector

`COUNTER_DETECTOR=auto` (the default) uses a YOLOv8 ONNX model when one exists
in `counter/ml_models/`, and falls back to HOG otherwise. Export one with:

```bash
python manage.py export_model            # yolov8n
python manage.py export_model yolov8s    # more accurate, slower
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
| `COUNTER_LINE_COMMIT_FRACTION` | `0.6` | How far past the line a track must get before it counts, as a fraction of its own box — see [Loitering](#loitering) |
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

## Measuring accuracy

Until you can say "within 3%", you cannot put a number in front of a client.
`measure` runs the real counter over a recording — same detector, same
tracker, same line logic as a live camera — and compares it with a hand count:

```bash
python manage.py measure door.mp4 --expect-in 143 --expect-out 139
```

```
door.mp4
  12043 frames, 10:02.15 at 20 fps · detector onnx · processed in 54.2s
  line horizontal at 0.5

  counted: 148 in, 141 out

   in: counted 148, truth 143 → +5 (3.5%)
  out: counted 141, truth 139 → +2 (1.4%)
```

Nothing is written to the database — this is a measurement, not a shift.

When the numbers disagree, the next question is *where*, and guessing is no
better than arguing. Every crossing is logged with the time in the clip it
happened, and `--record` writes back the annotated video:

```bash
python manage.py measure door.mp4 --record checked.mp4 --csv found.csv
```

```csv
frame,at,seconds,direction,track_id
118,00:05.90,5.9,in,6
125,00:06.25,6.25,in,1
```

Open `checked.mp4` at 00:06.25 and you can see what the counter saw.

`--line`, `--orientation` and `--invert` let you try a different setup on the
same clip without touching a camera in the admin, which turns "where should
the line go" into something you measure rather than guess.

## Tests

```bash
python manage.py test
```

352 tests covering tracker identity through crossings and occlusion,
event scoping and attribution, reconnection after a dropped stream,
supervisor reconciliation, peak occupancy, alert grace periods,
escalation and delivery, loitering on the line,
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
