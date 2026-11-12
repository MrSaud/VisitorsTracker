"""Django settings for the VisitorsTracker project."""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

# The counting defaults live beside the code that measured them, so a
# settings file cannot quietly disagree with them.
from counter.vision import DEFAULT_COMMIT_FRACTION, DEFAULT_JUMP_FRACTION

BASE_DIR = Path(__file__).resolve().parent.parent

# Code and data part company when this is frozen into one executable; see
# config/paths.py. In a checkout they are the same directory.
from config.paths import resolve_data_dir  # noqa: E402

DATA_DIR = resolve_data_dir(BASE_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# DEBUG first: the other two defaults depend on it.
#
# On by default, because the common case is somebody running this on their
# own machine to look at a camera. Every deployment artifact in deploy/ sets
# DJANGO_DEBUG=0, and `manage.py check_setup` says so when it is on.
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

DEV_SECRET_KEY = "django-insecure-j4v8%kvi*voz9@q*knnt-!o7#gg4%xzstk&1^!_^=2t8g*gug-"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")

# An install with no environment to set a key in -- the double-clicked
# desktop app -- points this at a file instead, which is created once and
# then reused so sessions survive a restart. Opt-in on purpose: a server
# that forgot its key should fail loudly, not invent one nobody has saved.
if not SECRET_KEY and os.environ.get("DJANGO_SECRET_KEY_FILE"):
    from counter.secretkey import read_or_create

    SECRET_KEY = read_or_create(os.environ["DJANGO_SECRET_KEY_FILE"])

if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = DEV_SECRET_KEY
    else:
        # Refuse rather than fall back. A shared key in a public repository
        # signs session cookies and password-reset tokens for every install
        # that has one, so a silent default is worse than not starting.
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY must be set when DJANGO_DEBUG=0. Generate one:\n"
            "  python -c \"from django.core.management.utils import "
            "get_random_secret_key as k; print(k())\""
        )

# Comma-separated, e.g. "counter.example.com,10.0.1.14". Wide open while
# DEBUG is on, because then it is somebody's laptop and the host varies.
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")
    if host.strip()
] or (["*"] if DEBUG else [])

if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must list the hostnames this serves when "
        "DJANGO_DEBUG=0, e.g. DJANGO_ALLOWED_HOSTS=counter.example.com"
    )

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "counter",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Wide open in development; in production only what is named. The API is
# read-open, so this is about which other sites may read it from a browser.
CORS_ALLOW_ALL_ORIGINS = DEBUG and not os.environ.get("DJANGO_CORS_ORIGINS")
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# Behind nginx or an ALB, Django sees the proxy's own http:// connection.
# Only trust the header when told to: believing it unconditionally lets a
# client claim HTTPS and defeat the secure-cookie flags below.
if os.environ.get("DJANGO_BEHIND_TLS_PROXY") == "1":
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_HSTS_SECONDS", "0"))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = SECURE_HSTS_SECONDS > 0
    SECURE_SSL_REDIRECT = os.environ.get("DJANGO_SSL_REDIRECT", "0") == "1"

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    # DRF claims ?format= for renderer negotiation, which 404s on
    # ?format=csv. JSON is the only renderer configured, so that mechanism
    # buys nothing; release the parameter for the reports to use.
    "URL_FORMAT_OVERRIDE": None,
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "counter.authentication.StaticTokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    # Reads are open; anything that changes state needs a staff session or
    # the service token. Resetting counts deletes history, so it must not
    # be reachable by anyone who can merely reach the host.
    "DEFAULT_PERMISSION_CLASSES": [
        "counter.permissions.IsOperatorOrReadOnly",
    ],
}

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATA_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "Asia/Kuwait")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = DATA_DIR / "staticfiles"

# Who serves /static/.
#
# Django only serves static files itself while DEBUG is on, and the
# desktop app deliberately turns DEBUG off -- which left the admin with no
# CSS at all, because nothing else was serving them either. There is no
# web server in front of a double-clicked app, so it has to do it.
#
# Set COUNTER_SERVE_STATIC=0 where nginx or a CDN serves them instead;
# nginx would normally intercept /static/ before Django sees it anyway.
COUNTER_SERVE_STATIC = os.environ.get("COUNTER_SERVE_STATIC", "1") == "1"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# VisitorsTracker settings
# ---------------------------------------------------------------------------

# Detector backend: "auto" uses the ONNX model when present and HOG otherwise.
# Force one with "onnx" or "hog".
COUNTER_DETECTOR = os.environ.get("COUNTER_DETECTOR", "auto")

# Optional ONNX person detector (YOLOv8n export). Dropped in counter/ml_models/.
COUNTER_ONNX_MODEL = os.environ.get("COUNTER_ONNX_MODEL", "yolov8n.onnx")

# Minimum detection confidence. Leave unset so each backend uses its own
# default (HOG 0.45 as an SVM score, YOLO 0.35 as a probability) -- the two
# are different scales and one shared number drops real detections.
_min_confidence = os.environ.get("COUNTER_MIN_CONFIDENCE", "").strip()
COUNTER_MIN_CONFIDENCE = float(_min_confidence) if _min_confidence else None

# Detect on every Nth frame; tracking fills the gaps (1 = every frame).
COUNTER_DETECT_EVERY = int(os.environ.get("COUNTER_DETECT_EVERY", "3"))

# How far past the line a track must get before the crossing is counted,
# as a fraction of its own width or height. This is the only guard against
# somebody standing in the doorway shifting their weight, which flips sides
# on every sway. Higher is steadier but will miss somebody who stops just
# over the line; see "Loitering" in the README.
COUNTER_LINE_COMMIT_FRACTION = float(
    os.environ.get("COUNTER_LINE_COMMIT_FRACTION", str(DEFAULT_COMMIT_FRACTION)))

# The largest single-frame step, in multiples of the track's own width, that
# is still the same person. Bigger than this is an identity swap, and
# counting it as a crossing invents visitors.
COUNTER_LINE_JUMP_FRACTION = float(
    os.environ.get("COUNTER_LINE_JUMP_FRACTION", str(DEFAULT_JUMP_FRACTION)))

# Frames a track may go unseen before it is dropped.
COUNTER_MAX_DISAPPEARED = int(os.environ.get("COUNTER_MAX_DISAPPEARED", "25"))

# Tracker: "sort" (Kalman motion + IoU + optimal assignment) or
# "centroid" (nearest-distance, kept as a fallback).
COUNTER_TRACKER = os.environ.get("COUNTER_TRACKER", "sort")

# Consecutive reconnection attempts before a live camera is given up on.
# 0 means never give up: a stream that is down for an hour is still the
# camera you want counting when it comes back. The wait doubles from 1s to
# 30s between attempts, so a camera that will never work costs two log
# lines a minute rather than a busy loop.
COUNTER_RECONNECT_ATTEMPTS = int(os.environ.get("COUNTER_RECONNECT_ATTEMPTS", "0"))

# Start every active camera when the server starts, and restart any that
# stop. Without this, counting does not resume after a reboot or a restart
# until somebody presses Start. Off by default: importing the app should
# not take hold of a camera, and it would fight "manage.py run_counter"
# over the device.
COUNTER_AUTOSTART = os.environ.get("COUNTER_AUTOSTART", "0") == "1"

# How often the supervisor checks that the right cameras are running.
COUNTER_SUPERVISE_INTERVAL = float(
    os.environ.get("COUNTER_SUPERVISE_INTERVAL", "10"))

# A worker with no new frame for this long is reported as stalled. A frozen
# stream otherwise looks identical to a healthy one.
COUNTER_STALE_FRAME_SECONDS = float(
    os.environ.get("COUNTER_STALE_FRAME_SECONDS", "30"))

# Shared token for machine clients: Authorization: Token <value>.
# Unset means no machine client can write; staff sessions still can.
COUNTER_API_TOKEN = os.environ.get("COUNTER_API_TOKEN", "")

# Whether unauthenticated callers may read stats and reports.
COUNTER_PUBLIC_READS = os.environ.get("COUNTER_PUBLIC_READS", "1") == "1"

# Local time of day at which occupancy returns to zero, as "HH:MM".
# Occupancy is counted from this boundary rather than from all history: a
# missed exit is never corrected, so an all-time (in - out) drifts upward
# forever. Set this to when the building is reliably empty.
COUNTER_OCCUPANCY_RESET_AT = os.environ.get("COUNTER_OCCUPANCY_RESET_AT", "00:00")

# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

# A dashboard only works while somebody is watching it. With this on, a
# camera that stops counting and a hall that fills past its capacity reach
# a person instead of waiting to be noticed.
COUNTER_ALERTS_ENABLED = os.environ.get("COUNTER_ALERTS_ENABLED", "0") == "1"

# Slack/Teams-compatible incoming webhook. The payload carries a "text"
# field, which both render, plus the structured alert beside it.
COUNTER_ALERT_WEBHOOK = os.environ.get("COUNTER_ALERT_WEBHOOK", "")

# Comma-separated recipients. Needs the EMAIL_* settings below to be real.
COUNTER_ALERT_EMAILS = os.environ.get("COUNTER_ALERT_EMAILS", "")

# How long a problem must persist before anyone is told. A camera that
# blinks for two seconds is not worth a phone buzzing at midnight.
COUNTER_ALERT_AFTER_SECONDS = float(
    os.environ.get("COUNTER_ALERT_AFTER_SECONDS", "60"))

# 1 alerts on degraded (stalled, reconnecting) as well as down; 2 only on
# down (failed, offline).
COUNTER_ALERT_MIN_SEVERITY = int(
    os.environ.get("COUNTER_ALERT_MIN_SEVERITY", "1"))

# Seconds between alert checks inside the supervisor.
COUNTER_ALERT_INTERVAL = float(os.environ.get("COUNTER_ALERT_INTERVAL", "30"))

# Named in every message, so a person with two venues can tell them apart.
COUNTER_SITE_NAME = os.environ.get("COUNTER_SITE_NAME", "VisitorsTracker")

# Email. The console backend prints instead of sending, which is right for
# development and useless in a venue: set EMAIL_BACKEND to the SMTP one and
# fill in the rest when you want alerts to leave the machine.
EMAIL_BACKEND = os.environ.get(
    "DJANGO_EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend")
EMAIL_HOST = os.environ.get("DJANGO_EMAIL_HOST", "localhost")
EMAIL_PORT = int(os.environ.get("DJANGO_EMAIL_PORT", "25"))
EMAIL_HOST_USER = os.environ.get("DJANGO_EMAIL_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("DJANGO_EMAIL_PASSWORD", "")
EMAIL_USE_TLS = os.environ.get("DJANGO_EMAIL_TLS", "0") == "1"
DEFAULT_FROM_EMAIL = os.environ.get(
    "DJANGO_FROM_EMAIL", "visitorstracker@localhost")

# Width frames are resized to before detection (smaller = faster).
COUNTER_FRAME_WIDTH = int(os.environ.get("COUNTER_FRAME_WIDTH", "640"))
