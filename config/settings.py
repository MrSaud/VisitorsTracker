"""Django settings for the VisitorsTracker project."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-j4v8%kvi*voz9@q*knnt-!o7#gg4%xzstk&1^!_^=2t8g*gug-",
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS = ["*"]

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

CORS_ALLOW_ALL_ORIGINS = True

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
        "NAME": BASE_DIR / "db.sqlite3",
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

# Frames a track may go unseen before it is dropped.
COUNTER_MAX_DISAPPEARED = int(os.environ.get("COUNTER_MAX_DISAPPEARED", "25"))

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

# Width frames are resized to before detection (smaller = faster).
COUNTER_FRAME_WIDTH = int(os.environ.get("COUNTER_FRAME_WIDTH", "640"))
