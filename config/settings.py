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
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
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

# Detector backend: "hog" needs no model file. "onnx" uses COUNTER_ONNX_MODEL.
COUNTER_DETECTOR = os.environ.get("COUNTER_DETECTOR", "hog")

# Optional ONNX person detector (YOLOv8n export). Dropped in counter/ml_models/.
COUNTER_ONNX_MODEL = os.environ.get("COUNTER_ONNX_MODEL", "yolov8n.onnx")

# Minimum detection confidence.
COUNTER_MIN_CONFIDENCE = float(os.environ.get("COUNTER_MIN_CONFIDENCE", "0.45"))

# Detect on every Nth frame; tracking fills the gaps (1 = every frame).
COUNTER_DETECT_EVERY = int(os.environ.get("COUNTER_DETECT_EVERY", "3"))

# Frames a track may go unseen before it is dropped.
COUNTER_MAX_DISAPPEARED = int(os.environ.get("COUNTER_MAX_DISAPPEARED", "25"))

# Width frames are resized to before detection (smaller = faster).
COUNTER_FRAME_WIDTH = int(os.environ.get("COUNTER_FRAME_WIDTH", "640"))
