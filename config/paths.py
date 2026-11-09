"""Where the code lives, and where its data lives. Not the same place.

In a checkout they are: the database sits beside manage.py and everyone is
happy. Frozen into a single executable they must not be. PyInstaller
unpacks the bundle into a temporary directory and **deletes it on exit**,
so a database written beside the code would be gone at the end of every
run, and a secret key with it -- logged out, no history, every launch.

So code paths keep pointing at the bundle, which is read-only and correct
for templates, migrations and the detector model, while anything written
goes to the platform's own place for application data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "VisitorsTracker"


def is_frozen() -> bool:
    """Whether this is running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def default_data_dir(app_name=APP_NAME) -> Path:
    """The platform's place for an application's own data."""
    if sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(root) / app_name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    root = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(root) / app_name


def resolve_data_dir(base_dir, app_name=APP_NAME) -> Path:
    """Where this install should keep its database, key and backups.

    An explicit COUNTER_DATA_DIR wins, so a server can put the database on
    a volume that is backed up. Otherwise it is beside the code in a
    checkout -- which is what a developer expects -- and the platform's
    application data directory when frozen, which is the only place that
    survives the bundle being thrown away.
    """
    explicit = os.environ.get("COUNTER_DATA_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if is_frozen():
        return default_data_dir(app_name)
    return Path(base_dir)
