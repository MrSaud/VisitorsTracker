"""Give a brand new install an administrator it can actually log in as.

A fresh database has no accounts, and a camera can only be added through
the Django admin -- there is no create endpoint. So without this, someone
who installs the app double-clicks it, sees an empty dashboard, and has
no way to configure anything short of a command line. That is not an
install anybody would call easy.

So the first launch makes one, with a generated password written where
the person can find it. Generated rather than fixed: a known default
password on every install is the oldest hole there is, and this app
starts a camera.
"""

from __future__ import annotations

import logging
import os
import secrets
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

USERNAME = "admin"
PASSWORD_FILE = "initial-admin.txt"
ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_password(length=16) -> str:
    """Readable rather than clever: this gets copied off a screen by hand.

    No l/1/I or O/0, because somebody is going to type it wrong once and
    conclude the app is broken.
    """
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def password_path(data_dir) -> Path:
    return Path(data_dir) / PASSWORD_FILE


def ensure_admin(data_dir):
    """Create the first administrator, if there is none.

    Returns ``(username, password_file)`` when one was created, else None.
    Never touches an install that already has accounts -- including one
    where the operator deleted the default on purpose.
    """
    from django.contrib.auth import get_user_model

    User = get_user_model()
    if User.objects.exists():
        return None

    password = generate_password()
    User.objects.create_superuser(USERNAME, "", password)

    path = password_path(data_dir)
    path.write_text(
        "VisitorsTracker administrator\n"
        f"username: {USERNAME}\n"
        f"password: {password}\n\n"
        "Change it at /admin/ and delete this file.\n",
        encoding="utf-8",
    )
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    logger.warning(
        "Created the first administrator (%s). The password is in %s",
        USERNAME, path)
    return USERNAME, path
