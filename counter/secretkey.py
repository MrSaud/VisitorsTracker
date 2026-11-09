"""A persistent secret key for installs that have no environment to set one.

A server gets DJANGO_SECRET_KEY from its environment file and refuses to
start without it. A double-clicked desktop app has no environment: there is
no shell, no systemd unit, nobody to export anything. Generating a fresh key
each launch is not an option either -- it signs session cookies, so every
restart would log the operator out.

So the app points DJANGO_SECRET_KEY_FILE at a file it owns, and this creates
it once. The file is deliberately opt-in: a server that forgets to set a key
should fail loudly rather than quietly invent one that nobody has backed up.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

MINIMUM_LENGTH = 50


def read_or_create(path) -> str:
    """The key in ``path``, created on first use. Owner-readable only."""
    path = Path(path)

    existing = _read(path)
    if existing:
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    key = _generate()

    try:
        # O_EXCL so two processes starting together cannot both write; the
        # loser reads what the winner wrote.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Either another process won the race, or what is there is unusable
        # -- truncated, empty, hand-edited. A key under MINIMUM_LENGTH
        # weakens every signature made with it, so replacing it is right;
        # the cost is that sessions signed with it stop validating, which
        # means logging in again.
        existing = _read(path)
        if existing:
            return existing
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC, 0o600)

    try:
        os.write(fd, key.encode("utf-8"))
    finally:
        os.close(fd)

    _restrict(path)
    return key


def _read(path) -> str:
    try:
        key = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    # A truncated or hand-edited file is worse than no file: it would weaken
    # every signature without saying so.
    return key if len(key) >= MINIMUM_LENGTH else ""


def _generate() -> str:
    from django.core.management.utils import get_random_secret_key

    return get_random_secret_key()


def _restrict(path):
    """Owner-only. Best effort: Windows does not model it this way."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
