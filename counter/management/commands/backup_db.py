"""Take a consistent copy of the database.

Copying the SQLite file with cp while something is writing to it can
produce a file that is subtly corrupt -- the copy may catch a page mid
write, or miss the journal that would complete the transaction. SQLite's
own backup API walks the database under a read lock and is safe against a
live writer, which this server always has: the camera workers.
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone


class Command(BaseCommand):
    help = "Write a consistent copy of the database, and prune old ones."

    def add_arguments(self, parser):
        parser.add_argument(
            "--into", default=None,
            help="Directory for the backups (default: backups/ beside the "
                 "database).")
        parser.add_argument(
            "--keep", type=int, default=14,
            help="How many to keep. 0 keeps everything. Default 14.")
        parser.add_argument(
            "--no-compress", action="store_true",
            help="Leave the copy uncompressed.")

    def handle(self, *args, **options):
        config = settings.DATABASES["default"]
        if "sqlite" not in config["ENGINE"]:
            raise CommandError(
                f"This backs up SQLite. For {config['ENGINE']} use that "
                "database's own tool (pg_dump, mysqldump).")

        name = str(config["NAME"])
        if ":memory:" in name or "mode=memory" in name:
            raise CommandError(
                "This database lives in memory, so there is nothing to back "
                "up. That is the case under `manage.py test`.")

        source = Path(name)
        if not source.exists():
            raise CommandError(f"No database at {source}.")

        into = Path(options["into"] or source.parent / "backups")
        into.mkdir(parents=True, exist_ok=True)

        stamp = timezone.localtime().strftime("%Y%m%d-%H%M%S")
        plain = into / f"{source.stem}-{stamp}.sqlite3"

        self.copy_safely(source, plain)

        final = plain
        if not options["no_compress"]:
            final = plain.with_suffix(plain.suffix + ".gz")
            with open(plain, "rb") as raw, gzip.open(final, "wb") as packed:
                shutil.copyfileobj(raw, packed)
            plain.unlink()

        size = final.stat().st_size / (1024 * 1024)
        self.stdout.write(self.style.SUCCESS(f"Wrote {final} ({size:.1f}MB)"))

        removed = self.prune(into, source.stem, options["keep"])
        if removed:
            self.stdout.write(f"Removed {removed} older backup(s).")

    def copy_safely(self, source, target):
        """SQLite's backup API: safe while the counter is writing."""
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as live, \
                sqlite3.connect(target) as copy:
            live.backup(copy)

    def prune(self, into, stem, keep) -> int:
        if keep <= 0:
            return 0
        backups = sorted(
            (p for p in into.iterdir()
             if p.name.startswith(f"{stem}-") and p.suffix in (".sqlite3", ".gz")),
            key=lambda p: p.name,
        )
        doomed = backups[:-keep] if len(backups) > keep else []
        for path in doomed:
            path.unlink()
        return len(doomed)
