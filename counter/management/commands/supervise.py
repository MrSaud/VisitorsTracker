"""Keep every active camera running, restarting whatever stops.

The alternative to ``COUNTER_AUTOSTART=1`` inside the web server, for
deployments that would rather keep counting in a process of its own:

    python manage.py supervise

Workers live in the process that started them, so the dashboard's Start and
Stop buttons act on the web server's workers, not on this command's. Pick
one home for counting -- the server with autostart, or this command -- and
leave the other to report.
"""

import signal
import time

from django.core.management.base import BaseCommand

from counter.camera import REGISTRY
from counter.supervisor import DEFAULT_COOLDOWN, DEFAULT_INTERVAL, Supervisor


class Command(BaseCommand):
    help = "Start every active camera and keep them running."

    def add_arguments(self, parser):
        parser.add_argument(
            "--interval", type=float, default=DEFAULT_INTERVAL,
            help=f"Seconds between checks (default {DEFAULT_INTERVAL:.0f}).",
        )
        parser.add_argument(
            "--cooldown", type=float, default=DEFAULT_COOLDOWN,
            help=(
                "Seconds to wait before restarting the same camera again "
                f"(default {DEFAULT_COOLDOWN:.0f})."
            ),
        )
        parser.add_argument(
            "--once", action="store_true",
            help="Do a single pass and exit, for cron or a smoke test.",
        )

    def handle(self, *args, **options):
        supervisor = Supervisor(
            registry=REGISTRY,
            interval=options["interval"],
            cooldown=options["cooldown"],
        )

        if options["once"]:
            self._report(supervisor.reconcile())
            return

        stopping = {"flag": False}

        def _handle_signal(signum, frame):
            stopping["flag"] = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        self.stdout.write(
            f"Supervising cameras every {supervisor.interval:.0f}s. Ctrl-C to stop."
        )
        try:
            while not stopping["flag"]:
                try:
                    self._report(supervisor.reconcile())
                except Exception as exc:  # pragma: no cover - keep supervising
                    self.stderr.write(self.style.WARNING(f"pass failed: {exc}"))
                waited = 0.0
                while waited < supervisor.interval and not stopping["flag"]:
                    time.sleep(0.2)
                    waited += 0.2
        finally:
            self.stdout.write("\nStopping cameras.")
            REGISTRY.stop_all()

    def _report(self, result):
        for pk in result.started:
            self.stdout.write(self.style.SUCCESS(f"started camera {pk}"))
        for pk in result.stopped:
            self.stdout.write(f"stopped camera {pk}")
        for pk in result.waiting:
            self.stdout.write(f"camera {pk} is down, waiting out its cooldown")
        for pk in result.finished:
            self.stdout.write(f"camera {pk} played its clip to the end")
