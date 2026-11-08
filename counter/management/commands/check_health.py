"""Check camera health. Intended for cron, a monitor, or a quick look.

Exit codes make it usable as an alerting check:

    0  everything healthy
    1  degraded (something stalled)
    2  down (a camera failed, or is active but not running)
"""

from django.core.management.base import BaseCommand

from counter.camera import REGISTRY
from counter.health import STATUS_DEGRADED, STATUS_DOWN, STATUS_OK, build_report
from counter.models import Camera
from counter.views import _last_event_times

EXIT_CODES = {STATUS_OK: 0, STATUS_DEGRADED: 1, STATUS_DOWN: 2}


class Command(BaseCommand):
    help = "Report camera health and exit non-zero when something is wrong."

    def add_arguments(self, parser):
        parser.add_argument("--quiet", action="store_true",
                            help="Print only problems.")

    def handle(self, *args, **options):
        cameras = list(Camera.objects.all())
        if not cameras:
            self.stdout.write("No cameras configured.")
            return

        report = build_report(cameras, REGISTRY, _last_event_times(cameras))

        if not options["quiet"]:
            for row in report["cameras"]:
                style = self.style.SUCCESS if row["healthy"] else self.style.ERROR
                line = f"  [{row['state']:8}] {row['camera_name']}"
                if row["seconds_since_frame"] is not None:
                    line += f"  last frame {row['seconds_since_frame']:.0f}s ago"
                if row["minutes_since_event"] is not None:
                    line += f"  last crossing {row['minutes_since_event']:.0f}m ago"
                self.stdout.write(style(line))
                if row["detail"] and not row["healthy"]:
                    self.stdout.write(f"             {row['detail']}")

        status = report["status"]
        message = f"status: {status}"
        if status == STATUS_OK:
            self.stdout.write(self.style.SUCCESS(message))
        else:
            self.stderr.write(self.style.ERROR(message))

        code = EXIT_CODES.get(status, 0)
        if code:
            raise SystemExit(code)
