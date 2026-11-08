"""Run one alerting pass, for cron or a quick look at what is wrong.

    python manage.py check_alerts            # check, open/resolve, notify
    python manage.py check_alerts --dry-run  # say what is wrong, send nothing

Like ``check_health``, camera health can only be judged in the process that
owns the workers. Run from cron it will see none of them, so capacity alerts
work and camera alerts read as offline; keep alerting in the counting
process (``COUNTER_ALERTS_ENABLED=1`` with the supervisor) unless you are
only watching capacity.

Exit codes: **0** nothing owed or everything delivered, **1** something
could not be delivered.
"""

from django.core.management.base import BaseCommand

from counter.alerts import (
    current_conditions,
    reconcile_alerts,
    send_pending,
)
from counter.models import Alert


class Command(BaseCommand):
    help = "Check for alert conditions and notify about them."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Report what is wrong without recording or sending anything.",
        )
        parser.add_argument(
            "--quiet", action="store_true", help="Only print problems.",
        )

    def handle(self, *args, **options):
        conditions = current_conditions()

        if options["dry_run"]:
            if not conditions:
                self.stdout.write(self.style.SUCCESS("nothing wrong"))
                return
            for condition in conditions:
                self.stdout.write(self.style.WARNING(
                    f"[{condition.level}] {condition.message}"))
            return

        opened, escalated, resolved = reconcile_alerts(conditions)
        sent, failed = send_pending()

        for alert in opened:
            self.stdout.write(self.style.WARNING(f"opened: {alert.message}"))
        for alert in escalated:
            self.stdout.write(self.style.ERROR(f"worse: {alert.message}"))
        for alert in resolved:
            self.stdout.write(self.style.SUCCESS(f"resolved: {alert.message}"))
        for alert in sent:
            if alert.delivery_error:
                self.stdout.write(self.style.WARNING(
                    f"not sent ({alert.delivery_error}): {alert.message}"))
            else:
                self.stdout.write(f"sent: {alert.message}")

        if failed:
            for alert in failed:
                self.stderr.write(self.style.ERROR(
                    f"delivery failed, will retry: {alert.delivery_error}"))
            raise SystemExit(1)

        if not options["quiet"]:
            open_now = Alert.objects.filter(resolved_at__isnull=True).count()
            self.stdout.write(
                self.style.SUCCESS("no open alerts") if not open_now
                else f"{open_now} alert(s) open")
