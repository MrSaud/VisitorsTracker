"""The daily boundary that occupancy is measured from.

Occupancy cannot be all-time ``in - out``: every counter misses some
crossings, and a missed exit is never corrected, so the error accumulates
forever. At 1% missed exits and 300 visitors a day, "inside now" reads 90
after a month with the building empty.

So occupancy is counted from the most recent daily reset instead. The
building really is empty across that boundary, which makes the reset a
truth, not an approximation.
"""

from datetime import datetime, time, timedelta

from django.conf import settings
from django.utils import timezone

DEFAULT_RESET_AT = "00:00"


def parse_reset_time(value=None) -> time:
    """Parse ``HH:MM``. Falls back to midnight rather than breaking the app."""
    raw = (value if value is not None else getattr(
        settings, "COUNTER_OCCUPANCY_RESET_AT", DEFAULT_RESET_AT)) or DEFAULT_RESET_AT
    raw = str(raw).strip()
    try:
        hour, _, minute = raw.partition(":")
        parsed = time(int(hour), int(minute or 0))
    except (TypeError, ValueError):
        return time(0, 0)
    return parsed


def occupancy_window_start(now=None) -> datetime:
    """The most recent reset instant at or before ``now`` (local, aware)."""
    now = now or timezone.localtime()
    if timezone.is_naive(now):
        now = timezone.make_aware(now, timezone.get_current_timezone())
    now = timezone.localtime(now)

    reset = parse_reset_time()
    tz = timezone.get_current_timezone()
    candidate = timezone.make_aware(datetime.combine(now.date(), reset), tz)
    if candidate > now:
        candidate -= timedelta(days=1)
    return candidate
