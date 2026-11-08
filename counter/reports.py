"""Per-date aggregation of crossings.

Events are stored as UTC instants, but a report is read in local time: a
visitor arriving at 01:30 in Kuwait belongs to that day, not to the previous
UTC one. Everything here buckets by the active timezone's date.
"""

from datetime import date as date_cls, datetime, time, timedelta

from django.db.models import Count
from django.db.models.functions import ExtractHour, TruncDate
from django.utils import timezone

from .daywindow import occupancy_window_start
from .models import Crossing
from .vision import DIRECTION_IN, DIRECTION_OUT

# Guard rails for the API: one day minimum, about two years maximum.
MIN_DAYS = 1
MAX_DAYS = 730
DEFAULT_DAYS = 30


def local_today() -> date_cls:
    return timezone.localtime().date()


def resolve_range(start=None, end=None, days=DEFAULT_DAYS):
    """Work out the inclusive date range to report on.

    Any of ``start`` / ``end`` / ``days`` may be given. Whatever is missing is
    derived, so ``days=7`` alone means "the last 7 days ending today".
    """
    days = max(MIN_DAYS, min(MAX_DAYS, int(days or DEFAULT_DAYS)))

    if start and end:
        pass
    elif start and not end:
        end = start + timedelta(days=days - 1)
    elif end and not start:
        start = end - timedelta(days=days - 1)
    else:
        end = local_today()
        start = end - timedelta(days=days - 1)

    if start > end:
        start, end = end, start

    # Never report a window larger than the cap, even via explicit dates.
    if (end - start).days + 1 > MAX_DAYS:
        start = end - timedelta(days=MAX_DAYS - 1)

    return start, end


def _bounds(start: date_cls, end: date_cls):
    """Half-open aware datetime bounds covering the local dates start..end."""
    tz = timezone.get_current_timezone()
    begin = timezone.make_aware(datetime.combine(start, time.min), tz)
    finish = timezone.make_aware(datetime.combine(end + timedelta(days=1), time.min), tz)
    return begin, finish


def daily_counts(start=None, end=None, days=DEFAULT_DAYS, camera=None, event=None):
    """One row per local date between start and end, zero-filled.

    Gaps are filled so a quiet day reads as 0 rather than vanishing -- a
    chart or CSV with missing days is easy to misread.
    """
    start, end = resolve_range(start, end, days)
    begin, finish = _bounds(start, end)

    crossings = Crossing.objects.filter(occurred_at__gte=begin, occurred_at__lt=finish)
    if camera is not None:
        crossings = crossings.filter(camera=camera)
    if event is not None:
        crossings = crossings.filter(event=event)

    rows = (
        crossings.annotate(day=TruncDate("occurred_at"))
        .values("day", "direction")
        .order_by()
        .annotate(total=Count("id"))
    )

    buckets: dict[date_cls, dict[str, int]] = {}
    for row in rows:
        day = row["day"]
        if isinstance(day, datetime):      # some backends hand back a datetime
            day = day.date()
        bucket = buckets.setdefault(day, {DIRECTION_IN: 0, DIRECTION_OUT: 0})
        bucket[row["direction"]] = row["total"]

    peaks = occupancy_peaks(start, end, camera=camera, event=event)

    report = []
    cursor = start
    while cursor <= end:
        bucket = buckets.get(cursor, {})
        total_in = bucket.get(DIRECTION_IN, 0)
        total_out = bucket.get(DIRECTION_OUT, 0)
        peak = peaks.get(cursor) or {}
        report.append({
            "date": cursor.isoformat(),
            "in": total_in,
            "out": total_out,
            "net": total_in - total_out,
            # The busiest moment of the day, which "in" cannot show: 400
            # visitors through the door is a different problem if they came
            # in ones and twos than if 300 were inside at once.
            "peak": peak.get("peak", 0),
            "peak_at": peak["at"].isoformat() if peak.get("at") else None,
        })
        cursor += timedelta(days=1)

    return report


def _crossing_stream(begin, finish, camera=None, event=None):
    """Every crossing in the window, oldest first, as (when, direction)."""
    crossings = Crossing.objects.filter(occurred_at__gte=begin, occurred_at__lt=finish)
    if camera is not None:
        crossings = crossings.filter(camera=camera)
    if event is not None:
        crossings = crossings.filter(event=event)
    return (
        crossings.order_by("occurred_at", "id")
        .values_list("occurred_at", "direction")
        .iterator()
    )


def occupancy_peaks(start=None, end=None, days=DEFAULT_DAYS, camera=None, event=None):
    """Highest simultaneous occupancy per local date.

    The live occupancy figure only answers "how many are inside now", which
    is no use the morning after: a hall can breach its capacity at 11:40 and
    read comfortably by the time anyone looks. This walks the crossings in
    order -- plus one for an entry, minus one for an exit -- and keeps the
    high-water mark and when it happened.

    Two deliberate choices:

    * The running count is clamped at zero and reset at the daily occupancy
      boundary, exactly as :meth:`counter.models.Camera.occupancy` is, so a
      missed exit cannot inflate every later day.
    * Peaks are bucketed by the date the crossing happened on, so the column
      lines up with the in/out columns beside it even when the reset is not
      at midnight.

    Returns ``{date: {"peak": int, "at": datetime | None}}``.
    """
    start, end = resolve_range(start, end, days)
    begin, finish = _bounds(start, end)

    # When the reset is not at midnight, the window covering the first day
    # began before it. Walk from there so the day does not start at zero
    # with people already inside.
    walk_from = min(begin, occupancy_window_start(begin))

    peaks: dict[date_cls, dict] = {}
    running = 0
    window = None

    for occurred_at, direction in _crossing_stream(walk_from, finish, camera, event):
        boundary = occupancy_window_start(occurred_at)
        if window is None or boundary != window:
            window = boundary
            running = 0

        running = max(0, running + (1 if direction == DIRECTION_IN else -1))

        if occurred_at < begin:
            continue                      # only here to build the running count

        day = timezone.localtime(occurred_at).date()
        best = peaks.get(day)
        if best is None or running > best["peak"]:
            peaks[day] = {"peak": running, "at": occurred_at}

    return peaks


def peak_occupancy(start=None, end=None, days=DEFAULT_DAYS, camera=None, event=None):
    """The busiest single moment in the window.

    ``{"peak": 142, "at": "...T11:40...", "date": "2026-11-08"}``, or a zero
    peak with no time when nothing was recorded.
    """
    peaks = occupancy_peaks(start, end, days, camera, event)
    if not peaks:
        return {"peak": 0, "at": None, "date": None}

    day, best = max(peaks.items(), key=lambda item: (item[1]["peak"], item[0]))
    if not best["peak"]:
        return {"peak": 0, "at": None, "date": None}
    return {
        "peak": best["peak"],
        "at": best["at"].isoformat(),
        "date": day.isoformat(),
    }


def summarise(rows):
    """Totals and the busiest day across a daily report."""
    if not rows:
        return {"in": 0, "out": 0, "net": 0, "busiest_date": None,
                "busiest_in": 0, "days": 0, "daily_average_in": 0.0,
                "peak_inside": 0, "peak_date": None, "peak_at": None}

    total_in = sum(r["in"] for r in rows)
    total_out = sum(r["out"] for r in rows)
    busiest = max(rows, key=lambda r: r["in"])
    fullest = max(rows, key=lambda r: r.get("peak", 0))
    peak_inside = fullest.get("peak", 0)
    return {
        "in": total_in,
        "out": total_out,
        "net": total_in - total_out,
        "busiest_date": busiest["date"] if busiest["in"] else None,
        "busiest_in": busiest["in"],
        "days": len(rows),
        "daily_average_in": round(total_in / len(rows), 2),
        # Footfall and fullness are different questions; report both.
        "peak_inside": peak_inside,
        "peak_date": fullest["date"] if peak_inside else None,
        "peak_at": fullest.get("peak_at") if peak_inside else None,
    }


def parse_date(value):
    """Parse YYYY-MM-DD, returning None for blanks and raising on nonsense."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"'{raw}' is not a date in YYYY-MM-DD form") from exc


# ---------------------------------------------------------------------------
# hour-of-day reporting
# ---------------------------------------------------------------------------

HOURS = range(24)

# Monday-first, matching date.weekday().
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def _hour_rows(start, end, camera=None, event=None):
    """Raw (local date, local hour, direction) -> count."""
    begin, finish = _bounds(start, end)

    crossings = Crossing.objects.filter(occurred_at__gte=begin, occurred_at__lt=finish)
    if camera is not None:
        crossings = crossings.filter(camera=camera)
    if event is not None:
        crossings = crossings.filter(event=event)

    rows = (
        crossings.annotate(
            day=TruncDate("occurred_at"),
            hour=ExtractHour("occurred_at"),
        )
        .values("day", "hour", "direction")
        .order_by()
        .annotate(total=Count("id"))
    )

    buckets = {}
    for row in rows:
        day = row["day"]
        if isinstance(day, datetime):
            day = day.date()
        key = (day, int(row["hour"]))
        bucket = buckets.setdefault(key, {DIRECTION_IN: 0, DIRECTION_OUT: 0})
        bucket[row["direction"]] = row["total"]
    return buckets


def hourly_profile(start=None, end=None, days=DEFAULT_DAYS, camera=None, event=None):
    """Crossings by hour of day, averaged over the window.

    ``average_in`` divides by every day in the range, including quiet ones.
    That is deliberate and predictable, but it flattens weekday peaks when
    the range spans a weekend -- use :func:`weekday_hour_matrix` when the
    question is "how many staff on a Tuesday morning".
    """
    start, end = resolve_range(start, end, days)
    buckets = _hour_rows(start, end, camera, event)
    span = (end - start).days + 1

    totals = {hour: {DIRECTION_IN: 0, DIRECTION_OUT: 0} for hour in HOURS}
    for (_day, hour), counts in buckets.items():
        totals[hour][DIRECTION_IN] += counts.get(DIRECTION_IN, 0)
        totals[hour][DIRECTION_OUT] += counts.get(DIRECTION_OUT, 0)

    profile = []
    grand_total = sum(t[DIRECTION_IN] for t in totals.values())
    for hour in HOURS:
        total_in = totals[hour][DIRECTION_IN]
        profile.append({
            "hour": hour,
            "label": f"{hour:02d}:00",
            "in": total_in,
            "out": totals[hour][DIRECTION_OUT],
            "average_in": round(total_in / span, 2) if span else 0.0,
            "share": round(total_in / grand_total * 100, 1) if grand_total else 0.0,
        })
    return profile


def busiest_hours(profile, count=3):
    """The ``count`` busiest hours, most visitors first. Silent hours excluded."""
    active = [row for row in profile if row["in"] > 0]
    return sorted(active, key=lambda r: r["in"], reverse=True)[:count]


def opening_span(profile):
    """First and last hour with any arrivals, or ``None`` when all quiet."""
    active = [row["hour"] for row in profile if row["in"] > 0]
    if not active:
        return None
    return {"first_hour": min(active), "last_hour": max(active)}


def weekday_hour_matrix(start=None, end=None, days=DEFAULT_DAYS, camera=None, event=None):
    """Average arrivals per weekday and hour.

    Each cell divides by how many times that weekday actually occurred in
    the range, so a Tuesday average is not diluted by Fridays.
    """
    start, end = resolve_range(start, end, days)
    buckets = _hour_rows(start, end, camera, event)

    # How many times each weekday falls inside the range.
    occurrences = {index: 0 for index in range(7)}
    cursor = start
    while cursor <= end:
        occurrences[cursor.weekday()] += 1
        cursor += timedelta(days=1)

    sums = {(weekday, hour): 0 for weekday in range(7) for hour in HOURS}
    for (day, hour), counts in buckets.items():
        sums[(day.weekday(), hour)] += counts.get(DIRECTION_IN, 0)

    matrix = []
    for weekday in range(7):
        seen = occurrences[weekday]
        cells = []
        for hour in HOURS:
            total = sums[(weekday, hour)]
            cells.append({
                "hour": hour,
                "in": total,
                "average_in": round(total / seen, 2) if seen else 0.0,
            })
        matrix.append({
            "weekday": weekday,
            "name": WEEKDAY_NAMES[weekday],
            "occurrences": seen,
            "hours": cells,
        })
    return matrix
