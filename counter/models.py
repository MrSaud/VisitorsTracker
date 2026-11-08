"""Cameras, the events they cover, and the crossings they record."""

from django.db import models
from django.utils import timezone

from .daywindow import occupancy_window_start
from .vision import DIRECTION_IN, DIRECTION_OUT, HORIZONTAL, VERTICAL


class Camera(models.Model):
    """A video source plus the counting line drawn across it."""

    ORIENTATION_CHOICES = [
        (HORIZONTAL, "Horizontal line (people move up/down)"),
        (VERTICAL, "Vertical line (people move left/right)"),
    ]

    name = models.CharField(max_length=120)
    source = models.CharField(
        max_length=500,
        default="0",
        help_text=(
            'Webcam index ("0"), a video file path, or an RTSP/HTTP stream URL.'
        ),
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Only active cameras are started by the counter.",
    )
    line_position = models.FloatField(
        default=0.5,
        help_text="Where the counting line sits, as a fraction of the frame (0.0 - 1.0).",
    )
    line_orientation = models.CharField(
        max_length=16, choices=ORIENTATION_CHOICES, default=HORIZONTAL
    )
    invert_direction = models.BooleanField(
        default=False,
        help_text="Tick this if IN and OUT come out swapped for this camera.",
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.source})"

    def resolved_source(self):
        """Webcam indexes must be passed to OpenCV as ints, URLs as strings."""
        raw = (self.source or "").strip()
        return int(raw) if raw.isdigit() else raw

    # -- derived counts ---------------------------------------------------

    def _direction_counts(self, queryset):
        rows = queryset.values("direction").order_by().annotate(
            total=models.Count("id"))
        counts = {row["direction"]: row["total"] for row in rows}
        return counts.get(DIRECTION_IN, 0), counts.get(DIRECTION_OUT, 0)

    def totals(self):
        """Lifetime crossings. Footfall, not occupancy."""
        total_in, total_out = self._direction_counts(self.crossings)
        return {"in": total_in, "out": total_out}

    def occupancy_window(self, now=None):
        """When the current occupancy count began, and why.

        Normally the daily reset. While an event is running, counting
        starts when the event did -- nobody inside beforehand is part of
        it. A multi-day event still gets the daily reset, because the
        later of the two boundaries wins and the venue empties overnight.
        """
        daily = occupancy_window_start(now)
        event = active_event_for(self, now)
        if event is not None and event.starts_at > daily:
            return event.starts_at, event
        return daily, event

    def occupancy(self, now=None):
        """How many are inside now.

        ``net`` is reported unclamped alongside ``inside``: a persistently
        negative net means exits outnumber entries, which is a sign the
        camera direction is inverted rather than a sign of an empty room.
        """
        since, event = self.occupancy_window(now)
        total_in, total_out = self._direction_counts(
            self.crossings.filter(occurred_at__gte=since))
        net = total_in - total_out
        return {
            "in": total_in,
            "out": total_out,
            "inside": max(0, net),
            "net": net,
            "since": since.isoformat(),
            "event_id": event.pk if event else None,
            "event_name": event.name if event else None,
        }

    def totals_today(self):
        start = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
        rows = (
            self.crossings.filter(occurred_at__gte=start)
            .values("direction")
            .order_by()
            .annotate(total=models.Count("id"))
        )
        counts = {row["direction"]: row["total"] for row in rows}
        return {
            "in": counts.get(DIRECTION_IN, 0),
            "out": counts.get(DIRECTION_OUT, 0),
        }


class Event(models.Model):
    """Something being counted: an exhibition, a conference, an open day.

    An event scopes counting in time, and optionally to particular cameras.
    While one is running, crossings are stamped with it, so attendance can
    be reported per event rather than only per date.

    Leaving ``cameras`` empty means the event covers every camera, which is
    the common case for a single-entrance venue.
    """

    name = models.CharField(max_length=200)
    starts_at = models.DateTimeField()
    ends_at = models.DateTimeField()
    cameras = models.ManyToManyField(
        Camera,
        blank=True,
        related_name="events",
        help_text="Leave empty to count every camera.",
    )
    capacity = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Optional. When set, the dashboard warns as occupancy approaches it.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Untick to stop attributing crossings to this event.",
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-starts_at"]
        indexes = [models.Index(fields=["starts_at", "ends_at"])]

    def __str__(self):
        return self.name

    def clean(self):
        from django.core.exceptions import ValidationError

        if self.starts_at and self.ends_at and self.ends_at <= self.starts_at:
            raise ValidationError({"ends_at": "The event must end after it starts."})

    # -- state ------------------------------------------------------------

    def is_running(self, now=None) -> bool:
        now = now or timezone.now()
        return bool(self.is_active and self.starts_at <= now < self.ends_at)

    def has_finished(self, now=None) -> bool:
        return (now or timezone.now()) >= self.ends_at

    def covers_camera(self, camera) -> bool:
        """An event with no cameras listed covers all of them."""
        if not self.pk:
            return True
        selected = list(self.cameras.all())
        return not selected or any(c.pk == camera.pk for c in selected)

    @property
    def duration(self):
        return self.ends_at - self.starts_at

    # -- counts -----------------------------------------------------------

    def _direction_counts(self, queryset):
        rows = queryset.values("direction").order_by().annotate(
            total=models.Count("id"))
        counts = {row["direction"]: row["total"] for row in rows}
        return counts.get(DIRECTION_IN, 0), counts.get(DIRECTION_OUT, 0)

    def totals(self):
        """Attendance for this event: entries, exits and who is still in."""
        total_in, total_out = self._direction_counts(self.crossings)
        return {
            "in": total_in,
            "out": total_out,
            "inside": max(0, total_in - total_out),
            "net": total_in - total_out,
        }

    def capacity_state(self):
        """How full it is, when a capacity has been set."""
        if not self.capacity:
            return None
        inside = self.totals()["inside"]
        share = inside / self.capacity
        if share >= 1.0:
            level = "over"
        elif share >= 0.9:
            level = "near"
        else:
            level = "ok"
        return {
            "capacity": self.capacity,
            "inside": inside,
            "percent": round(share * 100, 1),
            "level": level,
            "remaining": max(0, self.capacity - inside),
        }


def active_event_for(camera, now=None):
    """The event a crossing on ``camera`` belongs to, or None.

    Where events overlap, the most recently started one wins: scheduling a
    session inside a longer programme is the normal case, and the inner one
    is the more specific answer.
    """
    now = now or timezone.now()
    candidates = (
        Event.objects.filter(is_active=True, starts_at__lte=now, ends_at__gt=now)
        .prefetch_related("cameras")
        .order_by("-starts_at")
    )
    for event in candidates:
        if event.covers_camera(camera):
            return event
    return None


class Crossing(models.Model):
    """One person crossing the counting line in one direction."""

    DIRECTION_CHOICES = [
        (DIRECTION_IN, "In"),
        (DIRECTION_OUT, "Out"),
    ]

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="crossings")
    event = models.ForeignKey(
        "Event", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="crossings",
        help_text="The event running when this crossing happened, if any.",
    )
    track_id = models.PositiveIntegerField(
        help_text="Tracker-local id; unique per counter run, not across restarts."
    )
    direction = models.CharField(max_length=8, choices=DIRECTION_CHOICES, db_index=True)
    occurred_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-occurred_at", "-id"]
        indexes = [
            models.Index(fields=["camera", "occurred_at"]),
            models.Index(fields=["event", "occurred_at"]),
        ]

    def __str__(self):
        return f"{self.camera.name}: {self.direction} #{self.track_id}"
