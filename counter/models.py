"""Cameras and the visitor-crossing events they produce."""

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
        total_in, total_out = self._direction_counts(self.events)
        return {"in": total_in, "out": total_out}

    def occupancy(self):
        """How many are inside now, counted from the daily reset.

        ``net`` is reported unclamped alongside ``inside``: a persistently
        negative net means exits outnumber entries, which is a sign the
        camera direction is inverted rather than a sign of an empty room.
        """
        since = occupancy_window_start()
        total_in, total_out = self._direction_counts(
            self.events.filter(occurred_at__gte=since))
        net = total_in - total_out
        return {
            "in": total_in,
            "out": total_out,
            "inside": max(0, net),
            "net": net,
            "since": since.isoformat(),
        }

    def totals_today(self):
        start = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
        rows = (
            self.events.filter(occurred_at__gte=start)
            .values("direction")
            .order_by()
            .annotate(total=models.Count("id"))
        )
        counts = {row["direction"]: row["total"] for row in rows}
        return {
            "in": counts.get(DIRECTION_IN, 0),
            "out": counts.get(DIRECTION_OUT, 0),
        }


class CountEvent(models.Model):
    """One person crossing the counting line in one direction."""

    DIRECTION_CHOICES = [
        (DIRECTION_IN, "In"),
        (DIRECTION_OUT, "Out"),
    ]

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="events")
    track_id = models.PositiveIntegerField(
        help_text="Tracker-local id; unique per counter run, not across restarts."
    )
    direction = models.CharField(max_length=8, choices=DIRECTION_CHOICES, db_index=True)
    occurred_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-occurred_at", "-id"]
        indexes = [models.Index(fields=["camera", "occurred_at"])]

    def __str__(self):
        return f"{self.camera.name}: {self.direction} #{self.track_id}"
