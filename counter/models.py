"""Cameras, the events they cover, and the crossings they record."""

from datetime import timedelta

from django.db import models
from django.utils import timezone

from .daywindow import occupancy_window_start
from .vision import DIRECTION_IN, DIRECTION_OUT, HORIZONTAL, VERTICAL

# How long a counting lease is good for without being renewed. Long enough
# that a slow pass does not drop it, short enough that a process which dies
# without releasing it frees the camera promptly.
LEASE_SECONDS = 45
LEASE_RENEW_EVERY = 15


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

    # A camera may only be counted by one process at a time. Two processes
    # reading one RTSP stream both succeed -- the camera happily serves
    # several readers -- and both write crossings, so the counts come out
    # multiplied by the number of processes, silently. That is exactly what
    # `gunicorn -w 4` with COUNTER_AUTOSTART would do. Measured: four
    # processes over a clip containing 3 crossings recorded 12.
    lease_owner = models.CharField(max_length=120, blank=True, default="")
    lease_expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.source})"

    def resolved_source(self):
        """Webcam indexes must be passed to OpenCV as ints, URLs as strings."""
        raw = (self.source or "").strip()
        return int(raw) if raw.isdigit() else raw

    # -- the counting lease -----------------------------------------------

    def lease_held_by(self, now=None) -> str:
        """Who is counting this camera, or "" if nobody is."""
        if not self.lease_owner or self.lease_expires_at is None:
            return ""
        now = now or timezone.now()
        return self.lease_owner if self.lease_expires_at > now else ""

    def claim_lease(self, owner, seconds=None, now=None) -> bool:
        """Take the lease, unless a different live one exists.

        One UPDATE with the conditions in its WHERE clause, checked by row
        count. Deliberately not select_for_update: that is a no-op on
        SQLite, so it would look like a lock while providing none. A single
        conditional UPDATE is atomic on both.
        """
        seconds = LEASE_SECONDS if seconds is None else seconds
        now = now or timezone.now()
        expires = now + timedelta(seconds=seconds)

        claimed = Camera.objects.filter(pk=self.pk).filter(
            models.Q(lease_owner="")
            | models.Q(lease_owner=owner)
            | models.Q(lease_expires_at__isnull=True)
            | models.Q(lease_expires_at__lte=now)
        ).update(lease_owner=owner, lease_expires_at=expires)

        if claimed:
            self.lease_owner = owner
            self.lease_expires_at = expires
        return bool(claimed)

    def renew_lease(self, owner, seconds=None, now=None) -> bool:
        """Push the expiry out. Fails if the lease was taken meanwhile."""
        seconds = LEASE_SECONDS if seconds is None else seconds
        now = now or timezone.now()
        expires = now + timedelta(seconds=seconds)
        renewed = Camera.objects.filter(
            pk=self.pk, lease_owner=owner).update(lease_expires_at=expires)
        if renewed:
            self.lease_expires_at = expires
        return bool(renewed)

    def take_lease_from(self, holder, owner, seconds=None, now=None) -> bool:
        """Take a lease from a named holder, conditional on it still being theirs.

        For the one case the caller can prove is safe: the holder is a
        process on this machine that no longer exists. Still conditional on
        the holder, so it cannot stomp a lease that changed hands in the
        meantime.
        """
        seconds = LEASE_SECONDS if seconds is None else seconds
        now = now or timezone.now()
        expires = now + timedelta(seconds=seconds)
        taken = Camera.objects.filter(pk=self.pk, lease_owner=holder).update(
            lease_owner=owner, lease_expires_at=expires)
        if taken:
            self.lease_owner = owner
            self.lease_expires_at = expires
        return bool(taken)

    def release_lease(self, owner) -> bool:
        """Give it up, so another process need not wait for it to expire."""
        released = Camera.objects.filter(pk=self.pk, lease_owner=owner).update(
            lease_owner="", lease_expires_at=None)
        if released:
            self.lease_owner = ""
            self.lease_expires_at = None
        return bool(released)

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

    def peak_today(self):
        """The fullest moment so far today on this camera.

        Lives here for symmetry with :meth:`occupancy`, but the walk itself
        is in :mod:`counter.reports`; imported late because reports needs
        this module.
        """
        from .reports import peak_occupancy

        return peak_occupancy(camera=self, days=1)

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

    def peak_occupancy(self):
        """The fullest single moment of the event.

        Attendance says how many came; this says how many were in at once,
        which is the figure a capacity, a floor plan or a safety officer is
        actually about. Counted per day and maxed across them, so a
        multi-day event is not the sum of its mornings.
        """
        from .reports import peak_occupancy

        return peak_occupancy(
            start=timezone.localtime(self.starts_at).date(),
            end=timezone.localtime(self.ends_at).date(),
            event=self,
        )

    def capacity_state(self, peak=None):
        """How full it is, when a capacity has been set.

        ``peak`` may be passed in by a caller that has already walked the
        crossings, so a payload carrying both does not walk them twice.
        """
        if not self.capacity:
            return None
        inside = self.totals()["inside"]
        share = inside / self.capacity
        peak = peak or self.peak_occupancy()
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
            # Whether it was ever breached, not just whether it is now.
            # "We were fine when you looked" is not the same as "we were
            # never over".
            "peak": peak["peak"],
            "peak_at": peak["at"],
            "peak_percent": round(peak["peak"] / self.capacity * 100, 1),
            "breached": peak["peak"] >= self.capacity,
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


class Alert(models.Model):
    """Something worth telling somebody about, and whether we did.

    An alert is a *condition*, not an event: it opens when something goes
    wrong, stays open while it is still wrong, and resolves when it is put
    right. That is what makes "the hall was over capacity from 11:40 to
    11:52" answerable afterwards, and it is what stops a check every thirty
    seconds becoming a message every thirty seconds.

    Rows are written as soon as a condition is seen, but notifying waits
    for ``COUNTER_ALERT_AFTER_SECONDS``: a camera that blinks for two
    seconds is not worth anyone's phone buzzing, and a condition that
    clears inside the grace period is resolved having told nobody.
    """

    KIND_HEALTH = "health"
    KIND_CAPACITY = "capacity"
    KIND_CHOICES = [(KIND_HEALTH, "Camera health"), (KIND_CAPACITY, "Capacity")]

    LEVEL_WARNING = "warning"
    LEVEL_CRITICAL = "critical"
    LEVEL_CHOICES = [(LEVEL_WARNING, "Warning"), (LEVEL_CRITICAL, "Critical")]

    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    key = models.CharField(
        max_length=120, db_index=True,
        help_text="Identifies the condition, so the same problem is one alert.",
    )
    state = models.CharField(
        max_length=32,
        help_text='The state that triggered it: "offline", "over", and so on.',
    )
    level = models.CharField(max_length=16, choices=LEVEL_CHOICES,
                             default=LEVEL_WARNING)
    message = models.TextField()

    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True,
                               blank=True, related_name="alerts")
    event = models.ForeignKey("Event", on_delete=models.SET_NULL, null=True,
                              blank=True, related_name="alerts")

    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    last_seen_at = models.DateTimeField(default=timezone.now)
    resolved_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # Delivery is recorded rather than assumed: an alert nobody received is
    # worse than no alert, because the dashboard would imply it was sent.
    notified_at = models.DateTimeField(null=True, blank=True)
    resolved_notified_at = models.DateTimeField(null=True, blank=True)
    delivery_error = models.TextField(blank=True)
    superseded = models.BooleanField(
        default=False,
        help_text=(
            "Closed because the condition got worse, not because it was "
            "fixed. No recovery message is owed for these."
        ),
    )

    class Meta:
        ordering = ["-started_at", "-id"]
        indexes = [models.Index(fields=["key", "resolved_at"])]
        constraints = [
            models.UniqueConstraint(
                fields=["key"], condition=models.Q(resolved_at__isnull=True),
                name="one_open_alert_per_key",
            )
        ]

    def __str__(self):
        return f"{self.get_level_display()}: {self.message}"

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None

    @property
    def delivered(self) -> bool:
        """Whether a notification actually went out, not merely that we tried."""
        return bool(self.notified_at) and not self.delivery_error

    def duration(self, now=None):
        return (self.resolved_at or now or timezone.now()) - self.started_at

    def duration_label(self, now=None):
        seconds = int(self.duration(now).total_seconds())
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m"
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
