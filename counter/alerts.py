"""Telling somebody when counting breaks, or a room fills up.

A dashboard only works while somebody is looking at it. The two things
worth interrupting a person for are the camera that has stopped counting
and the hall that is filling past its capacity -- one because the data is
being lost as it happens, the other because it is a safety limit.

Alerts here are *conditions*, not messages. A condition opens when
something goes wrong, stays open while it is still wrong, and resolves
when it is put right, which is what keeps a check every thirty seconds
from becoming a message every thirty seconds. Three rules follow from
that, and they are the whole design:

* **Nothing is sent inside the grace period.** A camera that blinks for
  two seconds is not worth a phone buzzing. A condition that clears before
  ``COUNTER_ALERT_AFTER_SECONDS`` is resolved having told nobody.
* **Getting worse is not getting fixed.** "Near capacity" becoming "over
  capacity" closes one alert and opens another, and no "resolved" message
  goes out for the first.
* **Recovery is news too.** Whoever was told it broke is told when it is
  working again, with how long it was down.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections import namedtuple
from datetime import timedelta

from django.conf import settings
from django.core.mail import send_mail
from django.db import transaction
from django.utils import timezone

from .health import SEVERITY, camera_health
from .models import Alert, Camera, Event

logger = logging.getLogger(__name__)

DEFAULT_GRACE_SECONDS = 60.0
DEFAULT_MIN_SEVERITY = 1
WEBHOOK_TIMEOUT = 5.0
ALLOWED_WEBHOOK_SCHEMES = ("http", "https")

# One thing that is wrong now. ``key`` is the identity of the problem, so
# the same camera failing twice in a row is one alert, not two.
Condition = namedtuple(
    "Condition", "key kind state level message camera event")

Outcome = namedtuple("Outcome", "opened escalated resolved sent failed")


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def alerts_enabled() -> bool:
    return bool(getattr(settings, "COUNTER_ALERTS_ENABLED", False))


def grace_seconds() -> float:
    return float(getattr(settings, "COUNTER_ALERT_AFTER_SECONDS",
                         DEFAULT_GRACE_SECONDS) or 0)


def min_severity() -> int:
    return int(getattr(settings, "COUNTER_ALERT_MIN_SEVERITY",
                       DEFAULT_MIN_SEVERITY))


def webhook_url() -> str:
    return (getattr(settings, "COUNTER_ALERT_WEBHOOK", "") or "").strip()


def email_recipients() -> list[str]:
    raw = getattr(settings, "COUNTER_ALERT_EMAILS", "") or ""
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def alert_channels() -> list[str]:
    """Which ways out are configured. Empty means alerts go nowhere."""
    channels = []
    if webhook_url():
        channels.append("webhook")
    if email_recipients():
        channels.append("email")
    return channels


def site_label() -> str:
    return (getattr(settings, "COUNTER_SITE_NAME", "") or "VisitorsTracker").strip()


# ---------------------------------------------------------------------------
# what is wrong right now
# ---------------------------------------------------------------------------

def health_conditions(cameras=None, registry=None, now=None):
    """Cameras that are not working, at or above the configured severity."""
    if registry is None:
        from .camera import REGISTRY as registry
    cameras = list(Camera.objects.all() if cameras is None else cameras)
    threshold = min_severity()

    conditions = []
    for camera in cameras:
        row = camera_health(camera, registry.get(camera.pk))
        severity = SEVERITY.get(row["state"], 0)
        if severity < threshold:
            continue

        detail = (row.get("detail") or "").strip()
        message = f"{camera.name} is {row['state']}."
        if detail:
            message = f"{message} {detail}"
        conditions.append(Condition(
            key=f"health:camera:{camera.pk}",
            kind=Alert.KIND_HEALTH,
            state=row["state"],
            level=(Alert.LEVEL_CRITICAL if severity >= 2
                   else Alert.LEVEL_WARNING),
            message=message,
            camera=camera,
            event=None,
        ))
    return conditions


def capacity_conditions(now=None):
    """Running events at or over the capacity someone set for them."""
    now = now or timezone.now()
    events = (Event.objects.filter(is_active=True, capacity__isnull=False)
              .prefetch_related("cameras"))

    conditions = []
    for event in events:
        if not event.is_running(now):
            continue
        state = event.capacity_state()
        if state is None or state["level"] == "ok":
            continue

        if state["level"] == "over":
            message = (
                f"{event.name} is over capacity: {state['inside']} inside, "
                f"limit {state['capacity']} ({state['percent']}%)."
            )
            level = Alert.LEVEL_CRITICAL
        else:
            remaining = state["remaining"]
            message = (
                f"{event.name} is near capacity: {state['inside']} inside of "
                f"{state['capacity']} ({state['percent']}%), "
                f"{remaining} place{'' if remaining == 1 else 's'} left."
            )
            level = Alert.LEVEL_WARNING

        conditions.append(Condition(
            key=f"capacity:event:{event.pk}",
            kind=Alert.KIND_CAPACITY,
            state=state["level"],
            level=level,
            message=message,
            camera=None,
            event=event,
        ))
    return conditions


def current_conditions(cameras=None, registry=None, now=None):
    now = now or timezone.now()
    return health_conditions(cameras, registry, now) + capacity_conditions(now)


# ---------------------------------------------------------------------------
# opening, escalating and resolving
# ---------------------------------------------------------------------------

def _open_alert(condition, now):
    return Alert.objects.create(
        kind=condition.kind,
        key=condition.key,
        state=condition.state,
        level=condition.level,
        message=condition.message,
        camera=condition.camera,
        event=condition.event,
        started_at=now,
        last_seen_at=now,
    )


@transaction.atomic
def reconcile_alerts(conditions, now=None):
    """Bring the open alerts into line with what is wrong now."""
    now = now or timezone.now()
    open_alerts = {a.key: a for a in Alert.objects.filter(resolved_at__isnull=True)}
    seen = set()
    opened, escalated, resolved = [], [], []

    for condition in conditions:
        seen.add(condition.key)
        existing = open_alerts.get(condition.key)

        if existing is None:
            opened.append(_open_alert(condition, now))
            continue

        if existing.state != condition.state:
            # Near capacity becoming over capacity is a different problem,
            # not a fixed one. Close the old quietly and open the new.
            existing.resolved_at = now
            existing.superseded = True
            existing.save(update_fields=["resolved_at", "superseded"])
            escalated.append(_open_alert(condition, now))
            continue

        # Still the same problem; keep the detail current without saying
        # anything, since the detail carries a countdown and a frame count.
        existing.last_seen_at = now
        existing.message = condition.message
        existing.save(update_fields=["last_seen_at", "message"])

    for key, alert in open_alerts.items():
        if key in seen or alert.superseded:
            continue
        alert.resolved_at = now
        alert.save(update_fields=["resolved_at"])
        resolved.append(alert)

    return opened, escalated, resolved


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------

def _post_webhook(url, payload):
    scheme = url.split("://", 1)[0].lower()
    if scheme not in ALLOWED_WEBHOOK_SCHEMES:
        raise ValueError(f"webhook URL must be http or https, not {scheme!r}")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "VisitorsTracker"},
    )
    with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT) as response:
        return response.status


def notification_text(alert, resolved=False, now=None):
    """What a person reads. Short enough for a phone's lock screen."""
    now = now or timezone.now()
    site = site_label()
    if resolved:
        return (f"[{site}] Resolved after {alert.duration_label(now)}: "
                f"{alert.message}")
    tag = "CRITICAL" if alert.level == Alert.LEVEL_CRITICAL else "Warning"
    return f"[{site}] {tag}: {alert.message}"


def deliver(alert, resolved=False, now=None):
    """Send one alert. Returns ``(sent, error)``; an error means try again."""
    now = now or timezone.now()
    text = notification_text(alert, resolved=resolved, now=now)
    url = webhook_url()
    recipients = email_recipients()

    if not url and not recipients:
        # Nothing to retry, so this counts as handled -- but it is recorded
        # as an error so no screen can imply somebody was told.
        return False, "No alert channel configured."

    errors = []
    delivered = False

    if url:
        try:
            _post_webhook(url, {
                "text": text,
                "alert": {
                    "id": alert.pk,
                    "kind": alert.kind,
                    "key": alert.key,
                    "state": alert.state,
                    "level": alert.level,
                    "message": alert.message,
                    "resolved": bool(resolved),
                    "started_at": alert.started_at.isoformat(),
                    "resolved_at": (alert.resolved_at.isoformat()
                                    if alert.resolved_at else None),
                    "camera": alert.camera.name if alert.camera else None,
                    "event": alert.event.name if alert.event else None,
                },
            })
            delivered = True
        except (urllib.error.URLError, OSError, ValueError) as exc:
            errors.append(f"webhook: {exc}")

    if recipients:
        try:
            send_mail(
                subject=text[:200],
                message=f"{text}\n\nStarted {alert.started_at:%Y-%m-%d %H:%M:%S}.",
                from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
                recipient_list=recipients,
                fail_silently=False,
            )
            delivered = True
        except Exception as exc:  # pragma: no cover - any backend may raise
            errors.append(f"email: {exc}")

    return delivered, "; ".join(errors)


def pending(now=None):
    """``(to_announce, to_clear)``: alerts owed a message, grace observed."""
    now = now or timezone.now()
    cutoff = now - timedelta(seconds=grace_seconds())

    to_announce = list(Alert.objects.filter(
        notified_at__isnull=True,
        resolved_at__isnull=True,
        started_at__lte=cutoff,
    ))
    # Only where the opening was actually announced: nobody needs telling
    # that something they never heard about is better.
    to_clear = list(Alert.objects.filter(
        resolved_at__isnull=False,
        notified_at__isnull=False,
        resolved_notified_at__isnull=True,
        superseded=False,
    ))
    return to_announce, to_clear


def send_pending(now=None):
    """Deliver what is owed. Returns ``(sent, failed)`` alert lists."""
    now = now or timezone.now()
    to_announce, to_clear = pending(now)
    sent, failed = [], []

    for alert, resolved in ([(a, False) for a in to_announce]
                            + [(a, True) for a in to_clear]):
        delivered, error = deliver(alert, resolved=resolved, now=now)

        if not delivered and error and "No alert channel" not in error:
            # A webhook timeout must not lose the alert; leave it owed.
            alert.delivery_error = error
            alert.save(update_fields=["delivery_error"])
            failed.append(alert)
            logger.warning("alert %s not delivered: %s", alert.pk, error)
            continue

        stamp = "resolved_notified_at" if resolved else "notified_at"
        setattr(alert, stamp, now)
        alert.delivery_error = error
        alert.save(update_fields=[stamp, "delivery_error"])
        sent.append(alert)
        if delivered:
            logger.info("alert %s delivered: %s", alert.pk,
                        notification_text(alert, resolved, now))

    return sent, failed


def run_checks(cameras=None, registry=None, now=None):
    """One full pass: look, reconcile, then say what is owed."""
    now = now or timezone.now()
    conditions = current_conditions(cameras, registry, now)
    opened, escalated, resolved = reconcile_alerts(conditions, now)
    sent, failed = send_pending(now)
    return Outcome(opened, escalated, resolved, sent, failed)
