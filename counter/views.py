"""Dashboard, MJPEG stream and the JSON API."""

import csv

from django.conf import settings
from django.http import (
    Http404,
    HttpResponse,
    HttpResponseForbidden,
    StreamingHttpResponse,
)
from django.db.models import Max
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .camera import REGISTRY, mjpeg_frames
from .daywindow import occupancy_window_start
from .health import STATUS_OK, build_report
from .models import Camera, Crossing, Event, active_event_for
from .permissions import _is_operator
from .reports import (
    DEFAULT_DAYS,
    busiest_hours,
    daily_counts,
    hourly_profile,
    opening_span,
    parse_date,
    resolve_range,
    summarise,
    weekday_hour_matrix,
)


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def dashboard(request):
    cameras = list(Camera.objects.all())
    running = {row["camera_id"] for row in REGISTRY.stats() if row["running"]}
    for camera in cameras:
        camera.is_running = camera.pk in running
    return render(
        request,
        "counter/dashboard.html",
        {
            "cameras": cameras,
            "has_cameras": bool(cameras),
            # Controls are hidden rather than left to fail with a 403.
            "can_operate": _is_operator(request),
        },
    )


def _reads_allowed(request) -> bool:
    return getattr(settings, "COUNTER_PUBLIC_READS", True) or _is_operator(request)


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------

def video_feed(request, camera_id: int):
    """Live annotated MJPEG stream for a running camera."""
    if not _reads_allowed(request):
        return HttpResponseForbidden("Authentication required to view this stream.")

    worker = REGISTRY.get(camera_id)
    if worker is None:
        raise Http404("camera is not running")

    response = StreamingHttpResponse(
        mjpeg_frames(worker),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["X-Accel-Buffering"] = "no"
    return response


def snapshot(request, camera_id: int):
    """The most recent annotated frame as a single JPEG."""
    if not _reads_allowed(request):
        return HttpResponseForbidden("Authentication required to view this camera.")

    worker = REGISTRY.get(camera_id)
    if worker is None:
        raise Http404("camera is not running")
    jpeg = worker.snapshot()
    if jpeg is None:
        raise Http404("no frame captured yet")
    return HttpResponse(jpeg, content_type="image/jpeg")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

def _camera_payload(camera, worker=None):
    payload = {
        "id": camera.pk,
        "name": camera.name,
        "source": camera.source,
        "is_active": camera.is_active,
        "line_position": camera.line_position,
        "line_orientation": camera.line_orientation,
        "invert_direction": camera.invert_direction,
        "totals": camera.totals(),
        "occupancy": camera.occupancy(),
        "today": camera.totals_today(),
        "running": bool(worker and worker.is_alive()),
    }
    if worker is not None:
        payload["live"] = worker.stats()
    return payload


@api_view(["GET"])
def camera_list(request):
    cameras = Camera.objects.all()
    return Response(
        {"cameras": [_camera_payload(c, REGISTRY.get(c.pk)) for c in cameras]}
    )


@api_view(["GET"])
def camera_detail(request, camera_id: int):
    camera = get_object_or_404(Camera, pk=camera_id)
    return Response(_camera_payload(camera, REGISTRY.get(camera.pk)))


@api_view(["POST"])
def camera_start(request, camera_id: int):
    camera = get_object_or_404(Camera, pk=camera_id)
    worker = REGISTRY.start(camera)

    # Give the capture a moment so an unopenable source reports immediately.
    worker.join(timeout=1.5)
    if worker.error:
        return Response(
            {"detail": worker.error, "camera_id": camera.pk},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(_camera_payload(camera, worker))


@api_view(["POST"])
def camera_stop(request, camera_id: int):
    camera = get_object_or_404(Camera, pk=camera_id)
    stopped = REGISTRY.stop(camera.pk)
    return Response({"camera_id": camera.pk, "stopped": stopped})


@api_view(["GET"])
def camera_crossings(request, camera_id: int):
    camera = get_object_or_404(Camera, pk=camera_id)
    try:
        limit = min(500, max(1, int(request.query_params.get("limit", 50))))
    except (TypeError, ValueError):
        limit = 50
    crossings = camera.crossings.all()[:limit]
    return Response(
        {
            "camera_id": camera.pk,
            "count": len(crossings),
            "crossings": [
                {
                    "id": crossing.pk,
                    "track_id": crossing.track_id,
                    "direction": crossing.direction,
                    "occurred_at": crossing.occurred_at.isoformat(),
                    "event_id": crossing.event_id,
                }
                for crossing in crossings
            ],
        }
    )


@api_view(["POST"])
def camera_reset(request, camera_id: int):
    """Delete this camera's recorded crossings. Does not stop the worker."""
    camera = get_object_or_404(Camera, pk=camera_id)
    deleted, _ = Crossing.objects.filter(camera=camera).delete()
    return Response({"camera_id": camera.pk, "deleted": deleted})


@api_view(["GET"])
def overall_stats(request):
    cameras = Camera.objects.all()
    per_camera = [_camera_payload(c, REGISTRY.get(c.pk)) for c in cameras]
    total_in = sum(c["totals"]["in"] for c in per_camera)
    total_out = sum(c["totals"]["out"] for c in per_camera)

    # Occupancy is summed per camera from the daily reset, never derived
    # from lifetime totals, which drift upward as exits are missed.
    occupied = sum(c["occupancy"]["inside"] for c in per_camera)
    occupancy_net = sum(c["occupancy"]["net"] for c in per_camera)

    # Say what occupancy is actually measured from. While an event runs it
    # counts from the event, not the daily reset, and a tile reading
    # "since 00:00" would then be wrong.
    scoped = {c["occupancy"]["event_name"] for c in per_camera
              if c["occupancy"]["event_name"]}
    if len(scoped) == 1:
        scope_label = f"since {scoped.pop()} began"
    elif scoped:
        scope_label = "since each event began"
    else:
        scope_label = f"since {settings.COUNTER_OCCUPANCY_RESET_AT}"

    return Response(
        {
            "totals": {"in": total_in, "out": total_out},
            "occupancy": {
                "inside": occupied,
                "net": occupancy_net,
                "since": occupancy_window_start().isoformat(),
                "resets_at": settings.COUNTER_OCCUPANCY_RESET_AT,
                "scope": scope_label,
            },
            "today": {
                "in": sum(c["today"]["in"] for c in per_camera),
                "out": sum(c["today"]["out"] for c in per_camera),
            },
            "running_cameras": sum(1 for c in per_camera if c["running"]),
            "cameras": per_camera,
        }
    )


# ---------------------------------------------------------------------------
# per-date reporting
# ---------------------------------------------------------------------------

def _daily_params(request):
    """Pull and validate the reporting window from the query string."""
    params = request.query_params if hasattr(request, "query_params") else request.GET

    start = parse_date(params.get("start"))
    end = parse_date(params.get("end"))

    raw_days = (params.get("days") or "").strip()
    if raw_days:
        try:
            days = int(raw_days)
        except ValueError:
            raise ValueError(f"'{raw_days}' is not a whole number of days")
        if days < 1:
            raise ValueError("days must be at least 1")
    else:
        days = DEFAULT_DAYS

    camera = None
    raw_camera = (params.get("camera") or "").strip()
    if raw_camera:
        try:
            camera = Camera.objects.get(pk=int(raw_camera))
        except (ValueError, Camera.DoesNotExist):
            raise ValueError(f"no camera with id '{raw_camera}'")

    event = None
    raw_event = (params.get("event") or "").strip()
    if raw_event:
        try:
            event = Event.objects.get(pk=int(raw_event))
        except (ValueError, Event.DoesNotExist):
            raise ValueError(f"no event with id '{raw_event}'")

    return start, end, days, camera, event


@api_view(["GET"])
def daily_report(request):
    """Crossings per calendar date, in the project's local timezone.

    ``?days=30`` (default), or an explicit ``?start=&end=`` window.
    ``?camera=<id>`` narrows to one camera, ``?format=csv`` downloads it.
    """
    try:
        start, end, days, camera, event = _daily_params(request)
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    rows = daily_counts(start=start, end=end, days=days, camera=camera, event=event)
    window_start, window_end = resolve_range(start, end, days)

    params = request.query_params
    if (params.get("format") or "").lower() == "csv":
        label = f"{window_start.isoformat()}_{window_end.isoformat()}"
        if camera is not None:
            label = f"camera{camera.pk}_{label}"
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = (
            f'attachment; filename="visitors_{label}.csv"'
        )
        writer = csv.writer(response)
        writer.writerow(["date", "in", "out", "net"])
        for row in rows:
            writer.writerow([row["date"], row["in"], row["out"], row["net"]])
        return response

    return Response({
        "start": window_start.isoformat(),
        "end": window_end.isoformat(),
        "camera_id": camera.pk if camera else None,
        "camera_name": camera.name if camera else None,
        "event_id": event.pk if event else None,
        "event_name": event.name if event else None,
        "summary": summarise(rows),
        "days": rows,
    })


@api_view(["GET"])
def hourly_report(request):
    """Arrivals by hour of day, for staffing decisions.

    Same window parameters as the daily report. ``?matrix=1`` adds a
    weekday-by-hour breakdown, which is what you want when Sunday and
    Friday look nothing alike. ``?format=csv`` downloads the profile.
    """
    try:
        start, end, days, camera, event = _daily_params(request)
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    profile = hourly_profile(start=start, end=end, days=days, camera=camera, event=event)
    window_start, window_end = resolve_range(start, end, days)

    params = request.query_params
    if (params.get("format") or "").lower() == "csv":
        label = f"{window_start.isoformat()}_{window_end.isoformat()}"
        if camera is not None:
            label = f"camera{camera.pk}_{label}"
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = (
            f'attachment; filename="visitors_by_hour_{label}.csv"'
        )
        writer = csv.writer(response)
        writer.writerow(["hour", "in", "out", "average_in", "share_percent"])
        for row in profile:
            writer.writerow([row["label"], row["in"], row["out"],
                             row["average_in"], row["share"]])
        return response

    payload = {
        "start": window_start.isoformat(),
        "end": window_end.isoformat(),
        "camera_id": camera.pk if camera else None,
        "camera_name": camera.name if camera else None,
        "event_id": event.pk if event else None,
        "event_name": event.name if event else None,
        "days": (window_end - window_start).days + 1,
        "busiest": busiest_hours(profile),
        "opening_span": opening_span(profile),
        "hours": profile,
    }

    if (params.get("matrix") or "").lower() in ("1", "true", "yes"):
        payload["matrix"] = weekday_hour_matrix(
            start=start, end=end, days=days, camera=camera, event=event)

    return Response(payload)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def _last_event_times(cameras):
    """Most recent crossing per camera, in one query."""
    rows = (
        Crossing.objects.filter(camera__in=cameras)
        .values("camera_id")
        .order_by()
        .annotate(last=Max("occurred_at"))
    )
    return {row["camera_id"]: row["last"] for row in rows}


@api_view(["GET"])
def health(request):
    """Whether the cameras are actually working.

    Always 200 by default so a dashboard can poll it. Monitoring tools that
    alert on status codes should pass ``?strict=1``, which returns 503 when
    anything is wrong.
    """
    cameras = list(Camera.objects.all())
    report = build_report(cameras, REGISTRY, _last_event_times(cameras))

    strict = (request.query_params.get("strict") or "").lower() in ("1", "true", "yes")
    code = status.HTTP_200_OK
    if strict and report["status"] != STATUS_OK:
        code = status.HTTP_503_SERVICE_UNAVAILABLE

    return Response(report, status=code)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def _event_payload(event, now=None):
    now = now or timezone.now()
    running = event.is_running(now)
    payload = {
        "id": event.pk,
        "name": event.name,
        "starts_at": event.starts_at.isoformat(),
        "ends_at": event.ends_at.isoformat(),
        "is_active": event.is_active,
        "running": running,
        "finished": event.has_finished(now),
        "cameras": [{"id": c.pk, "name": c.name} for c in event.cameras.all()],
        "covers_all_cameras": not event.cameras.exists(),
        "attendance": event.totals(),
        "capacity": event.capacity_state(),
        "notes": event.notes,
    }
    if running:
        payload["seconds_remaining"] = max(
            0, int((event.ends_at - now).total_seconds()))
    return payload


@api_view(["GET"])
def event_list(request):
    """Scheduled events, newest first.

    ``?running=1`` narrows to what is happening now; ``?camera=<id>`` to
    events covering a particular camera.
    """
    now = timezone.now()
    events = Event.objects.prefetch_related("cameras")

    raw_camera = (request.query_params.get("camera") or "").strip()
    if raw_camera:
        try:
            camera = Camera.objects.get(pk=int(raw_camera))
        except (ValueError, Camera.DoesNotExist):
            return Response({"detail": f"no camera with id '{raw_camera}'"},
                            status=status.HTTP_400_BAD_REQUEST)
        events = [e for e in events if e.covers_camera(camera)]

    only_running = (request.query_params.get("running") or "").lower() in (
        "1", "true", "yes")
    if only_running:
        events = [e for e in events if e.is_running(now)]

    return Response({
        "count": len(list(events)),
        "events": [_event_payload(e, now) for e in events],
    })


@api_view(["GET"])
def event_detail(request, event_id: int):
    event = get_object_or_404(
        Event.objects.prefetch_related("cameras"), pk=event_id)
    payload = _event_payload(event)
    payload["hours"] = hourly_profile(
        start=timezone.localtime(event.starts_at).date(),
        end=timezone.localtime(event.ends_at).date(),
        event=event,
    )
    payload["busiest"] = busiest_hours(payload["hours"])
    return Response(payload)


@api_view(["GET"])
def current_event(request):
    """What is running right now, across all cameras."""
    now = timezone.now()
    running = [e for e in Event.objects.prefetch_related("cameras")
               if e.is_running(now)]
    return Response({
        "running": bool(running),
        "events": [_event_payload(e, now) for e in running],
    })
