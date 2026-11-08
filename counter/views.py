"""Dashboard, MJPEG stream and the JSON API."""

import csv

from django.conf import settings
from django.http import (
    Http404,
    HttpResponse,
    HttpResponseForbidden,
    StreamingHttpResponse,
)
from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .camera import REGISTRY, mjpeg_frames
from .daywindow import occupancy_window_start
from .models import Camera, CountEvent
from .permissions import _is_operator
from .reports import DEFAULT_DAYS, daily_counts, parse_date, resolve_range, summarise


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
def camera_events(request, camera_id: int):
    camera = get_object_or_404(Camera, pk=camera_id)
    try:
        limit = min(500, max(1, int(request.query_params.get("limit", 50))))
    except (TypeError, ValueError):
        limit = 50
    events = camera.events.all()[:limit]
    return Response(
        {
            "camera_id": camera.pk,
            "count": len(events),
            "events": [
                {
                    "id": event.pk,
                    "track_id": event.track_id,
                    "direction": event.direction,
                    "occurred_at": event.occurred_at.isoformat(),
                }
                for event in events
            ],
        }
    )


@api_view(["POST"])
def camera_reset(request, camera_id: int):
    """Delete this camera's recorded crossings. Does not stop the worker."""
    camera = get_object_or_404(Camera, pk=camera_id)
    deleted, _ = CountEvent.objects.filter(camera=camera).delete()
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

    return Response(
        {
            "totals": {"in": total_in, "out": total_out},
            "occupancy": {
                "inside": occupied,
                "net": occupancy_net,
                "since": occupancy_window_start().isoformat(),
                "resets_at": settings.COUNTER_OCCUPANCY_RESET_AT,
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

    return start, end, days, camera


@api_view(["GET"])
def daily_report(request):
    """Crossings per calendar date, in the project's local timezone.

    ``?days=30`` (default), or an explicit ``?start=&end=`` window.
    ``?camera=<id>`` narrows to one camera, ``?format=csv`` downloads it.
    """
    try:
        start, end, days, camera = _daily_params(request)
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    rows = daily_counts(start=start, end=end, days=days, camera=camera)
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
        "summary": summarise(rows),
        "days": rows,
    })
