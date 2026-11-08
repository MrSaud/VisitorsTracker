"""Dashboard, MJPEG stream and the JSON API."""

from django.http import Http404, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .camera import REGISTRY, mjpeg_frames
from .models import Camera, CountEvent


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def dashboard(request):
    cameras = list(Camera.objects.all())
    running = {row["camera_id"] for row in REGISTRY.stats() if row["running"]}
    for camera in cameras:
        camera.is_running = camera.pk in running
        camera.totals_cache = camera.totals()
    return render(
        request,
        "counter/dashboard.html",
        {"cameras": cameras, "has_cameras": bool(cameras)},
    )


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------

def video_feed(request, camera_id: int):
    """Live annotated MJPEG stream for a running camera."""
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
    return Response(
        {
            "totals": {
                "in": total_in,
                "out": total_out,
                "inside": max(0, total_in - total_out),
            },
            "today": {
                "in": sum(c["today"]["in"] for c in per_camera),
                "out": sum(c["today"]["out"] for c in per_camera),
            },
            "running_cameras": sum(1 for c in per_camera if c["running"]),
            "cameras": per_camera,
        }
    )
