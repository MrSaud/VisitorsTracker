"""API and streaming routes for the counter app."""

from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("stats/", views.overall_stats, name="overall-stats"),
    path("alerts/", views.alert_list, name="alert-list"),
    path("events/", views.event_list, name="event-list"),
    path("events/current/", views.current_event, name="event-current"),
    path("events/<int:event_id>/", views.event_detail, name="event-detail"),
    path("daily/", views.daily_report, name="daily-report"),
    path("hourly/", views.hourly_report, name="hourly-report"),
    path("cameras/", views.camera_list, name="camera-list"),
    path("cameras/<int:camera_id>/", views.camera_detail, name="camera-detail"),
    path("cameras/<int:camera_id>/start/", views.camera_start, name="camera-start"),
    path("cameras/<int:camera_id>/stop/", views.camera_stop, name="camera-stop"),
    path("cameras/<int:camera_id>/crossings/", views.camera_crossings, name="camera-crossings"),
    path("cameras/<int:camera_id>/reset/", views.camera_reset, name="camera-reset"),
    path("cameras/<int:camera_id>/stream/", views.video_feed, name="camera-stream"),
    path("cameras/<int:camera_id>/snapshot/", views.snapshot, name="camera-snapshot"),
]
