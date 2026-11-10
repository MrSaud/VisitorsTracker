from django.conf import settings
from django.contrib.staticfiles.views import serve as serve_static
from django.urls import re_path
"""URL configuration for the VisitorsTracker project."""

from django.contrib import admin
from django.urls import include, path

from counter.views import cameras_page, dashboard

admin.site.site_header = "VisitorsTracker Admin"
admin.site.site_title = "VisitorsTracker Admin"
admin.site.index_title = "VisitorsTracker administration"

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("cameras/", cameras_page, name="cameras"),
    path("admin/", admin.site.urls),
    path("api/v1/", include("counter.urls")),
]

if settings.COUNTER_SERVE_STATIC:
    # ``insecure=True`` because this bypasses collectstatic and reads from
    # the app directories. Django calls it insecure for a public server;
    # for a desktop app on 127.0.0.1 it is the only thing that will serve
    # the admin's CSS at all, since DEBUG is off.
    urlpatterns += [
        re_path(r"^static/(?P<path>.*)$", serve_static, {"insecure": True}),
    ]
