"""URL configuration for the VisitorsTracker project."""

from django.contrib import admin
from django.urls import include, path

from counter.views import dashboard

admin.site.site_header = "VisitorsTracker Admin"
admin.site.site_title = "VisitorsTracker Admin"
admin.site.index_title = "VisitorsTracker administration"

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("admin/", admin.site.urls),
    path("api/v1/", include("counter.urls")),
]
