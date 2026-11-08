from django.contrib import admin

from .models import Camera, CountEvent


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "is_active", "line_orientation", "line_position")
    list_filter = ("is_active", "line_orientation")
    search_fields = ("name", "source")


@admin.register(CountEvent)
class CountEventAdmin(admin.ModelAdmin):
    list_display = ("camera", "direction", "track_id", "occurred_at")
    list_filter = ("direction", "camera")
    date_hierarchy = "occurred_at"
