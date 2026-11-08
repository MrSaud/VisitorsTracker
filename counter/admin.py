from django.contrib import admin

from .models import Camera, Crossing, Event


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "is_active", "line_orientation", "line_position")
    list_filter = ("is_active", "line_orientation")
    search_fields = ("name", "source")


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ("name", "starts_at", "ends_at", "is_active",
                    "attendance", "occupancy_now")
    list_filter = ("is_active",)
    search_fields = ("name", "notes")
    date_hierarchy = "starts_at"
    filter_horizontal = ("cameras",)
    readonly_fields = ("created_at",)
    fieldsets = (
        (None, {"fields": ("name", "is_active", "notes")}),
        ("When", {"fields": ("starts_at", "ends_at")}),
        ("Where", {
            "fields": ("cameras",),
            "description": "Leave empty to count every camera.",
        }),
        ("Limits", {"fields": ("capacity",)}),
        ("Record", {"fields": ("created_at",)}),
    )

    @admin.display(description="Attendance")
    def attendance(self, obj):
        return obj.totals()["in"]

    @admin.display(description="Inside now")
    def occupancy_now(self, obj):
        return obj.totals()["inside"]


@admin.register(Crossing)
class CrossingAdmin(admin.ModelAdmin):
    list_display = ("camera", "event", "direction", "track_id", "occurred_at")
    list_filter = ("direction", "camera", "event")
    date_hierarchy = "occurred_at"
