from django.contrib import admin

from .models import Alert, Camera, Crossing, Event


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


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("started_at", "level", "kind", "state", "message",
                    "for_how_long", "was_delivered")
    list_filter = ("kind", "level", "state", "camera", "event")
    search_fields = ("message", "key")
    date_hierarchy = "started_at"
    readonly_fields = ("kind", "key", "state", "level", "message", "camera",
                       "event", "started_at", "last_seen_at", "resolved_at",
                       "notified_at", "resolved_notified_at", "delivery_error",
                       "superseded")

    def has_add_permission(self, request):
        # Alerts are observations, not something to type in by hand.
        return False

    @admin.display(description="For", ordering="started_at")
    def for_how_long(self, obj):
        return obj.duration_label() + ("" if obj.is_open else " (resolved)")

    @admin.display(description="Sent", boolean=True)
    def was_delivered(self, obj):
        return obj.delivered
