from django.contrib import admin
from django.shortcuts import render

from .exports import events_csv_response
from .models import Alert, Camera, Crossing, Event
from .reports import parse_date


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

    actions = ["export_csv"]

    @admin.action(description="Export visitor data to CSV")
    def export_csv(self, request, queryset):
        """Download daily counts for the selected events.

        An intermediate page because the dates matter: an action on its
        own cannot ask anything, and "export this event" usually means a
        particular window rather than whatever the default happens to be.
        Left blank, each event exports its own run.
        """
        events = list(queryset)
        context = {
            **self.admin_site.each_context(request),
            "events": events,
            "title": "Export visitor data",
            "opts": self.model._meta,
        }

        if "apply" not in request.POST:
            return render(request, "admin/export_events.html", context)

        try:
            start = parse_date(request.POST.get("start") or None)
            end = parse_date(request.POST.get("end") or None)
        except ValueError as exc:
            context["error"] = str(exc)
            return render(request, "admin/export_events.html", context)

        if start and end and start > end:
            context["error"] = "The start date is after the end date."
            return render(request, "admin/export_events.html", context)

        return events_csv_response(events, start=start, end=end)

    @admin.display(description="Attendance")
    def attendance(self, obj):
        return obj.totals()["in"]

    @admin.display(description="Inside now")
    def occupancy_now(self, obj):
        return obj.totals()["inside"]


@admin.register(Crossing)
class CrossingAdmin(admin.ModelAdmin):
    list_display = ("camera", "event", "direction", "track_id", "occurred_at",
                    "sureness", "line_offset", "speed")
    list_filter = ("direction", "camera", "event")
    date_hierarchy = "occurred_at"

    @admin.display(description="Confidence", ordering="confidence")
    def sureness(self, obj):
        """Plain words beside the number, so a weak count is obvious.

        The thresholds are descriptive, not a judgement: nothing is
        discarded, it is only labelled.
        """
        if obj.confidence is None:
            return "—"
        if obj.confidence >= 0.75:
            return f"{obj.confidence:.2f} strong"
        if obj.confidence >= 0.5:
            return f"{obj.confidence:.2f} fair"
        return f"{obj.confidence:.2f} weak"


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
