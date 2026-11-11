from django.contrib import admin, messages
from django.shortcuts import render
from django.utils import timezone

from .exports import events_csv_response
from .models import Alert, Camera, Crossing, Event
from .reports import parse_date


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "is_active", "line_orientation", "line_position")
    # Editable in the list, so turning a camera off does not mean opening
    # its form and hunting for a tickbox.
    list_editable = ("is_active",)
    list_filter = ("is_active", "line_orientation")
    search_fields = ("name", "source")
    actions = ["enable_cameras", "disable_cameras"]

    @admin.action(description="Enable selected cameras")
    def enable_cameras(self, request, queryset):
        changed = queryset.filter(is_active=False).update(is_active=True)
        self.message_user(
            request,
            f"Enabled {changed} camera(s). The supervisor starts them within "
            f"a pass; without one, press Start."
            if changed else "Those cameras were already enabled.")

    @admin.action(description="Disable selected cameras")
    def disable_cameras(self, request, queryset):
        """Stop them now as well as marking them inactive.

        Waiting for the supervisor's next pass would leave a camera the
        operator just switched off still counting for several seconds.
        """
        from .camera import REGISTRY

        stopped = sum(1 for camera in queryset if REGISTRY.stop(camera.pk))
        changed = queryset.filter(is_active=True).update(is_active=False)
        self.message_user(
            request,
            f"Disabled {changed} camera(s); {stopped} were running and have "
            f"been stopped.")


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ("name", "running_now", "starts_at", "ends_at", "is_active",
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

    actions = ["export_csv", "run_now", "end_now"]

    @admin.display(description="Counting now", boolean=True)
    def running_now(self, obj):
        """Whether crossings are being attributed to this event.

        Not the same as "active": an event can be ticked active and still
        be over, or not started. This is the question people actually
        have, so the list answers it directly.
        """
        return obj.pk in self._running_event_ids()

    def _running_event_ids(self):
        from .models import Camera, active_event_for

        return {
            event.pk
            for event in (active_event_for(camera) for camera in Camera.objects.all())
            if event is not None
        }

    @admin.action(description="Count for this event from now on")
    def run_now(self, request, queryset):
        """Switch counting to one event.

        An event is counted for when it is active and its window covers
        now; where several overlap the most recently started one wins. So
        this starts the chosen event now, which both makes it current and
        makes it win, without touching the others or their figures --
        crossings are attributed by foreign key, so nothing already
        recorded moves.
        """
        if queryset.count() != 1:
            self.message_user(
                request, "Pick exactly one event to count for.",
                level=messages.ERROR)
            return

        event = queryset.first()
        now = timezone.now()

        if event.ends_at and event.ends_at <= now:
            self.message_user(
                request,
                f"{event.name} ended on "
                f"{timezone.localtime(event.ends_at):%d %b %H:%M}. Give it a "
                f"later end date first; this will not quietly extend it.",
                level=messages.ERROR)
            return

        already = event.pk in self._running_event_ids()
        changes = []
        if not event.is_active:
            event.is_active = True
            changes.append("marked active")
        if not already:
            event.starts_at = now
            changes.append("started now")
        event.save()

        if changes:
            self.message_user(
                request,
                f"Counting for {event.name}: {', '.join(changes)}. Crossings "
                f"already recorded keep the event they were counted under.")
        else:
            self.message_user(request, f"{event.name} was already the one counting.")

    @admin.action(description="Stop counting for this event")
    def end_now(self, request, queryset):
        """End an event now, so crossings stop being attributed to it.

        Sets the end to now rather than unticking active, because that is
        what happened: the event finished. Unticking active would leave a
        misleading schedule behind.
        """
        now = timezone.now()
        ended = 0
        for event in queryset:
            if event.ends_at and event.ends_at <= now:
                continue
            event.ends_at = now
            event.save(update_fields=["ends_at"])
            ended += 1
        self.message_user(
            request,
            f"Ended {ended} event(s) at {timezone.localtime(now):%H:%M}. "
            f"Counting falls back to whichever event still covers now, or none."
            if ended else "Those events had already ended.")

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
