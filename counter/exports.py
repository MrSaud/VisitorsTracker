"""Turning report rows into a CSV file somebody can open in a spreadsheet.

This is the admin's export, which always concerns events and may cover
several at once, so every row names its event. The API's ``?format=csv``
is a different shape on purpose: it can be scoped to a camera or to
nothing at all, where an event column would be blank on every row.
"""

from __future__ import annotations

import csv
import re

from django.http import HttpResponse

from .reports import daily_counts, resolve_range

DAILY_COLUMNS = ["event", "date", "in", "out", "net", "peak_inside", "peak_at"]


def safe_filename(*parts) -> str:
    """A filename a browser and a filesystem will both accept.

    Event names are free text: "Doors open 9–5 (east)" would otherwise
    produce a download nobody can save.
    """
    joined = "_".join(str(p) for p in parts if p)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", joined)
    # Collapse dot runs and drop leading ones: "..-..-etc" is a filename
    # nobody meant to ask for.
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip("-_.")
    return (cleaned or "export")[:120]


def event_rows(event, start=None, end=None):
    """Daily rows for one event, as ``(event_name, row)`` pairs.

    Without an explicit window it uses the event's own: exporting "the
    conference" should mean the conference, not the last thirty days.
    """
    if start is None and end is None:
        start = event.starts_at.date()
        end = event.ends_at.date() if event.ends_at else None

    rows = daily_counts(start=start, end=end, event=event)
    return [(event.name, row) for row in rows]


def write_daily_csv(response, pairs):
    writer = csv.writer(response)
    writer.writerow(DAILY_COLUMNS)
    for event_name, row in pairs:
        writer.writerow([
            event_name,
            row["date"],
            row["in"],
            row["out"],
            row["net"],
            row.get("peak", ""),
            row.get("peak_at") or "",
        ])
    return response


def events_csv_response(events, start=None, end=None):
    """A CSV download covering one or several events."""
    pairs = []
    for event in events:
        pairs.extend(event_rows(event, start=start, end=end))

    if len(events) == 1:
        name = safe_filename("visitors", events[0].name)
    else:
        name = safe_filename("visitors", f"{len(events)}-events")

    if start or end:
        window_start, window_end = resolve_range(start, end, None)
        name = safe_filename(name, window_start.isoformat(), window_end.isoformat())

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{name}.csv"'
    return write_daily_csv(response, pairs)
