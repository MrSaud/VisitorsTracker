#!/bin/bash
#
# Run VisitorsTracker from Terminal, so the counter can reach the webcam.
#
# macOS grants camera access to the *application* that owns the process,
# not to Python. An editor or assistant that launches the server appears
# in System Settings > Privacy & Security > Camera only once it has asked
# for access, and that pane has no way to add one by hand -- so a server
# started from inside one cannot open a webcam at all.
#
# Terminal is normally in that list already. Double-click this file (or
# run it with `open -a Terminal`) and the server, and the camera, belong
# to Terminal instead.
#
# Not needed for an RTSP or HTTP camera: only local capture devices are
# behind that permission.
#
set -e
cd "$(dirname "$0")/.."

echo "VisitorsTracker — starting with the cameras on (COUNTER_AUTOSTART=1)."
echo "Dashboard: http://127.0.0.1:8000/     Ctrl-C to stop."
echo

exec env COUNTER_AUTOSTART=1 .venv/bin/python manage.py runserver 127.0.0.1:8000 --noreload
