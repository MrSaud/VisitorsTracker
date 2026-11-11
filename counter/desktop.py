"""Run VisitorsTracker as a desktop window.

Nothing here is platform-specific: pywebview uses WebKit on macOS and the
Edge WebView2 runtime on Windows, and the server underneath is the same one
on both. ``manage.py build_app`` wraps this in a macOS .app bundle.

On macOS the bundle is not cosmetic. macOS grants camera access to the
*application* that asks for it, and a Python process started from a shell
has no application of its own -- the permission lands on whoever ran it,
which is why the webcam launcher has to go through Terminal.app and why
there is no way to add a plain script to Privacy & Security > Camera.
Inside a bundle the app owns that permission under its own name.

What it runs is the same server as always, on 127.0.0.1:8000 -- which is
this machine only. The window is a view onto a real HTTP server rather than
a replacement for it, so pointing it at 0.0.0.0 would let a phone or a PC on
the network open the same dashboard; it does not do that by default, because
reads are unauthenticated and that would put them on the LAN.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
WINDOW_TITLE = "VisitorsTracker"
WINDOW_SIZE = (1240, 880)
WINDOW_MIN_SIZE = (820, 600)
STARTUP_TIMEOUT = 40.0


def health_url(host=DEFAULT_HOST, port=DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/api/v1/health/"


def dashboard_url(host=DEFAULT_HOST, port=DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/"


def port_is_free(port, host=DEFAULT_HOST) -> bool:
    """Whether the server could bind here.

    SO_REUSEADDR matters: Django's WSGIServer sets allow_reuse_address, so a
    probe without it is stricter than the thing it is testing for. Sockets
    left in TIME_WAIT by the previous run would then read as "busy" and the
    app would refuse to start for a minute after every restart -- on exactly
    the port it had just been using itself.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def ours_is_already_serving(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=2.0) -> bool:
    """Whether a VisitorsTracker server already answers on this port.

    Double-clicking the app twice should not start a second server: the two
    would fight over the camera, since the device opens once. The second
    launch opens a window onto the first one instead.
    """
    try:
        with urllib.request.urlopen(health_url(host, port), timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(payload, dict) and "cameras" in payload


def wait_for_server(url, timeout=STARTUP_TIMEOUT, pause=0.1) -> bool:
    """Block until the server answers, so the window never opens on an error."""
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except urllib.error.HTTPError as error:
            if error.code < 500:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(pause)
    return False


def build_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """A threaded WSGI server that does not block the app from quitting.

    Django's ThreadedWSGIServer sets ``block_on_close``, so closing it waits
    for every request thread to finish. An MJPEG live view is a request that
    never finishes, so the default would hang the app on quit.
    """
    from django.core.servers.basehttp import ThreadedWSGIServer, WSGIRequestHandler
    from django.core.wsgi import get_wsgi_application

    class DesktopWSGIServer(ThreadedWSGIServer):
        daemon_threads = True
        block_on_close = False

    httpd = DesktopWSGIServer((host, int(port)), WSGIRequestHandler, ipv6=False)
    httpd.set_app(get_wsgi_application())
    thread = threading.Thread(target=httpd.serve_forever, name="wsgi", daemon=True)
    thread.start()
    return httpd, thread


def prepare_database():
    """Bring the schema up to date before anything reads it.

    A double-clicked app on a fresh copy has never run ``migrate``, and the
    first thing the supervisor does is ask which cameras are active.
    """
    from django.core.management import call_command

    call_command("migrate", verbosity=0, interactive=False)


def start_cameras():
    """Start the supervisor explicitly, once the schema is known to be there.

    COUNTER_AUTOSTART is deliberately *not* set in the environment: that
    would start cameras from AppConfig.ready, before migrate has run.
    """
    from django.conf import settings

    from .supervisor import maybe_autostart

    settings.COUNTER_AUTOSTART = True
    return maybe_autostart()


def shutdown(httpd=None, supervisor=None):
    """Release the camera and stop serving. Safe to call twice."""
    if supervisor is not None:
        try:
            supervisor.stop()
        except Exception:
            pass

    from .camera import REGISTRY

    try:
        REGISTRY.stop_all()
    except Exception:
        pass

    if httpd is not None:
        try:
            httpd.shutdown()
        finally:
            httpd.server_close()


def production_defaults():
    """Run as a deployed app, not a development one.

    A double-clicked app has no shell, no systemd unit and nobody to export
    anything, so it has to configure itself. Every one of these is a
    setdefault: a server or a developer that sets the variable still wins.

    DEBUG is the one that matters. It serves a traceback containing settings
    to anyone who can reach the port, and it makes Django keep the last
    9000 queries in memory -- pointless for a process meant to run for
    weeks. The app binds 127.0.0.1, so nobody else should reach it anyway;
    this is the second lock, not the first.
    """
    from config.paths import resolve_data_dir

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = resolve_data_dir(project_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    os.environ.setdefault("DJANGO_DEBUG", "0")
    os.environ.setdefault("DJANGO_ALLOWED_HOSTS", f"{DEFAULT_HOST},localhost")
    # Created once and reused, so a restart does not log the operator out.
    # Beside the database, not beside the code: frozen, the code lives in a
    # temporary directory that is deleted when the app exits.
    os.environ.setdefault(
        "DJANGO_SECRET_KEY_FILE", str(data_dir / ".secret_key"))


def start_logging(data_dir):
    """Frozen, there is no console, so a failed launch leaves no trace.

    Everything below -- a camera that will not open, a port in use, a
    migration that failed -- is logged already; it just has nowhere to go
    once the console window is gone. This gives it somewhere.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        data_dir / "visitorstracker.log", maxBytes=2_000_000, backupCount=3,
        encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # The crash is the case this log exists for, and it is the one case
    # plain logging misses: an uncaught exception goes to stderr, and a
    # windowed build has no stderr. Without this the app dies leaving an
    # empty log file, which is what it did the first time it was frozen.
    def log_and_die(kind, value, traceback):
        root.critical("unhandled exception", exc_info=(kind, value, traceback))
        sys.__excepthook__(kind, value, traceback)

    sys.excepthook = log_and_die

    # Threads get their own hook; a camera worker dying must not be silent.
    def log_thread_death(args):
        root.critical("unhandled exception in %s", args.thread_name,
                      exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = log_thread_death

    root.info("VisitorsTracker starting, data directory %s", data_dir)
    return handler.baseFilename


def detach_output(data_dir, limit=5_000_000):
    """Give the window's stdout and stderr somewhere of their own.

    A windowed build inherits whatever handles launched it. Started from
    Finder those are /dev/null and nothing is lost, but started from a
    terminal they are that terminal's pipe -- and when the terminal goes
    away, the development server's next request log writes to a closed pipe
    and the app dies without a word. It did exactly that here. Pointing the
    file descriptors at a file of our own removes the dependency on who
    started us, and has the side benefit that request logs survive.

    The descriptors are redirected rather than only ``sys.stdout``, because
    OpenCV and the Python runtime also write to them directly.
    """
    path = data_dir / "output.log"
    try:
        # It is appended to for the life of an install, so cap it rather
        # than let a long-running counter fill the disk with request logs.
        if path.exists() and path.stat().st_size > limit:
            path.unlink()
        handle = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError:
        return None  # An unwritable data directory is reported elsewhere.

    try:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
    except OSError:
        handle.close()
        return None

    sys.stdout = handle
    sys.stderr = handle
    return path


class DesktopApi:
    """What the page may ask the application to do for it.

    One thing, so far: save a file. The window has no address bar and no
    Back button, so a download link that navigates leaves the operator
    staring at raw CSV with no way out -- which is exactly what the
    Download CSV buttons used to do. A webview is not a browser and must
    not be asked to behave like one.
    """

    def save_file(self, filename, content):
        """Ask where to put it, then write it. Returns what happened."""
        import webview

        windows = webview.windows
        if not windows:
            return {"saved": False, "reason": "no window"}

        chosen = windows[0].create_file_dialog(
            webview.SAVE_DIALOG, save_filename=filename or "export.csv")
        if not chosen:
            return {"saved": False, "reason": "cancelled"}

        # Older pywebview returns a sequence here, newer a string.
        if isinstance(chosen, (list, tuple)):
            chosen = chosen[0]

        try:
            Path(chosen).write_text(content, encoding="utf-8")
        except OSError as error:
            return {"saved": False, "reason": str(error)}
        return {"saved": True, "path": str(chosen)}


def main(argv=None):
    production_defaults()

    from config.paths import is_frozen, resolve_data_dir

    if is_frozen():
        data_dir = resolve_data_dir(Path(__file__).resolve().parent.parent)
        start_logging(data_dir)
        detach_output(data_dir)

    import django

    django.setup()

    host, port = DEFAULT_HOST, int(os.environ.get("COUNTER_DESKTOP_PORT", DEFAULT_PORT))
    httpd = supervisor = None

    if ours_is_already_serving(host, port):
        # Another copy is already running and already holds the camera.
        url = dashboard_url(host, port)
    else:
        if not port_is_free(port, host):
            raise SystemExit(
                f"Port {port} is busy with something that is not VisitorsTracker. "
                f"Stop it, or set COUNTER_DESKTOP_PORT to another port."
            )
        prepare_database()
        # A fresh install has no accounts, and a camera can only be added
        # through the admin. Without this it cannot be configured at all.
        from .firstrun import ensure_admin

        ensure_admin(resolve_data_dir(Path(__file__).resolve().parent.parent))
        httpd, _thread = build_server(host, port)
        if not wait_for_server(health_url(host, port)):
            shutdown(httpd)
            raise SystemExit("The server did not start.")
        supervisor = start_cameras()
        url = dashboard_url(host, port)

    import webview

    webview.create_window(
        WINDOW_TITLE,
        url,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN_SIZE,
        js_api=DesktopApi(),
    )
    try:
        webview.start()
    finally:
        shutdown(httpd, supervisor)


if __name__ == "__main__":
    main()
