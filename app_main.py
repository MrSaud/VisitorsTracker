"""Entry point for the frozen application.

PyInstaller runs its entry script as ``__main__``, not as a module inside
its package, so every relative import in that file fails with "attempted
relative import with no known parent package". Pointing the build at a
script that imports the real module keeps ``counter.desktop`` a package
module, where ``from .supervisor import ...`` means what it says.

With arguments it runs a management command instead of opening the
window, because otherwise a frozen install has no way to create its first
administrator: the dashboard's admin link leads to a login nobody can
pass, and the staff-only controls are unreachable for good.

    VisitorsTracker.exe createsuperuser
    VisitorsTracker.exe check_setup --benchmark
    VisitorsTracker.exe backup_db

A checkout needs none of this: manage.py is right there.
"""

import sys


def attach_console():
    """Get a console for a windowed build, so prompts have somewhere to go.

    A windowed build has sys.stdin set to None, and ``input()`` raises
    "lost sys.stdin" the moment anything asks a question --
    createsuperuser being the one that matters.

    Attaching to the parent's console is the tidy answer and it does not
    work here: PyInstaller's onefile bootloader starts the real program as
    a *child*, so ATTACH_PARENT_PROCESS reaches the bootloader, which is
    itself windowed and has no console to borrow. So fall back to opening
    one. It is a second window rather than the shell you typed into, which
    is not elegant, but a prompt you can answer beats a traceback.

    Returns the kind of console obtained, or "" for none.
    """
    if not sys.platform.startswith("win"):
        return "inherited"

    import ctypes

    kernel32 = ctypes.windll.kernel32
    ATTACH_PARENT_PROCESS = -1

    kind = ""
    if kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
        kind = "attached"
    elif kernel32.AllocConsole():
        kind = "allocated"
    else:
        return ""

    # The streams are None until something is opened on the new console.
    for name, path, mode in (("stdin", "CONIN$", "r"),
                             ("stdout", "CONOUT$", "w"),
                             ("stderr", "CONOUT$", "w")):
        try:
            setattr(sys, name, open(path, mode, buffering=1))
        except OSError:
            return ""
    return kind


def wants_a_reader(argv):
    """Whether a person is waiting to read the result.

    The pause exists because an allocated console closes with the process.
    It must not happen when nothing can press a key: a scripted
    `createsuperuser --noinput` would hang forever, which is exactly what
    it did the first time this ran unattended.
    """
    import os

    if os.environ.get("VISITORSTRACKER_NO_PAUSE") == "1":
        return False
    return not any(a in ("--noinput", "--no-input") for a in argv)


def main():
    if len(sys.argv) > 1:
        console = attach_console()
        from counter.desktop import production_defaults

        production_defaults()
        from django.core.management import execute_from_command_line

        try:
            execute_from_command_line(sys.argv)
        finally:
            if console == "allocated" and wants_a_reader(sys.argv):
                # The console goes away with the process, taking the
                # result with it. Hold it open long enough to read.
                try:
                    input("\nDone. Press Enter to close this window.")
                except Exception:
                    pass
        return

    from counter.desktop import main as run_window

    run_window()


if __name__ == "__main__":
    main()
