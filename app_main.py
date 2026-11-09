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
    """Borrow the console that launched us, on Windows.

    A windowed build has no stdout, so a management command would run
    with its prompts and its output going nowhere -- createsuperuser in
    particular would appear to hang. AttachConsole(-1) takes the console
    of the process that started us, which is the cmd window somebody
    typed into. There is nothing to attach to when double-clicked, and
    that is fine: nobody is reading it.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        ATTACH_PARENT_PROCESS = -1
        if not ctypes.windll.kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
            return False
        for name, stream in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
            try:
                setattr(sys, name, open("CON" + ("IN$" if stream == "r" else "OUT$"),
                                        stream))
            except OSError:
                pass
        return True
    except Exception:
        return False


def main():
    if len(sys.argv) > 1:
        attach_console()
        from counter.desktop import production_defaults

        production_defaults()
        from django.core.management import execute_from_command_line

        execute_from_command_line(sys.argv)
        return

    from counter.desktop import main as run_window

    run_window()


if __name__ == "__main__":
    main()
