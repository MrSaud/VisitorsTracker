"""Entry point for the frozen application.

PyInstaller runs its entry script as ``__main__``, not as a module inside
its package, so every relative import in that file fails with "attempted
relative import with no known parent package". Pointing the build at a
script that imports the real module keeps ``counter.desktop`` a package
module, where ``from .supervisor import ...`` means what it says.

A checkout does not need this: `python -m counter.desktop` already runs it
as a module, which is what the .app bundle and the Windows .bat do.
"""

from counter.desktop import main

if __name__ == "__main__":
    main()
