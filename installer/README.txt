VisitorsTracker
===============

Counts people crossing a line in a camera's view.

Starting it
-----------
Use the Start Menu entry, or the desktop shortcut if you asked for one.
The first launch takes about half a minute: it prepares its database and
loads the detector before the window appears. Later launches are quick.

First run
---------
An administrator account is created for you, and the dashboard shows the
username and password at the top, with a Copy password button beside it. Use them to sign in at the "admin" link,
add your camera, then change the password. The notice disappears once you
delete this file:

    %LOCALAPPDATA%\VisitorsTracker\initial-admin.txt

The camera
----------
If the camera will not open, check:

    Settings > Privacy & security > Camera
    "Let desktop apps access your camera" must be on

Network cameras need no permission. Paste the RTSP address in as the
camera's source.

Where your data is
------------------
    %LOCALAPPDATA%\VisitorsTracker\

That holds the database, the log and the password file. Uninstalling does
NOT delete it, so your counting history survives an upgrade. Remove that
folder by hand if you want a clean slate.

Seeing it from another device
-----------------------------
The app listens on 127.0.0.1 only, which is this machine and nothing else.
Opening it to the network is a deliberate step and means anyone who can
reach this machine can see the counts and the live camera view.

"Windows protected your PC"
---------------------------
The installer is not code-signed, so SmartScreen warns the first time.
Click "More info", then "Run anyway". A signing certificate is issued only
to a registered business and costs a few hundred a year.
