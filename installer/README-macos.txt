VisitorsTracker for macOS
=========================

Counts people crossing a line in a camera's view. Everything runs on this
Mac; nothing is uploaded, and no internet connection is needed.

Installing
----------
Drag VisitorsTracker to the Applications folder in this window.

Then open it ONCE like this:

    Right-click VisitorsTracker in Applications > Open > Open

A plain double-click the first time gets you "Apple could not verify
VisitorsTracker is free of malware" and no way forward. That is not a
problem with the app: Apple charges developers a yearly fee to register
apps, and this one is not registered. The right-click is how macOS lets you
make that decision yourself. You only do it once -- after that, double-click
as normal.

Starting it
-----------
The first launch takes about half a minute: it prepares its database and
loads the detector before the window appears. Later launches are quicker.

First run
---------
An administrator account is created for you, and the dashboard shows the
username and password at the top. Use them to sign in at the "admin" link,
add your camera, then change the password. The notice disappears once you
delete this file:

    ~/Library/Application Support/VisitorsTracker/initial-admin.txt

The camera
----------
macOS asks for camera permission the first time the app opens a camera. If
you miss the prompt or say no, turn it back on at:

    System Settings > Privacy & Security > Camera

VisitorsTracker has its own entry there. Quit and reopen the app after
changing it.

Network cameras need no permission. Paste the RTSP address in as the
camera's source.

Where your data is
------------------
    ~/Library/Application Support/VisitorsTracker/

That holds the database, the log and the password file. Replacing the app
does NOT touch it, so your counting history survives an upgrade. Delete that
folder by hand if you want a clean slate.

Two log files are in there. visitorstracker.log is the app's own account of
what it did -- start that one when something looks wrong. output.log is the
raw web-server chatter, useful mostly for a developer.

Seeing it from another device
-----------------------------
The window is the dashboard, and the dashboard is a web page served on this
Mac at 127.0.0.1:8000. It is deliberately not reachable from the network. To
put it on a screen elsewhere, run the server version instead -- see the
project's README.

Which Macs this runs on
-----------------------
There are two images, one per chip, because OpenCV publishes no single
binary covering both:

    VisitorsTracker-1.0.0-arm64.dmg    Apple Silicon (M1 and later)
    VisitorsTracker-1.0.0-x86_64.dmg   Intel

Both need macOS 14 (Sonoma) or newer. That floor is not a choice: the
OpenCV and numpy binaries inside are built against it. The app refuses to
start on an older Mac rather than half-working.

Opening the wrong one gives "You can't open the application because it is
not supported on this type of Mac". Check which you have under
Apple menu > About This Mac, and use the other image.

Removing it
-----------
Drag the app to the Trash. Delete the folder above too if you want the
counting history gone with it.
