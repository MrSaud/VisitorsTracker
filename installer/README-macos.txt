VisitorsTracker for macOS
=========================

Counts people crossing a line in a camera's view. Everything runs on this
Mac; nothing is uploaded, and no internet connection is needed.

Installing
----------
1. Drag VisitorsTracker to the Applications folder in this window.
   Do this first. Opening it from inside the disk image makes the next
   step harder than it needs to be.

2. Double-click it. You will get

       "Apple could not verify VisitorsTracker is free of malware"

   and the only buttons will be Move to Trash and Cancel. Click Cancel.
   This is expected, and it is step 2 of 3, not a failure.

3. Open  System Settings > Privacy & Security  and scroll down to the
   Security section. There will be a line saying VisitorsTracker was
   blocked. Click  Open Anyway,  confirm, and it starts.

You do this once. After that it opens by double-click like anything else.

On macOS 14 and earlier there is a shortcut: right-click the app in
Applications and choose Open, which offers the same override in a dialog.
Apple removed that route in macOS 15, which is why the steps above go
through System Settings instead -- they work on every version.

If the Security section shows no such line, this does the same thing from
Terminal and works on all versions:

    xattr -dr com.apple.quarantine /Applications/VisitorsTracker.app

Why any of this happens
-----------------------
macOS runs an unknown app only if Apple has seen and approved it in
advance, which costs the developer a paid Apple Developer account and a
submission for each release. This app is signed but not submitted, so macOS
has nothing to check it against and asks you to make the decision instead.

The warning says Apple cannot verify the app, not that anything is wrong
with it. Nothing about the app changes when you click Open Anyway; you are
only telling macOS you trust where you got it from.

To remove the step entirely, the app has to be notarised with an Apple
Developer account. That is a change to how it is built, not something you
can switch on here.

Starting it
-----------
The first launch takes about half a minute: it prepares its database and
loads the detector before the window appears. Later launches are quicker.

First run
---------
An administrator account is created for you, and the dashboard shows the
username and password at the top, with a Copy password button next to it --
the password is generated and long, so copy it rather than retyping it. Use
them to sign in at the "admin" link, add your camera, then change the
password. Once you are signed in, the notice grows a "Delete it now"
button that removes the password file for you -- it asks twice, because
the file holds the only copy of that password. Deleting it by hand does
the same thing:

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

    VisitorsTracker-1.0.0-arm64.dmg    Apple Silicon (M1 and later),
                                       macOS 11 (Big Sur) or newer
    VisitorsTracker-1.0.0-x86_64.dmg   Intel,
                                       macOS 12 (Monterey) or newer

Those floors come from the OpenCV and numpy binaries inside, not from a
preference. The line at the top of this file states the one for the build
you are holding. An older Mac will not start it at all, rather than
half-working.

Opening the wrong one gives "You can't open the application because it is
not supported on this type of Mac". Check which you have under
Apple menu > About This Mac, and use the other image.

Removing it
-----------
Drag the app to the Trash. Delete the folder above too if you want the
counting history gone with it.
