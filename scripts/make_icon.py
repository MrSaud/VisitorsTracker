#!/usr/bin/env python
"""Turn a square source image into the app's .icns.

    python scripts/make_icon.py artwork.png

The two platforms want opposite things from the same artwork.

macOS 26 and later draw their own rounded container and place the icon
inside it, so a pre-rounded icon with a margin shows as a rounded square
inside a rounded square, shrunk and ringed in grey. The .icns therefore
bleeds to the edges and lets the system do the rounding -- the artwork's own
corners are transparent, so the background is painted into them.

Windows does no such masking: it draws the .ico exactly as given. Full-bleed
artwork would read as a square tile among rounded neighbours, so the .ico
keeps the artwork's own rounded corners and transparency.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

CANVAS = 1024
SIZES = (16, 32, 128, 256, 512)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def solid_bbox(image):
    """The opaque artwork, ignoring the soft edge of a drop shadow."""
    alpha = np.array(image)[:, :, 3] > 200
    rows = np.where(alpha.any(axis=1))[0]
    cols = np.where(alpha.any(axis=0))[0]
    if not len(rows) or not len(cols):
        raise SystemExit("That image has no opaque pixels.")
    return cols.min(), rows.min(), cols.max(), rows.max()


def full_bleed(art):
    """The artwork edge to edge, opaque, with no corners of its own.

    The artwork's rounded corners are transparent, and the canvas has to be
    square and opaque. Cropping inside the arc would work but throws away a
    slice of the design at every edge, so instead the background is painted
    into the corners by inpainting from the pixels around them -- the corner
    of this kind of icon is plain gradient, which extends cleanly.
    """
    import cv2        # already a dependency of the counter itself

    x0, y0, x1, y1 = solid_bbox(art)
    shape = art.crop((x0, y0, x1 + 1, y1 + 1)).resize(
        (CANVAS, CANVAS), Image.LANCZOS)

    rgba = np.array(shape)
    holes = (rgba[:, :, 3] <= 250).astype(np.uint8)
    share = holes.mean()
    if share > 0.15:
        raise SystemExit(
            f"{share:.0%} of the artwork is transparent; that is not a "
            "rounded square with a solid background.")
    print(f"painting the background into {share*100:.1f}% of the canvas "
          "(the artwork's own corners)")

    if holes.any():
        # Dilate a little so the antialiased rim is repainted too, not
        # smeared into the result as a pale edge.
        mask = cv2.dilate(holes, np.ones((5, 5), np.uint8), iterations=2)
        filled = cv2.inpaint(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR),
                             mask, 12, cv2.INPAINT_TELEA)
        rgba[:, :, :3] = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def rounded(art):
    """The artwork as it came, square and centred, corners still transparent."""
    x0, y0, x1, y1 = solid_bbox(art)
    shape = art.crop((x0, y0, x1 + 1, y1 + 1))
    side = max(shape.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(shape, ((side - shape.width) // 2,
                                   (side - shape.height) // 2))
    return square.resize((CANVAS, CANVAS), Image.LANCZOS)


def write_icns(master):
    """macOS. Needs iconutil, so it is skipped elsewhere."""
    if not shutil.which("iconutil"):
        print("iconutil not found (not macOS): skipping AppIcon.icns")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in SIZES:
            for scale in (1, 2):
                px = base * scale
                name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
                master.resize((px, px), Image.LANCZOS).save(iconset / name)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset),
             "-o", str(ASSETS / "AppIcon.icns")],
            check=True)
    return True


def write_ico(master):
    """Windows. Pillow writes a multi-resolution .ico with no outside tool."""
    master.save(ASSETS / "AppIcon.ico",
                sizes=[(px, px) for px in ICO_SIZES])


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)

    source = Path(argv[1])
    if not source.exists():
        raise SystemExit(f"No such file: {source}")

    art = Image.open(source).convert("RGBA")
    if art.width != art.height:
        print(f"warning: {source.name} is {art.width}x{art.height}, not square")

    ASSETS.mkdir(exist_ok=True)

    mac_master = full_bleed(art)
    mac_master.save(ASSETS / "AppIcon.png")
    opaque = (np.array(mac_master)[:, :, 3] > 250).mean()
    print(f"macOS master: {CANVAS}x{CANVAS}, {opaque*100:.0f}% opaque "
          "(full bleed; macOS rounds the corners itself)")
    made_icns = write_icns(mac_master)

    win_master = rounded(art)
    transparent = (np.array(win_master)[:, :, 3] <= 250).mean()
    print(f"Windows master: {CANVAS}x{CANVAS}, {transparent*100:.0f}% "
          "transparent (its own rounded corners; Windows does not mask)")
    write_ico(win_master)

    print()
    print(f"wrote {ASSETS / 'AppIcon.png'}")
    if made_icns:
        print(f"wrote {ASSETS / 'AppIcon.icns'}")
    print(f"wrote {ASSETS / 'AppIcon.ico'}")
    print("\nRebuild the launcher to pick it up:  python manage.py build_app")


if __name__ == "__main__":
    main(sys.argv)
