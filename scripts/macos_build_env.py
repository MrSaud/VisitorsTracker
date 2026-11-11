#!/usr/bin/env python3
"""Create a build virtualenv for one Mac architecture.

Why this exists rather than a plain `pip install -r requirements.txt`:

pip picks the most specific wheel the *running* Mac accepts. numpy publishes
two macOS wheels per architecture of the same version -- one built against an
old SDK (macosx_10_13 / macosx_11_0) and one against a new one
(macosx_14_0) -- and on a current Mac pip takes the macosx_14_0 build. That
single choice dragged this app's floor from macOS 11 up to macOS 14, with
nothing anywhere saying so.

So the two architecture-specific wheels are chosen here by name, fetched
from PyPI, and checked against the digest PyPI publishes for them. Everything
else is ordinary pure-Python dependency resolution.

    python3 scripts/macos_build_env.py --arch x86_64 \
        --interpreter /path/to/standalone/python/bin/python3.12 \
        --venv /tmp/venv-x86_64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

# The wheel to take for each architecture, by the platform tag in its name.
# These are the oldest-SDK builds each project publishes, which is what keeps
# the app's deployment target low.
# (package, version, interpreter tag, platform tag). The interpreter tag is
# needed as well as the platform one: numpy publishes a macosx_11_0_arm64
# wheel for every Python it supports, so the platform tag alone matches six.
WHEELS = {
    "arm64": [
        ("numpy", "2.5.4", "cp312", "macosx_11_0_arm64"),
        ("opencv-contrib-python-headless", "4.10.0.84", "abi3", "macosx_11_0_arm64"),
    ],
    "x86_64": [
        ("numpy", "2.5.4", "cp312", "macosx_10_13_x86_64"),
        ("opencv-contrib-python-headless", "4.10.0.84", "abi3", "macosx_12_0_x86_64"),
    ],
}

PROJECT = Path(__file__).resolve().parent.parent


def pypi_wheel(package: str, version: str, py_tag: str,
               platform_tag: str) -> tuple[str, str, int]:
    """The URL, SHA-256 and size of one published wheel, or raise."""
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as response:
        files = json.load(response)["urls"]

    matches = [f for f in files
               if f["filename"].endswith(".whl")
               and platform_tag in f["filename"]
               and py_tag in f["filename"]]
    if not matches:
        published = sorted({re.search(r"macosx_[0-9_]+_\w+", f["filename"]).group(0)
                            for f in files if "macosx" in f["filename"]})
        raise SystemExit(
            f"{package} {version} publishes no {platform_tag} wheel.\n"
            f"  it publishes: {', '.join(published) or 'none for macOS'}\n"
            f"  if the project changed its tags, WHEELS in this script needs "
            f"updating -- do not silently fall back to pip's choice, that is "
            f"the bug this script exists to prevent.")
    if len(matches) > 1:
        raise SystemExit(
            f"{py_tag}/{platform_tag} matches {len(matches)} wheels of "
            f"{package} {version}; the tags are not specific enough:\n  "
            + "\n  ".join(sorted(f["filename"] for f in matches)))

    best = matches[0]
    return best["url"], best["digests"]["sha256"], best["size"]


def fetch(url: str, expected_sha: str, into: Path) -> Path:
    target = into / url.rsplit("/", 1)[-1]
    with urllib.request.urlopen(url, timeout=600) as response:
        target.write_bytes(response.read())

    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != expected_sha:
        target.unlink(missing_ok=True)
        raise SystemExit(f"{target.name}: sha256 {actual}, "
                         f"but PyPI publishes {expected_sha}")
    return target


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def deployment_floor(path: Path) -> str | None:
    """The highest macOS version anything under `path` was built against."""
    floors: list[tuple[int, ...]] = []
    for binary in list(path.rglob("*.so")) + list(path.rglob("*.dylib")):
        try:
            listing = subprocess.run(["otool", "-l", str(binary)],
                                     capture_output=True, text=True,
                                     timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        # LC_BUILD_VERSION carries "minos", the older LC_VERSION_MIN_MACOSX
        # carries "version". A reader that knows only the first finds nothing
        # in exactly the old-SDK wheels this script goes out of its way to get.
        for match in re.finditer(
                r"(LC_BUILD_VERSION.*?minos ([0-9.]+))"
                r"|(LC_VERSION_MIN_MACOSX.*?version ([0-9.]+))",
                listing, re.S):
            found = match.group(2) or match.group(4)
            floors.append(tuple(int(n) for n in found.split(".")))
    if not floors:
        return None
    return ".".join(str(n) for n in max(floors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=sorted(WHEELS))
    parser.add_argument("--interpreter", required=True,
                        help="a relocatable python3.12 of that architecture; "
                             "NOT Homebrew's, which is built for this Mac")
    parser.add_argument("--venv", required=True)
    args = parser.parse_args()

    venv = Path(args.venv).resolve()
    wheels = venv.parent / f"{venv.name}-wheels"
    wheels.mkdir(parents=True, exist_ok=True)

    prefix = ["arch", "-x86_64"] if args.arch == "x86_64" else []

    print(f"==> virtualenv for {args.arch}")
    run(*prefix, args.interpreter, "-m", "venv", str(venv))
    python = str(venv / "bin" / "python")
    run(*prefix, python, "-m", "pip", "install", "--quiet", "--upgrade", "pip")

    print("==> the two wheels that set the deployment target")
    files = []
    for package, version, py_tag, tag in WHEELS[args.arch]:
        url, sha, size = pypi_wheel(package, version, py_tag, tag)
        print(f"    {package} {version} {py_tag}-{tag} ({size / 1e6:.0f} MB)")
        files.append(str(fetch(url, sha, wheels)))
        print("      digest matches PyPI")
    run(*prefix, python, "-m", "pip", "install", "--quiet", *files)

    print("==> everything else")
    run(*prefix, python, "-m", "pip", "install", "--quiet",
        "-r", str(PROJECT / "requirements.txt"),
        "-r", str(PROJECT / "requirements-desktop.txt"),
        "pyinstaller==6.22.3")

    # requirements.txt names the same versions, so a resolver that re-fetched
    # them would have replaced the chosen wheels with pip's own picks.
    print("==> confirming the chosen wheels survived")
    for package, version, _py_tag, _tag in WHEELS[args.arch]:
        module = {"numpy": "numpy",
                  "opencv-contrib-python-headless": "cv2"}[package]
        got = subprocess.run(
            [*prefix, python, "-c",
             f"import {module}; print({module}.__version__)"],
            capture_output=True, text=True, check=True).stdout.strip()
        if not version.startswith(got.rsplit(".", 1)[0].rsplit(".", 1)[0]):
            print(f"    WARNING: {package} is {got}, expected {version}")
        installed = next((p for p in (venv / "lib").rglob("site-packages")), None)
        print(f"    {package}: {got}")

    site = next((venv / "lib").rglob("site-packages"), None)
    floor = deployment_floor(site) if site else None
    print(f"\nReady: {venv}")
    print(f"  oldest macOS these wheels allow: {floor or 'unknown'}")
    print(f"  build with: PYTHON={python} "
          f"{'arch -x86_64 ' if prefix else ''}./scripts/build_macos_app.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
