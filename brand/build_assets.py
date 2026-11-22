#!/usr/bin/env python3
"""Render the liquine brand assets from brand/liquine-mark.svg into brand/webui/ (committed; `make webui` copies
them into the web UI). Needs rsvg-convert and ImageMagick (magick). Re-run after changing the mark.

usage: brand/build_assets.py [splash_name_source_dir]   (default build/webui: which apple-splash files to make)
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "webui")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "build", "webui")
mark = open(os.path.join(HERE, "liquine-mark.svg")).read()
paths = re.findall(r"<path[^>]*/>", mark)  # the mark's shapes, viewBox 0 0 600 600 (content centre ~ (300, 309))
GRAD = ('<linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#14b8a6"/><stop offset="1" stop-color="#2563eb"/></linearGradient>')


def svg(body, size=600, defs=""):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" fill="none" '
            f'xmlns="http://www.w3.org/2000/svg"><defs>{defs}</defs>{body}</svg>\n')


def shapes(fill):
    return "".join(re.sub(r'fill="[^"]*"', f'fill="{fill}"', p) for p in paths)


def placed(fill, box, scale):  # mark centred in a box x box canvas at the given scale
    tx, ty = box / 2 - 300 * scale, box / 2 - 309 * scale
    return f'<g transform="translate({tx:.2f} {ty:.2f}) scale({scale})">{shapes(fill)}</g>'


icon = svg(shapes("url(#g)"), defs=GRAD)                                   # favicon: gradient mark
app = svg(f'<rect width="512" height="512" rx="112" fill="url(#g)"/>' + placed("#ffffff", 512, 0.62), 512, GRAD)
maskable = svg(f'<rect width="512" height="512" fill="url(#g)"/>' + placed("#ffffff", 512, 0.5), 512, GRAD)

os.makedirs(OUT, exist_ok=True)
tmp = tempfile.mkdtemp()


def write(name, text):
    p = os.path.join(tmp, name)
    open(p, "w").write(text)
    return p


def raster(src, out, w, h=None, bg=None):
    cmd = ["rsvg-convert", "-w", str(w), "-h", str(h or w), src, "-o", out]
    if bg:
        cmd[1:1] = ["-b", bg]
    subprocess.run(cmd, check=True)


open(os.path.join(HERE, "liquine-icon.svg"), "w").write(icon)
open(os.path.join(HERE, "liquine-app-icon.svg"), "w").write(app)
open(os.path.join(OUT, "favicon.svg"), "w").write(icon)
open(os.path.join(OUT, "favicon-dark.svg"), "w").write(icon)
icon_p, app_p, mask_p = write("icon.svg", icon), write("app.svg", app), write("mask.svg", maskable)
for n in (16, 32, 48):
    raster(icon_p, os.path.join(tmp, f"ico{n}.png"), n)
ico_in = [os.path.join(tmp, f"ico{n}.png") for n in (16, 32, 48)]
subprocess.run(["magick", *ico_in, os.path.join(OUT, "favicon.ico")], check=True)
subprocess.run(["magick", *ico_in, os.path.join(OUT, "favicon-dark.ico")], check=True)
for n in (64, 192, 512):
    raster(app_p, os.path.join(OUT, f"pwa-{n}x{n}.png"), n)
raster(mask_p, os.path.join(OUT, "maskable-icon-512x512.png"), 512)
raster(app_p, os.path.join(OUT, "apple-touch-icon-180x180.png"), 180)

# splash screens: same names/sizes as the bundle's, mark centred, light (#111 on white) and dark (#fafafa on #111)
for fn in sorted(os.listdir(SRC)):
    m = re.match(r"apple-splash-(portrait|landscape)-(dark-)?(\d+)x(\d+)\.png$", fn)
    if not m:
        continue
    w, h = int(m.group(3)), int(m.group(4))
    dark = bool(m.group(2))
    side = int(min(w, h) * 0.22)
    mk = write("splash_mark.svg", svg(shapes("#fafafa" if dark else "#111111")))
    mpng = os.path.join(tmp, "splash_mark.png")
    raster(mk, mpng, side)
    subprocess.run(["magick", "-size", f"{w}x{h}", f"xc:{'#111111' if dark else '#ffffff'}", mpng, "-gravity", "center",
                    "-composite", "-strip", "-define", "png:compression-level=9",
                    os.path.join(OUT, fn)], check=True)
print("assets in", OUT, len(os.listdir(OUT)), "files")
