"""
StudioLight README graphics — camera-body design system.

Reference: Sony RX1R III / Leica top plates.
  - near-black body, soft top light
  - small engraved uppercase type, wide tracking
  - hairlines and lens-style scales instead of boxes
  - monochrome only

All layout values are in GitHub display pixels (README column ~880px wide).
Images are rendered at SCALE x for retina sharpness.

Usage:  python tools/readme_graphics.py
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont

SCALE = 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
FONT_DIR = os.path.expanduser("~/Library/Fonts")

# --- palette -------------------------------------------------------------
BODY = (8, 8, 8)
ENGRAVE = (238, 238, 238)      # primary engraved white
SECOND = (140, 140, 140)       # secondary markings
FAINT = (88, 88, 88)           # minor ticks, footnotes
HAIR = (52, 52, 52)            # hairlines


def font(weight, size):
    files = {
        "light": "AkzidGrtskNext-Light.otf",
        "regular": "AkzidGrtskNext-Regular.otf",
        "medium": "AkzidGrtskNext-Med.otf",
    }
    return ImageFont.truetype(os.path.join(FONT_DIR, files[weight]), int(size * SCALE))


class Plate:
    """A black camera-body plate drawn in display-pixel coordinates."""

    def __init__(self, w, h, glow=None):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SCALE, h * SCALE), BODY)
        if glow:
            self._top_light(*glow)
        self.d = ImageDraw.Draw(self.img)

    def _top_light(self, cx, cy, radius, peak):
        """Soft studio light falling on the body, dithered to avoid banding."""
        W, H = self.img.size
        px = self.img.load()
        rnd = random.Random(7)
        cx, cy, r = cx * SCALE, cy * SCALE, radius * SCALE
        for y in range(H):
            for x in range(0, W):
                dx, dy = (x - cx) / r, (y - cy) / (r * 0.55)
                t = max(0.0, 1.0 - (dx * dx + dy * dy))
                v = BODY[0] + (peak - BODY[0]) * (t ** 2.2) + rnd.uniform(-0.6, 0.6)
                v = int(max(0, min(255, round(v))))
                px[x, y] = (v, v, v)

    # --- primitives --------------------------------------------------------
    def s(self, v):
        return v * SCALE

    def line(self, x1, y1, x2, y2, color=HAIR, width=1.0):
        self.d.line([(self.s(x1), self.s(y1)), (self.s(x2), self.s(y2))],
                    fill=color, width=max(1, int(round(width * SCALE))))

    def text_width(self, txt, f, tracking):
        return sum(f.getlength(c) for c in txt) / SCALE + tracking * (len(txt) - 1)

    def text(self, x, y, txt, f, color=ENGRAVE, tracking=0.0, anchor="l"):
        """Tracked text. y is the cap-height top. anchor: l | c | r."""
        w = self.text_width(txt, f, tracking)
        if anchor == "c":
            x -= w / 2
        elif anchor == "r":
            x -= w
        cx = self.s(x)
        for ch in txt:
            self.d.text((cx, self.s(y)), ch, font=f, fill=color, anchor="ls")
            cx += f.getlength(ch) + tracking * SCALE
        return w

    def arrow_right(self, x, y, color=SECOND, size=4):
        s = self.s
        self.d.polygon([(s(x), s(y)), (s(x - size * 1.6), s(y - size / 2)),
                        (s(x - size * 1.6), s(y + size / 2))], fill=color)

    def save(self, name):
        path = os.path.join(ASSETS, name)
        self.img.save(path, "PNG", optimize=True)
        print("saved", path, self.img.size)


# --- banner ----------------------------------------------------------------
def banner():
    W, H = 880, 300
    p = Plate(W, H, glow=(W / 2, 40, 520, 30))
    cx = W / 2

    # wordmark
    p.text(cx, 122, "STUDIOLIGHT", font("light", 46), ENGRAVE, tracking=9, anchor="c")

    # Kelvin scale, engraved like a lens aperture ring; daylight sits on the center index
    stops = ["2000", "2800", "3200", "4300", "5600", "6500", "7500", "8500", "10000"]
    index = stops.index("5600")
    sx0, sx1, ty = cx - 240, cx + 240, 172
    step = (sx1 - sx0) / (len(stops) - 1)
    for i in range(len(stops) - 1):
        for k in range(1, 4):
            x = sx0 + step * (i + k / 4)
            p.line(x, ty, x, ty + 3, FAINT, 0.8)
    f_scale = font("regular", 8.5)
    for i, label in enumerate(stops):
        x = sx0 + step * i
        major = i == index
        p.line(x, ty - (3 if major else 0), x, ty + 6, ENGRAVE if major else SECOND, 1.0)
        p.text(x, ty + 20, label, f_scale, ENGRAVE if major else SECOND, tracking=1.2, anchor="c")
    p.text(sx1 + 22, ty + 20, "K", f_scale, FAINT, tracking=0, anchor="l")

    # descriptor
    p.text(cx, 236, "3-POINT PHOTOMETRIC LIGHTING FOR FLUX", font("regular", 9.5),
           SECOND, tracking=3.2, anchor="c")

    # corner markings
    f_corner = font("regular", 7.5)
    p.text(32, H - 26, "METAMATERIAL", f_corner, FAINT, tracking=2.4)
    p.text(W - 32, H - 26, "V1.0.0", f_corner, FAINT, tracking=2.4, anchor="r")
    p.save("studiolight_banner.png")


# --- pipeline ----------------------------------------------------------------
def pipeline():
    W, H = 880, 204
    p = Plate(W, H)

    stations = [
        ("01", "PROMPT", ["Subject + wardrobe", "Albedo suffix"]),
        ("02", "DRAFT", ["Pass 1 · steps 0–10", "Locks geometry"]),
        ("03", "GEOMETRY", ["DSINE normals", "Metric depth"]),
        ("04", "LIGHT RIG", ["Key · Fill · Rim", "PCSS shadows"]),
        ("05", "COMPOSITE", ["Linear Rec.709", "Diffuse + specular"]),
        ("06", "DARKROOM", ["Shadow / highlight", "Tone sculpt"]),
        ("07", "REFINE", ["Pass 2 · steps 10–30", "Denoise 0.55"]),
    ]
    core = 3  # LIGHT RIG gets the index mark

    x0, x1 = 64, W - 64
    step = (x1 - x0) / (len(stations) - 1)
    xs = [x0 + step * i for i in range(len(stations))]
    base = 104

    # stage brackets (what runs where)
    f_group = font("regular", 7.5)
    groups = [(0, 1, "PASS 1"), (2, 5, "STUDIOLIGHT"), (6, 6, "PASS 2")]
    gy = 44
    for a, b, label in groups:
        ga, gb = xs[a] - step * 0.38, xs[b] + step * 0.38
        p.line(ga, gy, gb, gy, HAIR, 1)
        p.line(ga, gy, ga, gy + 4, HAIR, 1)
        p.line(gb, gy, gb, gy + 4, HAIR, 1)
        p.text((ga + gb) / 2, gy - 8, label, f_group, FAINT, tracking=2.4, anchor="c")

    # main scale line with minor ticks
    p.line(x0 - 24, base, x1 + 30, base, HAIR, 1)
    p.arrow_right(x1 + 34, base, SECOND, 4)
    for i in range(len(xs) - 1):
        for k in range(1, 5):
            x = xs[i] + step * k / 5
            p.line(x, base - 2, x, base + 2, FAINT, 0.8)

    f_num = font("regular", 8)
    f_name = font("medium", 10.5)
    f_spec = font("regular", 9)
    for i, (num, name, spec) in enumerate(stations):
        x = xs[i]
        is_core = i == core
        p.line(x, base - (14 if is_core else 7), x, base + 7,
               ENGRAVE if is_core else SECOND, 1.1 if is_core else 1.0)
        p.text(x, base - (20 if is_core else 14), num, f_num,
               ENGRAVE if is_core else FAINT, tracking=1.5, anchor="c")
        p.text(x, base + 30, name, f_name, ENGRAVE, tracking=2.2, anchor="c")
        for j, line in enumerate(spec):
            p.text(x, base + 52 + j * 16, line, f_spec, SECOND, tracking=0.4, anchor="c")

    # output marker
    p.save("studiolight_pipeline.png")


if __name__ == "__main__":
    banner()
    pipeline()
