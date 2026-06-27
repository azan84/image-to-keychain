"""Step 8 — wording plate below the keychain.

Renders a line (or lines) of text into a rounded-rectangle "name plate"
that sits just below the image silhouette, and emits two pieces of
geometry in keychain-mm space:

    plate   — the backing strip (same thickness as the base so it welds to
              the keychain in the slicer)
    glyphs  — the letters, positioned/centred on the plate

The orchestrator turns these into two separate parts/bodies (``text_plate``
and ``text``), exactly like the base and tab, so the user assigns a
contrasting filament in Bambu Studio and welds the plate onto the keychain.

Letters are vectorised the same way as the artwork: render to a high-res
raster mask with Pillow, then trace with vtracer via
``mask_to_multipolygon``. That keeps the whole pipeline raster->trace->extrude
and means glyph holes (o, a, e, ...) come out correctly.

Font resolution (``text_font``):
    - a path to a .ttf/.otf file -> used directly
    - a family name (e.g. "arialbd") -> looked up in the system Fonts dirs
    - "disney" / "waltograph" -> tries the free Waltograph font if installed
    - empty -> Arial Bold, then Arial, then DejaVuSans as fallbacks
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box

from .svg_polygons import mask_to_multipolygon
from .util import get_logger


# High-res raster height (px) used for tracing — large enough that vtracer
# reproduces crisp glyph corners (incl. decorative fonts like Waltograph);
# the real size is set later in mm.
_RENDER_PX = 480
_PAD_PX = 48


@dataclass
class TextLabel:
    plate: MultiPolygon          # mm coords, Y-up — the backing strip
    glyphs: MultiPolygon         # mm coords, Y-up — letters, positioned on the plate
    plate_thickness: float       # mm (Z)
    text_thickness: float        # mm (Z, how far letters rise above the plate)
    recessed: bool
    text_color: tuple[int, int, int]
    plate_color: tuple[int, int, int]

    @property
    def is_empty(self) -> bool:
        return self.plate.is_empty or self.glyphs.is_empty


# ---- font resolution -------------------------------------------------------

def _font_dirs() -> list[Path]:
    dirs: list[Path] = []
    # Fonts bundled with this project (DejaVuSans-Bold default; plus any
    # personal font you drop in locally, e.g. waltographUI.ttf for "disney").
    dirs.append(Path(__file__).resolve().parent.parent / "fonts")
    win = os.environ.get("WINDIR", r"C:\Windows")
    dirs.append(Path(win) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    # macOS / Linux common dirs (harmless if absent). The truetype/dejavu dir
    # is where the `fonts-dejavu-core` apt package (see packages.txt) installs
    # DejaVuSans-Bold.ttf on the Hugging Face Spaces Linux container.
    dirs += [Path("/Library/Fonts"), Path("/System/Library/Fonts"),
             Path.home() / "Library" / "Fonts",
             Path("/usr/share/fonts/truetype/dejavu"),
             Path("/usr/share/fonts/truetype"),
             Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    return [d for d in dirs if d.exists()]


def _try_font(path: str | os.PathLike, size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return None


def resolve_font(spec: str | None, size: int = _RENDER_PX) -> ImageFont.FreeTypeFont:
    """Resolve ``text_font`` to a Pillow font, with sensible fallbacks."""
    candidates: list[str] = []
    spec = (spec or "").strip()

    if spec:
        p = Path(spec)
        if p.exists():
            f = _try_font(p, size)
            if f is not None:
                return f
        low = spec.lower()
        # Disney-style alias -> Waltograph (free), if the user installed it
        if "disney" in low or "walt" in low:
            for fn in ("waltographUI.ttf", "waltograph42.ttf", "Waltograph.otf",
                       "waltograph.ttf"):
                candidates.append(fn)
        # Treat the spec as a family/filename to look up in the font dirs
        for ext in ("", ".ttf", ".otf", ".TTF"):
            candidates.append(f"{spec}{ext}")

    # Defaults — bold first (prints/reads better small), then regular
    candidates += ["arialbd.ttf", "Arial Bold.ttf", "arial.ttf", "Arial.ttf",
                   "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]

    dirs = _font_dirs()
    for name in candidates:
        # Absolute or cwd-relative
        f = _try_font(name, size)
        if f is not None:
            return f
        for d in dirs:
            f = _try_font(d / name, size)
            if f is not None:
                return f

    raise FileNotFoundError(
        "Could not load any font for the text label. Set 'text_font' to a "
        "valid .ttf path (e.g. C:/Windows/Fonts/arialbd.ttf)."
    )


# ---- rasterise text --------------------------------------------------------

_LINE_SPACING_PX = 8   # extra px between lines at render resolution


def _render_native(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Pillow's own (multi-)line layout — crisp for every font."""
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 0))
    bbox = probe.multiline_textbbox((0, 0), text, font=font,
                                    align="center", spacing=_LINE_SPACING_PX)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    W = max(1, int(math.ceil(w)) + 2 * _PAD_PX)
    H = max(1, int(math.ceil(h)) + 2 * _PAD_PX)
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).multiline_text(
        (_PAD_PX - bbox[0], _PAD_PX - bbox[1]), text, fill=255, font=font,
        align="center", spacing=_LINE_SPACING_PX)
    return np.asarray(img) > 127


def _render_text_mask(text: str, font: ImageFont.FreeTypeFont,
                      tracking_frac: float) -> np.ndarray:
    """Render ``text`` (may contain newlines) to a bool mask, True = ink.

    With no letter tracking we let Pillow lay out the whole string in one
    pass, which respects each glyph's side bearings and kerning — essential
    for decorative fonts (e.g. Waltograph) whose ink is wider than its
    advance width. Only when tracking is requested do we fall back to a
    per-character layout that can space glyphs apart.
    """
    if tracking_frac <= 0:
        return _render_native(text, font)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    tracking_px = tracking_frac * font.size
    lines = text.split("\n")

    # Width of each line (advance widths + tracking between chars).
    line_widths: list[float] = []
    for ln in lines:
        w = sum(font.getlength(ch) for ch in ln)
        if len(ln) > 1:
            w += tracking_px * (len(ln) - 1)
        line_widths.append(w)

    block_w = max(line_widths) if line_widths else 1.0
    block_h = line_h * len(lines)

    W = max(1, int(math.ceil(block_w)) + 2 * _PAD_PX)
    H = max(1, int(math.ceil(block_h)) + 2 * _PAD_PX)
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    y = _PAD_PX
    for ln, lw in zip(lines, line_widths):
        x = _PAD_PX + (block_w - lw) / 2.0   # centre each line in the block
        for ch in ln:
            draw.text((x, y), ch, fill=255, font=font)
            x += font.getlength(ch) + tracking_px
        y += line_h

    return np.asarray(img) > 127


# ---- public API ------------------------------------------------------------

def build_text_label(bounds_mm: tuple[tuple[float, float], tuple[float, float]],
                     cfg: dict[str, Any], inter_dir: Path,
                     bottom_y: float | None = None,
                     verbose: bool = True) -> TextLabel | None:
    """Build the wording plate + letters in keychain-mm space.

    ``bounds_mm`` is the silhouette bbox after the px->mm transform
    (((min_x, min_y), (max_x, max_y))). The plate is centred under the
    subject and placed ``text_margin_mm`` below ``bottom_y`` (defaults to the
    silhouette's bottom edge). Returns ``None`` when disabled or empty.
    """
    log = get_logger(verbose=verbose)
    if not bool(cfg.get("text_enabled", False)):
        return None
    text = str(cfg.get("text_string", "") or "").strip("\n")
    if not text.strip():
        log.info("Step 8: text_enabled but text_string is empty — skipping")
        return None

    (x0, y0), (x1, y1) = bounds_mm
    if bottom_y is None:
        bottom_y = y0

    height_mm = float(cfg.get("text_height_mm", 8.0))
    text_t = float(cfg.get("text_thickness_mm", 1.0))
    plate_t = cfg.get("text_plate_thickness_mm", None)
    plate_t = float(plate_t) if plate_t else float(cfg.get("base_thickness", 2.0))
    margin = float(cfg.get("text_margin_mm", 1.5))
    pad = float(cfg.get("text_padding_mm", 3.0))
    corner_r = float(cfg.get("text_corner_radius_mm", 2.0))
    tracking = float(cfg.get("text_letter_spacing", 0.0))
    recessed = bool(cfg.get("text_recessed", False))
    text_color = tuple(int(c) for c in cfg.get("text_color", [25, 25, 25]))[:3]
    plate_color = tuple(int(c) for c in cfg.get("text_plate_color", [230, 230, 230]))[:3]

    # 1. Rasterise + trace to polygons (pixel coords, Y already flipped up).
    font = resolve_font(cfg.get("text_font"), size=_RENDER_PX)
    mask = _render_text_mask(text, font, tracking)
    glyphs_px = mask_to_multipolygon(mask, inter_dir, "text", flip_y=True, min_area=2.0)
    if glyphs_px.is_empty:
        log.warning("Step 8: text traced to no polygons — skipping")
        return None

    # 2. Scale so the wording's ink height == text_height_mm.
    gx0, gy0, gx1, gy1 = glyphs_px.bounds
    glyph_h = max(1e-6, gy1 - gy0)
    scale = height_mm / glyph_h
    glyphs = affinity.scale(glyphs_px, xfact=scale, yfact=scale, origin=(0, 0))
    gx0, gy0, gx1, gy1 = glyphs.bounds
    content_w = gx1 - gx0
    content_h = gy1 - gy0

    # 3. Plate sized to the wording + padding, centred under the subject,
    #    placed `margin` below the keychain's bottom edge.
    plate_w = content_w + 2 * pad
    plate_h = content_h + 2 * pad
    center_x = 0.5 * (x0 + x1)
    plate_x0 = center_x - plate_w / 2.0
    plate_x1 = center_x + plate_w / 2.0
    plate_y1 = bottom_y - margin            # plate top edge
    plate_y0 = plate_y1 - plate_h           # plate bottom edge

    r = max(0.0, min(corner_r, 0.5 * min(plate_w, plate_h) - 1e-6))
    if r <= 1e-6:
        plate = box(plate_x0, plate_y0, plate_x1, plate_y1)
    else:
        plate = box(plate_x0, plate_y0, plate_x1, plate_y1).buffer(
            -r, quad_segs=16).buffer(r, quad_segs=16)

    # 4. Centre the glyphs on the plate.
    dx = (plate_x0 + pad) - gx0
    dy = (plate_y0 + pad) - gy0
    glyphs = affinity.translate(glyphs, xoff=dx, yoff=dy)

    def _mp(g) -> MultiPolygon:
        if g.is_empty:
            return MultiPolygon()
        if g.geom_type == "Polygon":
            return MultiPolygon([g])
        if g.geom_type == "MultiPolygon":
            return g
        polys = [p for p in getattr(g, "geoms", [])
                 if p.geom_type == "Polygon" and not p.is_empty]
        return MultiPolygon(polys)

    log.info("Step 8: text plate \"%s\"  %.1f×%.1f mm at y=%.1f..%.1f (font=%s)",
             text.replace("\n", " "), plate_w, plate_h, plate_y0, plate_y1,
             Path(getattr(font, "path", "?")).name)

    return TextLabel(plate=_mp(plate), glyphs=_mp(glyphs),
                     plate_thickness=plate_t, text_thickness=text_t,
                     recessed=recessed, text_color=text_color,
                     plate_color=plate_color)
