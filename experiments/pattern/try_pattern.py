"""Try out a felt sewing pattern for any image, using the current layered segmentation.

Usage (from the project folder, with the venv active):
    python scripts/try_pattern.py path/to/image.png
    python scripts/try_pattern.py path/to/image.png --size 25 --out my_pattern.pdf

Makes a PDF (letter paper) with:
  - a cover page: preview of the finished front, piece list, and assembly order by layer
  - full-size pattern pages: body front + back (with seam allowance) and every felt piece
Saves the PDF and a PNG preview of the segmentation in this script's folder.

Assumes a flat 2-sided plush: front and back silhouettes are sewn together and stuffed,
and every other piece is felt glued/stitched onto the front, bottom layer first.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                               # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists() and ROOT != ROOT.parent:   # walk up to the project folder
    ROOT = ROOT.parent
sys.path.append(str(ROOT))
from src.background_removal import load_rgb, remove_background
from src.segmentation_layers import segment_layers, render_layers

PAGE_W, PAGE_H = 215.9, 279.4                       # US letter, mm
MARGIN, GAP = 10, 6                                 # mm


# ---------- smoothing: pixels -> clean curves ----------

def smooth_mask(mask, blur_px=2.5):
    """Blur the mask and re-threshold it. Rounds off the pixel staircase and tiny bumps
    before tracing, the same way you'd sand an edge."""
    if blur_px <= 0:
        return mask
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), blur_px)
    return soft > 0.5


def chaikin(points, rounds=3):
    """Chaikin corner cutting: replace every corner with two points 1/4 and 3/4 along each edge.
    Repeating it turns a jagged polygon into a smooth curve."""
    pts = np.asarray(points, float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    for _ in range(rounds):
        nxt = np.roll(pts, -1, axis=0)
        pts = np.stack([0.75 * pts + 0.25 * nxt, 0.25 * pts + 0.75 * nxt], 1).reshape(-1, 2)
    return pts


def smooth_ring(ring_pts, simplify_px, rounds):
    """Simplify a traced outline (drop the staircase points), then round it with Chaikin."""
    line = Polygon(ring_pts).exterior.simplify(simplify_px)
    pts = np.asarray(line.coords)
    return chaikin(pts, rounds) if len(pts) >= 4 else pts


def mask_to_shape(mask, blur_px=2.5, simplify_px=1.5, rounds=3, min_px=30):
    """Binary mask -> smooth shapely (Multi)Polygon in pixel units, holes included."""
    mask = smooth_mask(mask, blur_px)
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return None
    polys = []
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1 or len(c) < 4 or cv2.contourArea(c) < min_px:
            continue                                # holes are attached to their parent below
        outer = smooth_ring(c[:, 0, :], simplify_px, rounds)
        holes, child = [], hier[0][i][2]
        while child != -1:
            h = contours[child]
            if len(h) >= 4 and cv2.contourArea(h) >= min_px:
                holes.append(smooth_ring(h[:, 0, :], simplify_px, rounds))
            child = hier[0][child][0]
        if len(outer) >= 4:
            polys.append(Polygon(outer, [h for h in holes if len(h) >= 4]).buffer(0))
    return unary_union(polys) if polys else None


# ---------- felt color names ----------

FELT_COLORS = {
    "white": (245, 245, 240), "cream": (240, 228, 200), "light pink": (245, 190, 200),
    "pink": (235, 130, 160), "red": (210, 40, 45), "orange": (235, 130, 40),
    "yellow": (245, 215, 70), "light yellow": (250, 240, 160), "lime": (170, 200, 60),
    "green": (70, 160, 70), "teal": (70, 180, 175), "light blue": (170, 210, 235),
    "blue": (60, 110, 180), "navy": (25, 50, 100), "purple": (130, 90, 170),
    "tan": (210, 170, 120), "caramel": (190, 120, 40), "brown": (110, 70, 45),
    "dark brown": (60, 40, 30), "gray": (160, 160, 160), "black": (25, 25, 25),
}
_NAMES = list(FELT_COLORS)
_LAB = cv2.cvtColor(np.array([list(FELT_COLORS.values())], np.uint8), cv2.COLOR_RGB2LAB)[0].astype(float)


def felt_name(rgb):
    lab = cv2.cvtColor(np.array([[rgb]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return _NAMES[int(np.argmin(np.linalg.norm(_LAB - lab, axis=1)))]


# ---------- layered pieces -> pattern pieces ----------

def build_pieces(rgba, layers, size_mm, seam_mm, min_width_mm, blur_px, detail_blur_px=0.8):
    fg = rgba[..., 3] > 127
    sil = mask_to_shape(fg, blur_px=blur_px)
    sil = max(sil.geoms, key=lambda g: g.area) if isinstance(sil, MultiPolygon) else sil
    sil = Polygon(sil.exterior)                     # body outline, no holes
    minx, miny, maxx, maxy = sil.bounds
    s = size_mm / max(maxx - minx, maxy - miny)     # mm per pixel

    def to_mm(g):
        return affinity.scale(affinity.translate(g, -minx, -miny), s, s, origin=(0, 0))

    base = layers[0]                                # biggest bottom-layer shape = body color
    front = to_mm(sil)
    pieces = [
        dict(name="Body front", shape=front, color=base["color"], cut="cut 1", seam=seam_mm, layer=-1),
        dict(name="Body back", shape=affinity.scale(front, -1, 1, origin="center"),
             color=base["color"], cut="cut 1 (mirrored)", seam=seam_mm, layer=-1),
    ]
    for p in layers[1:]:
        g = mask_to_shape(p["mask"], blur_px=detail_blur_px, simplify_px=1.0, rounds=2)  # lighter on small pieces
        if g is None or g.is_empty:
            continue
        g = to_mm(g)
        avg_width = 2 * g.area / max(g.length, 1e-6)
        if avg_width < min_width_mm:                # too thin to cut: fatten it
            g = g.buffer((min_width_mm - avg_width) / 2, join_style="round")
        if g.area < 4:                              # under ~2x2 mm: skip
            continue
        kind = {"shape": "piece", "detail": "detail", "feature": "feature"}[p["kind"]]
        pieces.append(dict(name=kind, shape=g, color=p["color"], cut="cut 1", seam=0, layer=p["layer"]))

    for i, p in enumerate(pieces, 1):
        p["num"] = i
        p["felt"] = felt_name(p["color"])
        if p["layer"] >= 0:
            p["name"] = f"{p['felt'].capitalize()} {p['name']}"
        p["cut_shape"] = p["shape"].buffer(p["seam"], join_style="round") if p["seam"] else p["shape"]
    return pieces, front


# ---------- page layout ----------

def pack(pieces):
    """Shelf packing: tallest first, left to right, new row when full, new page when out of room."""
    usable_w = PAGE_W - 2 * MARGIN
    order = sorted(pieces, key=lambda p: -(p["cut_shape"].bounds[3] - p["cut_shape"].bounds[1]))
    pages, page, x, y, row_h = [], [], MARGIN, MARGIN, 0
    for p in order:
        g = p["cut_shape"]
        w, h = g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1]
        p["rot"] = 0
        if w > usable_w and h <= usable_w:          # turn sideways if too wide
            p["rot"] = 90
            g = affinity.rotate(g, 90, origin="centroid")
            w, h = h, w
        if x + w > PAGE_W - MARGIN:
            x, y, row_h = MARGIN, y + row_h + GAP, 0
        if y + h + 8 > PAGE_H - MARGIN - 6 and page:
            pages.append(page)
            page, x, y, row_h = [], MARGIN, MARGIN, 0
        p["dx"], p["dy"] = x - g.bounds[0], y - g.bounds[1]
        page.append(p)
        x += w + GAP
        row_h = max(row_h, h + 8)
    pages.append(page)
    return pages


def placed(p, g):
    if p["rot"]:
        g = affinity.rotate(g, 90, origin=p["cut_shape"].centroid)
    return affinity.translate(g, p["dx"], p["dy"])


# ---------- drawing ----------

def patch(g, **kw):
    verts, codes = [], []
    for poly in getattr(g, "geoms", [g]):
        for ring in [poly.exterior, *poly.interiors]:
            pts = np.asarray(ring.coords)
            verts += pts.tolist()
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), **kw)


def new_page():
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W); ax.set_ylim(PAGE_H, 0); ax.axis("off")
    return fig, ax


def hexc(c):
    return "#%02x%02x%02x" % tuple(c)


def draw_cover(title, pieces, front):
    fig, ax = new_page()
    ax.text(MARGIN, 18, title, fontsize=18, weight="bold")
    ax.text(MARGIN, 25, "Flat 2-sided plush  ·  print pattern pages at 100% / Actual size", fontsize=8, color="0.35")
    pv = 100 / max(front.bounds[2], front.bounds[3])  # preview scale
    for p in sorted(pieces, key=lambda p: p["layer"]):
        if p["num"] == 2:
            continue
        g = affinity.translate(affinity.scale(p["shape"], pv, pv, origin=(0, 0)), MARGIN, 35)
        ax.add_patch(patch(g, facecolor=hexc(p["color"]), edgecolor="0.25", lw=0.5))
    for p in pieces:                                # numbers on top of everything
        if p["layer"] < 0:
            continue
        g = affinity.translate(affinity.scale(p["shape"], pv, pv, origin=(0, 0)), MARGIN, 35)
        c = g.representative_point()
        ax.text(c.x, c.y, str(p["num"]), fontsize=4.5, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none", alpha=0.85))

    tx, ty = 122, 40
    ax.text(tx, ty - 6, "Pieces (in assembly order)", fontsize=10, weight="bold")
    last_layer = None
    for p in sorted(pieces, key=lambda p: (p["layer"], p["num"])):
        if ty > PAGE_H - 25:
            ax.text(tx, ty, "…more on the pattern pages", fontsize=6.5); break
        if p["layer"] != last_layer:
            label = "Body" if p["layer"] < 0 else f"Layer {p['layer']}"
            ax.text(tx, ty, label, fontsize=7, weight="bold", color="0.3"); ty += 4.5
            last_layer = p["layer"]
        ax.add_patch(Rectangle((tx, ty - 2.5), 4, 4, facecolor=hexc(p["color"]), edgecolor="0.3", lw=0.4))
        ax.text(tx + 6, ty, f"{p['num']:>2}. {p['name']}  ·  {p['felt']} felt  ·  {p['cut']}", fontsize=6.3, va="center")
        ty += 5

    steps = ["1. Print the pattern pages at 100% and check the test squares (1 in / 2 cm).",
             "2. Cut each piece from the listed felt color. Body: cut on the solid line.",
             "3. Attach pieces to the body front layer by layer (lowest layer first).",
             "4. Put front and back together, decorated side out, and stitch on the dashed line,",
             "   leaving a 4 cm gap. Stuff firmly, then close the gap."]
    for i, s in enumerate(steps):
        ax.text(MARGIN, 160 + i * 6, s, fontsize=7.3)
    return fig


def draw_pattern_page(page, page_no, n_pages, title):
    fig, ax = new_page()
    for p in page:
        cut = placed(p, p["cut_shape"])
        ax.add_patch(patch(cut, facecolor=hexc(p["color"]) + "40", edgecolor="black", lw=0.7))
        if p["seam"]:
            ax.add_patch(patch(placed(p, p["shape"]), facecolor="none", edgecolor="0.3", lw=0.5, ls=(0, (3, 2))))
        b = cut.bounds
        ax.text(b[0], b[3] + 4, f"#{p['num']} {p['name']} · {p['felt']} · {p['cut']}", fontsize=5.5)
    if page_no == 1:                                # print-scale check
        ax.add_patch(Rectangle((PAGE_W - MARGIN - 25.4, PAGE_H - MARGIN - 31.4), 25.4, 25.4, fill=False, lw=0.8))
        ax.text(PAGE_W - MARGIN - 12.7, PAGE_H - MARGIN - 18.7, "1 in", fontsize=6, ha="center", va="center")
        ax.add_patch(Rectangle((PAGE_W - MARGIN - 55, PAGE_H - MARGIN - 26), 20, 20, fill=False, lw=0.8))
        ax.text(PAGE_W - MARGIN - 45, PAGE_H - MARGIN - 16, "2 cm", fontsize=6, ha="center", va="center")
    ax.text(MARGIN, PAGE_H - 5, f"{title} · page {page_no}/{n_pages} · solid = cut, dashed = sew",
            fontsize=6, color="0.4")
    return fig


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Make a test felt pattern PDF from any image.")
    ap.add_argument("image", help="path to the image (jpg/png/webp)")
    ap.add_argument("--size", type=float, default=18, help="longest side of the finished plush, in cm (default 18)")
    ap.add_argument("--seam", type=float, default=6, help="seam allowance on body pieces, in mm (default 6)")
    ap.add_argument("--smooth", type=float, default=2.5, help="edge smoothing on the body, 0 = none (default 2.5)")
    ap.add_argument("--detail-smooth", type=float, default=0.8,
                    help="edge smoothing on the felt pieces, 0 = none (default 0.8)")
    ap.add_argument("--out", help="output PDF path (default: <image name>_pattern.pdf in this script's folder)")
    args = ap.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    out_dir = Path(__file__).resolve().parent              # save next to this script
    out = Path(args.out) if args.out else out_dir / (img_path.stem + "_pattern.pdf")
    title = img_path.stem

    print(f"1/4 removing background from {img_path.name}")
    rgba, info = remove_background(load_rgb(img_path, 1024))
    if not info["confident"]:
        print("    note: busy background, the cutout may be rough")

    print("2/4 segmenting into layers")
    layers, _ = segment_layers(rgba)

    print("3/4 building pattern pieces")
    pieces, front = build_pieces(rgba, layers, args.size * 10, args.seam, min_width_mm=3, blur_px=args.smooth,
                                 detail_blur_px=args.detail_smooth)
    pages = pack(pieces)

    print("4/4 drawing PDF")
    figs = [draw_cover(title, pieces, front)]
    figs += [draw_pattern_page(pg, i, len(pages), title) for i, pg in enumerate(pages, 1)]
    with PdfPages(out) as pdf:
        for f in figs:
            pdf.savefig(f)
            plt.close(f)

    preview = out.with_name(out.stem + "_layers.png")   # segmentation preview, for debugging
    cv2.imwrite(str(preview), cv2.cvtColor(np.hstack([rgba[..., :3], render_layers(layers, rgba.shape[:2])]),
                                           cv2.COLOR_RGB2BGR))
    print(f"done: {len(pieces)} pieces on {len(pages)} pattern pages")
    print(f"  pattern: {out}")
    print(f"  preview: {preview}")


if __name__ == "__main__":
    main()