"""Layered segmentation: pieces can sit on top of other pieces, like real felt appliqué.

Builds on segmentation.py (same line mask, colors and flat regions), then:
  1. objects   - dark shapes inside the body that don't touch the outline (eyes, noses,
                 mouths). Closed outlines (big cartoon eyes) are filled in to a complete shape.
  2. shapes    - areas fenced in by outlines ("compartments"). Neighboring compartments with
                 the same main color merge into one shape (4 clover leaves -> 1 clover,
                 ear + head -> one head). The lines between them are kept as stitch lines.
  3. details   - colors inside a shape that differ from its main color (inner ear, muzzle,
                 blush) become pieces layered on top.

Output: a list of pieces, each with its own full mask, bottom layer first.
"""
import cv2
import numpy as np
from scipy import ndimage

from src.segmentation import (to_lab, line_mask, quantize_colors, color_regions,
                              absorb_specks, merge_regions, fill_lines)


# ---------- helpers ----------

def outer_edge(fg, width=3):
    """Band of pixels along the cutout's edge."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    return fg & ~cv2.erode(fg.astype(np.uint8), k).astype(bool)


def dark_mask(rgb, fg, dark_L=45, local_drop=18):
    """Same 'dark pixel' test that line_mask uses."""
    L = to_lab(rgb)[..., 0]
    local = cv2.medianBlur(L.astype(np.uint8), 15).astype(np.float32)
    return ((L < dark_L) | ((local - L) > local_drop)) & fg


def mean_lab(lab, mask):
    return lab[mask].mean(0) if mask.any() else np.zeros(3)


def color_dist(a, b):
    return float(np.linalg.norm(a - b))


# ---------- 1. objects: eyes, noses, mouths ----------

def find_objects(rgb, fg, dark_L=45, close_gaps_px=3, max_shape_frac=0.15, min_px=15):
    """Dark shapes that float inside the body (not touching the outer outline).

    Each one is filled in, so a ring-shaped eye outline becomes a complete oval.
    close_gaps_px: how big a gap in an outline can be and still count as closed.
                   Higher = more aggressive about completing shapes.
    max_shape_frac: anything bigger than this fraction of the plushie isn't a 'feature'.
    """
    dark = dark_mask(rgb, fg, dark_L).astype(np.uint8)
    if close_gaps_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_gaps_px + 1,) * 2)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)
    dark = dark.astype(bool) & fg
    edge = outer_edge(fg)
    n, comp = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    objects = []
    for i in range(1, n):
        m = comp == i
        if m.sum() < min_px or (m & edge).any():
            continue
        filled = ndimage.binary_fill_holes(m)
        if filled.sum() > max_shape_frac * fg.sum():
            continue
        objects.append(dict(stroke=m, mask=filled, closed=filled.sum() > 1.3 * m.sum()))
    return objects


# ---------- 2. shapes: compartments fenced in by lines ----------

def find_compartments(fg, walls, min_px, thin_frac=0.012):
    """Areas between outlines. Tiny or thin ones (like the strip between an outline and the
    cutout edge) are absorbed; wall pixels go to the nearest area."""
    n, comp = cv2.connectedComponents((fg & ~walls).astype(np.uint8), connectivity=4)
    comp = absorb_specks(comp, walls, min_px)
    # thickness of each compartment = radius of the biggest circle that fits inside it
    dist = ndimage.distance_transform_edt(comp > 0)
    ids = np.unique(comp[comp > 0])
    radius = np.asarray(ndimage.maximum(dist, comp, ids)) if len(ids) else np.array([])
    thin = ids[radius < thin_frac * min(fg.shape)]
    comp[np.isin(comp, thin)] = 0
    return fill_lines(comp, fg)          # every foreground pixel now belongs to a compartment


def border_color(flat, flat_lab_means, comp_mask, walls, dark):
    """The main color of a compartment = the flat region that lines most of its border.
    (Details like an inner ear sit in the middle; the base color runs along the edge.)
    Outline pixels never count: anything hugging a line, or any mostly-dark region
    (thick bits of outline), is skipped unless the compartment has nothing else (a navy hat)."""
    near_wall = cv2.dilate(walls.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ring = comp_mask & ~cv2.erode(comp_mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    ids_in = np.unique(flat[comp_mask & (flat > 0)])
    darkish = {r for r in ids_in if dark[flat == r].mean() > 0.5}
    for candidates, skip_dark in ((ring & ~near_wall, True), (comp_mask & ~near_wall, True),
                                  (comp_mask, False)):
        ids = flat[candidates]
        ids = ids[ids > 0]
        if skip_dark:
            ids = ids[~np.isin(ids, list(darkish))]
        if len(ids):
            best = np.bincount(ids).argmax()
            return best, flat_lab_means[best]
    return None, None


def group_compartments(comps, comp_colors, group_tol):
    """Merge touching compartments whose main colors are within group_tol (union-find)."""
    ids = [i for i in comp_colors if comp_colors[i] is not None]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = set()
    for a, b in ((comps[:, :-1], comps[:, 1:]), (comps[:-1], comps[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs |= set(map(tuple, np.unique(np.sort(np.stack([a[m], b[m]], 1), 1), axis=0)))
    for a, b in sorted(pairs, key=lambda p: color_dist(comp_colors[p[0]], comp_colors[p[1]])
                       if p[0] in parent and p[1] in parent else 1e9):
        if a in parent and b in parent and color_dist(comp_colors[a], comp_colors[b]) < group_tol:
            parent[find(a)] = find(b)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------- all together ----------

def segment_layers(rgba, k=10, merge_tol=20.0, min_area_frac=0.002,
                   close_gaps_px=3, max_shape_frac=0.15, group_tol=10.0, detail_tol=15.0,
                   outline_reach_frac=0.008):
    """rgba: output of remove_background.

    Returns (pieces, stitch_lines):
      pieces: list of dicts, bottom layer first, each with
              id, mask (full shape, even where covered), color (RGB), layer,
              kind ("shape", "detail" or "feature"), parent (id of the piece it sits on)
      stitch_lines: bool mask of outlines that fall *inside* a merged shape
    """
    rgb, fg = rgba[..., :3], rgba[..., 3] > 127
    lab = to_lab(rgb)
    min_area = min_area_frac * fg.sum()
    speck_px = max(20, 0.0003 * fg.sum())

    # flat regions, same as segmentation.py (used to read colors and find details)
    walls = line_mask(rgb, fg)
    cluster, _ = quantize_colors(rgb, fg, k)
    flat = color_regions(cluster, walls)
    flat = absorb_specks(flat, walls, speck_px)
    flat = merge_regions(flat, lab, min_area=min_area, merge_tol=merge_tol)
    flat_means = np.zeros((flat.max() + 1, 3))
    for r in np.unique(flat[flat > 0]):
        flat_means[r] = mean_lab(lab, flat == r)

    dark = dark_mask(rgb, fg)

    # 1. objects (eyes, noses, mouths); their pixels stop acting as walls
    objects = find_objects(rgb, fg, close_gaps_px=close_gaps_px, max_shape_frac=max_shape_frac)
    obj_area = np.zeros_like(fg)
    for o in objects:
        obj_area |= o["mask"]
    shape_walls = walls & ~cv2.dilate(obj_area.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    # 2. compartments -> shapes
    comps = find_compartments(fg, shape_walls, max(speck_px, min_area / 2))
    comp_colors, comp_base = {}, {}
    for c in np.unique(comps[comps > 0]):
        base, color = border_color(flat, flat_means, comps == c, walls | obj_area, dark)
        comp_colors[c], comp_base[c] = color, base
    groups = group_compartments(comps, comp_colors, group_tol)

    pieces, stitch = [], np.zeros_like(fg)
    # a shape only reaches a few pixels into its outline. Taking the whole outline makes
    # shapes swallow the gaps between lobes (a clover turns into a blob).
    reach = max(2, int(outline_reach_frac * min(fg.shape)))
    reach_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
    join_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * reach + 1,) * 2)
    for gi, g in enumerate(sorted(groups, key=lambda g: -np.isin(comps, g).sum())):
        m = np.isin(comps, g)
        if gi > 0:   # the biggest shape is the body: it keeps its full outline
            core = (m & ~walls).astype(np.uint8)
            core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, join_k)    # rejoin parts split by inner lines
            core = cv2.morphologyEx(core, cv2.MORPH_OPEN, reach_k)    # drop thin slivers
            m = ndimage.binary_fill_holes(m & cv2.dilate(core, reach_k).astype(bool))
            if m.sum() < min_area:
                continue
        base_ids = {comp_base[c] for c in g}
        base_px = m & np.isin(flat, list(base_ids))
        pieces.append(dict(mask=m, lab=np.mean([comp_colors[c] for c in g], axis=0), layer=0,
                           kind="shape", parent=None, color_px=base_px if base_px.any() else m,
                           base_ids=base_ids))
        if len(g) > 1:   # outlines running through the middle of a merged shape = stitch lines
            inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            stitch |= walls & inner
    stitch &= ~obj_area

    # 3. details: flat regions inside a shape whose color differs from the shape's color
    near_wall = cv2.dilate(walls.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    for si, s in enumerate(list(pieces)):
        for r in np.unique(flat[s["mask"] & ~obj_area]):
            if r == 0 or r in s["base_ids"]:
                continue
            rm = (flat == r) & s["mask"] & ~obj_area
            if rm.sum() < min_area or color_dist(flat_means[r], s["lab"]) < detail_tol:
                continue
            # skip outline leftovers: regions that mostly hug a line
            if (rm & near_wall).sum() > 0.6 * rm.sum():
                continue
            pieces.append(dict(mask=ndimage.binary_fill_holes(rm), lab=flat_means[r], layer=1,
                               kind="detail", parent=si, color_px=rm))

    # 4. objects on top, with whatever is inside a closed object as its own details
    for o in objects:
        owner = max(range(len(pieces)), key=lambda i: (pieces[i]["mask"] & o["mask"]).sum())
        top = max([p["layer"] for p in pieces if p["mask"][o["mask"]].any()] + [0])
        oi = len(pieces)
        pieces.append(dict(mask=o["mask"], lab=mean_lab(lab, o["stroke"]), layer=top + 1,
                           kind="shape" if o["closed"] else "feature", parent=owner, color_px=o["stroke"]))
        if o["closed"]:
            inside = o["mask"] & ~o["stroke"]
            for r in np.unique(flat[inside]):
                rm = (flat == r) & inside
                if r == 0 or rm.sum() < 15:
                    continue
                if color_dist(flat_means[r], pieces[oi]["lab"]) < detail_tol:
                    continue
                pieces.append(dict(mask=rm, lab=flat_means[r], layer=top + 2, kind="detail", parent=oi,
                                   color_px=rm))

    for i, p in enumerate(pieces):   # real color = median of the pixels that define the piece
        p["id"] = i                     # 'parent' refers to these ids
        p["color"] = tuple(int(v) for v in np.median(rgb[p.pop("color_px")], axis=0))
        p.pop("base_ids", None)
        p.pop("lab")
    pieces.sort(key=lambda p: p["layer"])
    return pieces, stitch


# ---------- viewing ----------

def render_layers(pieces, shape, upto=None):
    """Paint pieces bottom to top in their real colors, each outlined."""
    out = np.full(shape + (3,), 255, np.uint8)
    for p in pieces:
        if upto is not None and p["layer"] > upto:
            continue
        out[p["mask"]] = p["color"]
        edge = p["mask"] & ~cv2.erode(p["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[edge] = (40, 40, 40)
    return out