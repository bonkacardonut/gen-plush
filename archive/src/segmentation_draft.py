"""DRAFT (v1) of segmentation, kept for reference. The current version is segmentation.py.

Difference from the current version: every drawn line is treated as a wall, so facial
features drawn as lines (mouths, closed eyes) disappear into their neighbors, and tiny
pieces always merge, even high-contrast ones like small black eyes.

Split a background-removed plushie image into regions (candidate felt pieces).

Pipeline:
  1. line mask   - find thin dark outlines (walls between pieces)
  2. color split - smooth, then group pixels into a few flat colors (k-means)
  3. regions     - connected blobs of one color that don't cross a line
  4. merge       - fold tiny blobs into neighbors, merge near-identical neighbors
  5. fill lines  - give each outline pixel to the nearest region
"""
import cv2
import numpy as np
import heapq

from scipy import ndimage
from skimage.morphology import skeletonize


def to_lab(rgb):
    """RGB uint8 -> Lab float32 with L in 0-100 and a/b centered on 0."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


# ---------- 1. line mask ----------

def line_mask(rgb, fg, dark_L=45, local_drop=18, max_stroke_frac=0.04, elongation=2.0):
    """Outline strokes inside the foreground. Returns bool HxW.

    Two tests:
      1. thin:  the pixel belongs to a dark stroke narrower than ~4% of the image
                (big dark fills like a navy hat or a brown star fail this)
      2. long:  the thin piece it belongs to is longer than it is wide
                (small round eyes fail this, so they stay as regions)
    """
    L = to_lab(rgb)[..., 0]

    # dark in absolute terms (black/brown outlines) ...
    dark = L < dark_L
    # ... or much darker than the area around it (colored outlines, e.g. dark blue)
    local = cv2.medianBlur(L.astype(np.uint8), 15).astype(np.float32)
    dark |= (local - L) > local_drop
    dark = (dark & fg).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # test 1: opening with a big round brush erases everything thinner than the brush
    d = max(3, int(max_stroke_frac * min(fg.shape)) | 1)
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, brush)
    thin = (dark & (1 - thick)).astype(np.uint8)

    # test 2: per thin piece, compare centerline length to average width
    n, comp = cv2.connectedComponents(thin, connectivity=8)
    skel = skeletonize(thin.astype(bool))
    area = np.bincount(comp.ravel(), minlength=n).astype(float)
    length = np.bincount(comp.ravel(), weights=skel.ravel(), minlength=n)
    width = area / np.maximum(length, 1)
    is_line = length >= elongation * width
    is_line[0] = False
    lines = is_line[comp]

    # where thick lines cross, the junction is a small round blob that survives the
    # opening. Small thick blobs touching a line are junctions, so they join the walls.
    n2, tcomp, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    near_line = cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    touches = np.bincount(tcomp[near_line], minlength=n2) > 0
    small = stats[:, cv2.CC_STAT_AREA] < 3 * d * d
    junction = touches & small
    junction[0] = False
    lines |= junction[tcomp]

    # the cutout's outer edge always counts as a wall
    edge = fg & ~cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return lines | edge


# ---------- 2. color split ----------

def quantize_colors(rgb, fg, k=10, seed=0):
    """Smooth, then k-means on Lab colors of foreground pixels. Returns (cluster id per pixel, Lab image)."""
    smooth = cv2.pyrMeanShiftFiltering(rgb, sp=7, sr=20)  # flattens shading, keeps edges
    lab = to_lab(smooth)
    pixels = lab[fg]
    k = min(k, len(np.unique(pixels.round(), axis=0)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(seed)
    _, ids, _ = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    cluster = np.full(fg.shape, -1, np.int32)
    cluster[fg] = ids.ravel()
    return cluster, to_lab(rgb)


# ---------- 3. regions ----------

def color_regions(cluster, walls):
    """Connected blobs of a single color cluster, never crossing a wall. 0 = not a region."""
    labels = np.zeros(cluster.shape, np.int32)
    next_id = 1
    for c in np.unique(cluster[cluster >= 0]):
        n, lab = cv2.connectedComponents(((cluster == c) & ~walls).astype(np.uint8), connectivity=4)
        m = lab > 0
        labels[m] = lab[m] + next_id - 1
        next_id += n - 1
    return labels


# ---------- 4. merge ----------

def absorb_specks(labels, walls, min_px):
    """Tiny blobs (anti-aliasing, jpeg noise) join whichever region is nearest. Walls stay 0."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    speck = np.isin(labels, ids[counts < min_px])
    labels = np.where(speck, 0, labels)
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[iy, ix]
    labels[speck & ~walls] = nearest[speck & ~walls]
    return labels


def region_stats(labels, lab):
    """Area and mean Lab color for every region id (index = id)."""
    n = labels.max() + 1
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    color_sum = np.stack([np.bincount(flat, lab[..., ch].ravel(), n) for ch in range(3)], 1)
    return area, color_sum


def adjacent_pairs(labels):
    """Unique (a, b) pairs of different regions that touch (4-neighbors), ignoring 0."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.sort(np.stack([a[m], b[m]], 1), axis=1))
    return np.unique(np.concatenate(pairs), axis=0)


def merge_regions(labels, lab, min_area, merge_tol=12.0):
    """Greedy merge: always merge the most similar touching pair first.
    Pairs merge if their mean colors are within merge_tol, or if either is smaller than min_area."""
    area, csum = region_stats(labels, lab)
    parent = np.arange(len(area))                       # union-find: who each region merged into
    nbrs = {i: set() for i in range(len(area))}
    for a, b in adjacent_pairs(labels):
        nbrs[a].add(b); nbrs[b].add(a)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def cost(a, b):
        d = np.linalg.norm(csum[a] / area[a] - csum[b] / area[b])
        if min(area[a], area[b]) < min_area:
            return d - 1000                              # tiny regions go first, to their closest color
        return d if d < merge_tol else None

    heap = []
    for a in nbrs:
        for b in nbrs[a]:
            if a < b and (c := cost(a, b)) is not None:
                heapq.heappush(heap, (c, a, b))

    while heap:
        c, a, b = heapq.heappop(heap)
        a, b = find(a), find(b)
        if a == b:
            continue
        if (now := cost(a, b)) is None or abs(now - c) > 1e-6:
            continue                                     # stale entry: colors changed since it was queued
        keep, drop = (a, b) if area[a] >= area[b] else (b, a)
        parent[drop] = keep
        area[keep] += area[drop]; csum[keep] += csum[drop]
        nbrs[keep] |= nbrs.pop(drop)
        nbrs[keep] = {find(x) for x in nbrs[keep]} - {keep}
        for x in nbrs[keep]:
            nbrs[x] = {find(y) for y in nbrs[x]}
            if (c2 := cost(keep, x)) is not None:
                heapq.heappush(heap, (c2, min(keep, x), max(keep, x)))

    roots = np.array([find(i) for i in range(len(parent))])
    merged = roots[labels]
    merged[labels == 0] = 0
    # tiny regions boxed in by lines on all sides had nobody to merge with: drop them
    a2 = np.bincount(merged.ravel())
    merged[(a2[merged] < min_area) & (merged > 0)] = 0
    return merged


# ---------- 5. fill lines ----------

def fill_lines(labels, fg):
    """Assign every unlabeled foreground pixel (outlines) to the nearest region."""
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    filled = labels[iy, ix]
    filled[~fg] = 0
    return filled


def relabel(labels):
    """Renumber regions 1..N, largest first."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    order = ids[np.argsort(-counts)]
    lut = np.zeros(labels.max() + 1, np.int32)
    lut[order] = np.arange(1, len(order) + 1)
    return lut[labels]


# ---------- all together ----------

def segment(rgba, k=10, merge_tol=15.0, min_area_frac=0.002):
    """rgba: output of remove_background. Returns (regions HxW int, lines HxW bool)."""
    rgb, fg = rgba[..., :3], rgba[..., 3] > 127
    lines = line_mask(rgb, fg)
    cluster, lab = quantize_colors(rgb, fg, k)
    regions = color_regions(cluster, lines)
    regions = absorb_specks(regions, lines, min_px=max(20, 0.0003 * fg.sum()))
    regions = merge_regions(regions, lab, min_area=min_area_frac * fg.sum(), merge_tol=merge_tol)
    regions = relabel(fill_lines(regions, fg))
    return regions, lines


def colorize(regions, seed=1):
    """Random color per region for viewing."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 255, (regions.max() + 1, 3), dtype=np.uint8)
    palette[0] = 255
    return palette[regions]
