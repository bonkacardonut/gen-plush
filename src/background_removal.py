"""Background removal for GenPlush.

Main method (fast, no AI model):
    Clipart usually sits on a flat background and has outlines, so we
    flood-fill the background in from the image edges. White areas *inside*
    an outline (faces, bellies) are protected because they don't touch the edge.
    ~50 ms per image on CPU.

Fallback (optional):
    If the border isn't flat (photo, busy background), the result is flagged
    as low-confidence. If rembg is installed, we retry with the BiRefNet model.

Usage from a notebook:
    import sys; sys.path.append("..")           # if running from notebooks/
    from src.background_removal import remove_background, load_rgb
    rgba, info = remove_background(load_rgb("data/test_images/2d/easy/1e.jpg"))

Batch from the terminal (run in the project folder):
    python -m src.background_removal
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------- loading ----------

def load_rgb(path, max_side=1024):
    """Load any image as an RGB uint8 array, shrunk so the longest side <= max_side."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        # flatten transparency onto white so the border color is predictable
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))  # never upscales
    return np.array(img)


# ---------- main method: edge flood fill ----------

def remove_bg_clipart(rgb, tol=18.0, close_px=3, min_area_frac=0.0005, feather=1):
    """rgb: HxWx3 uint8. Returns (HxWx4 RGBA uint8, confident: bool)."""
    h, w = rgb.shape[:2]

    # 1. Lab color space, so "distance" matches how different colors look
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128

    # 2. background color = median of a thin border strip
    b = max(2, min(h, w) // 100)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg_color = np.median(border, axis=0)
    border_spread = np.median(np.linalg.norm(border - bg_color, axis=1))

    # 3. pixels that don't look like the background
    dist = np.linalg.norm(lab - bg_color, axis=2)
    fg_candidate = (dist > tol).astype(np.uint8)

    # seal small gaps in outlines so the fill can't leak inside
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        fg_candidate = cv2.morphologyEx(fg_candidate, cv2.MORPH_CLOSE, k)

    # 4. background = bg-colored regions that touch the image edge
    bg_like = 1 - fg_candidate
    _, labels = cv2.connectedComponents(bg_like, connectivity=4)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    mask = (~np.isin(labels, edge_labels)).astype(np.uint8)

    # 5. drop tiny specks (watermarks, jpeg noise)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_frac * h * w
    mask = keep[labels].astype(np.uint8) * 255

    # 6. soften the edge slightly (feather=0 gives a hard binary mask)
    if feather:
        mask = cv2.GaussianBlur(mask, (2 * feather + 1,) * 2, 0)

    confident = bool(border_spread < 6)  # flat border = clipart-style image
    return np.dstack([rgb, mask]), confident


# ---------- fallback: AI model (optional) ----------

_ai_session = None


def remove_bg_ai(rgb, model="birefnet-general"):
    """Fallback for busy backgrounds. Needs `pip install "rembg[cpu]"`."""
    global _ai_session
    from rembg import new_session, remove  # imported here so rembg stays optional
    if _ai_session is None:
        _ai_session = new_session(model)  # downloads the model once (~900 MB)
    return np.array(remove(Image.fromarray(rgb), session=_ai_session).convert("RGBA"))


# ---------- what the rest of the pipeline should call ----------

def remove_background(rgb, use_ai_fallback=True, **kwargs):
    """Returns (rgba, info). info["method"] says which method produced the result."""
    rgba, confident = remove_bg_clipart(rgb, **kwargs)
    if confident or not use_ai_fallback:
        return rgba, {"method": "clipart", "confident": confident}
    try:
        return remove_bg_ai(rgb), {"method": "ai", "confident": True}
    except ImportError:
        return rgba, {"method": "clipart", "confident": False}


def process_folder(src_root, out_root, max_side=1024, use_ai_fallback=True):
    """Run remove_background on every image under src_root, mirroring folders into out_root."""
    src_root, out_root = Path(src_root), Path(out_root)
    for path in sorted(src_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        out_path = out_root / path.relative_to(src_root).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgba, info = remove_background(load_rgb(path, max_side), use_ai_fallback)
        Image.fromarray(rgba).save(out_path)
        flag = "" if info["confident"] else "  <- low confidence, check this one"
        print(f"{path.relative_to(src_root)}: {info['method']}{flag}")


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent
    process_folder(project / "data/test_images/2d", project / "data/2d-cleaned/2d_nobg_v2")
