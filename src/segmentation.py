"""
Blood-smear cell segmentation (NumPy + SciPy only, no torch).

The classifier was trained on the NIH dataset, where every image is ONE
red blood cell cropped tightly and placed on a black background. A full
microscope field (many cells on a light background) looks nothing like
that, so feeding it to the model directly produces meaningless output.

This module turns a field image into NIH-style single-cell crops:
  1. separate cells from the bright background (Otsu threshold)
  2. clean the mask and split touching cells (distance-transform seeds)
  3. crop each cell tightly and black out everything that isn't the cell
"""

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

MAX_SIDE = 1400          # downscale huge photos before segmenting
BLACK_LEVEL = 20         # pixel max-channel value treated as "black"


def is_single_cell_crop(image: Image.Image) -> bool:
    """True when the image already looks like an NIH crop: one cell on black."""
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    if max(h, w) > 400:
        return False
    black = arr.max(axis=2) < BLACK_LEVEL
    border = np.concatenate([black[0], black[-1], black[:, 0], black[:, -1]])
    return black.mean() > 0.05 and border.mean() > 0.3


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[np.argmax(between)])


def _foreground_mask(arr: np.ndarray) -> np.ndarray:
    """Cells are darker and more saturated than the smear background."""
    rgb = arr.astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    saturation = (mx - mn) / np.maximum(mx, 1)
    darkness = 1 - rgb[..., 1] / 255.0          # green is absorbed by stain

    def robust(x):
        # percentile scaling so a few very dark pixels (arrows, stain
        # blobs, text) can't dominate the threshold
        lo, hi = np.percentile(x, [1, 99])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    score = 0.5 * robust(saturation) + 0.5 * robust(darkness)
    score = ndi.gaussian_filter(score, 1.5)

    mask = score > _otsu(score.ravel())
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)   # pale cell centres
    return mask


def _typical_area(areas: np.ndarray, image_pixels: int) -> float:
    """Median size of real objects (ignores specks, robust to big clumps)."""
    min_area = max(30.0, 2e-4 * image_pixels)
    real = areas[areas >= min_area]
    if len(real) == 0:
        return float(areas.max())
    return float(np.median(real))


def _purple_fraction(crop: np.ndarray, cell_mask: np.ndarray) -> float:
    """Share of the cell covered by dark purple (nuclear / parasite) stain."""
    px = crop[cell_mask].astype(np.int16)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    purple = (b - g > 35) & (r - g > 15) & (g < 140)
    return float(purple.mean()) if len(px) else 0.0


def _split_touching(mask: np.ndarray) -> np.ndarray:
    """Label cells, splitting clumps with distance-transform seeds."""
    labels, n = ndi.label(mask)
    if n == 0:
        return labels

    areas = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    radius = np.sqrt(_typical_area(areas, mask.size) / np.pi)

    dist = ndi.distance_transform_edt(mask)
    seeds = dist > 0.55 * radius
    seed_labels, n_seeds = ndi.label(seeds)
    if n_seeds == 0:
        return labels

    # Every mask pixel takes the label of its nearest seed (Voronoi split
    # restricted to the foreground) - a cheap watershed substitute.
    # A pixel may only join a seed from its own connected component, so
    # separate debris never gets glued onto a nearby cell.
    _, (iy, ix) = ndi.distance_transform_edt(seed_labels == 0, return_indices=True)
    split = seed_labels[iy, ix] * mask
    same_component = labels[iy, ix] == labels
    split[~same_component] = 0

    # Components with no seed (small but real cells) keep their own label.
    orphan = mask & (split == 0)
    orphan_labels, n_orphan = ndi.label(orphan)
    split[orphan] = orphan_labels[orphan] + n_seeds
    return split


def extract_cells(image: Image.Image, max_cells: int = 150):
    """
    Return (cells, scale) where cells is a list of dicts:
      {"crop": PIL image (NIH style), "box": (x0, y0, x1, y1) in ORIGINAL coords}
    """
    image = image.convert("RGB")
    scale = 1.0
    if max(image.size) > MAX_SIDE:
        scale = MAX_SIDE / max(image.size)
        work = image.resize((round(image.width * scale), round(image.height * scale)),
                            Image.BILINEAR)
    else:
        work = image

    arr = np.asarray(work)
    h, w = arr.shape[:2]
    labels = _split_touching(_foreground_mask(arr))
    n = int(labels.max())
    if n == 0:
        return [], scale

    # Keep only the largest connected piece of every label.
    for idx, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == idx
        parts, k = ndi.label(region)
        if k > 1:
            sizes = ndi.sum(region, parts, index=np.arange(1, k + 1))
            drop = region & (parts != (np.argmax(sizes) + 1))
            labels[sl][drop] = 0

    objects = ndi.find_objects(labels)
    areas = ndi.sum(labels > 0, labels, index=np.arange(1, n + 1))
    typical = _typical_area(areas, h * w)

    cells = []
    for idx, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        area = areas[idx - 1]
        # drop debris/platelets and huge merged blobs
        if area < 0.3 * typical or area > 3.0 * typical:
            continue
        ys, xs = sl
        touches_edge = ys.start == 0 or xs.start == 0 or ys.stop >= h or xs.stop >= w
        if touches_edge and area < 0.7 * typical:
            continue  # mostly cut-off cell at the image border

        pad = 2
        y0, y1 = max(ys.start - pad, 0), min(ys.stop + pad, h)
        x0, x1 = max(xs.start - pad, 0), min(xs.stop + pad, w)
        cell_mask = labels[y0:y1, x0:x1] == idx
        cell_mask = ndi.binary_dilation(cell_mask, iterations=1)
        crop = arr[y0:y1, x0:x1].copy()
        # Stained cells are never near-black; this rejects arrows, text
        # and scale bars drawn on textbook / Google images.
        if np.median(crop[cell_mask].max(axis=1)) < 90:
            continue
        # White blood cells: larger than a red cell and dominated by a big
        # purple nucleus. The classifier never saw them and would read the
        # stain as a parasite, so leave them out.
        if area > 1.4 * typical and _purple_fraction(crop, cell_mask) > 0.18:
            continue
        crop[~cell_mask] = 0     # NIH crops have a pure black background
        cells.append({
            "crop": Image.fromarray(crop),
            "box": tuple(int(round(v / scale)) for v in (x0, y0, x1, y1)),
            "area": float(area),
        })

    # On very large fields keep the most cell-like (closest to typical size).
    if len(cells) > max_cells:
        cells.sort(key=lambda c: abs(c["area"] - typical))
        cells = cells[:max_cells]
    return cells, scale
