"""Shared preprocessing: alpha masking + color quantization.

Produces a dense label map of the input image so downstream stages
(line extraction, color extraction, silhouette) all work from the same
deterministic color clustering. k-means merges similar tones/shades into
a single cluster, so soft shading does not get treated as a color change.

Results are cached in-memory keyed by (input_path_mtime, fingerprint of
relevant config keys) so repeated calls within one pipeline run are free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .util import get_logger, load_image_rgba


@dataclass
class PreprocessResult:
    rgba: np.ndarray          # (H, W, 4) uint8 original image
    bg_mask: np.ndarray       # (H, W) bool — True where transparent/background
    labels: np.ndarray        # (H, W) int — cluster id for opaque pixels, -1 for bg
    palette: np.ndarray       # (K, 3) uint8 — centroid RGB for each cluster id

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgba.shape[:2]


_cache: dict[tuple, PreprocessResult] = {}


def _cfg_fingerprint(cfg: dict[str, Any]) -> tuple:
    keys = ("max_colors", "alpha_threshold", "kmeans_seed", "kmeans_sample")
    return tuple(cfg.get(k) for k in keys)


def _kmeans(pixels: np.ndarray, k: int, *, seed: int, sample: int, iters: int) -> np.ndarray:
    """Lloyd's k-means. Samples `sample` pixels to fit centroids, then
    returns the `k x 3` float centroid array. Caller assigns all pixels."""
    rng = np.random.default_rng(seed)
    if len(pixels) > sample:
        idx = rng.choice(len(pixels), size=sample, replace=False)
        pts = pixels[idx].astype(np.float32)
    else:
        pts = pixels.astype(np.float32)

    # k-means++ init
    centers = np.empty((k, 3), dtype=np.float32)
    centers[0] = pts[rng.integers(len(pts))]
    for i in range(1, k):
        d2 = np.min(np.sum((pts[:, None, :] - centers[None, :i, :]) ** 2, axis=2), axis=1)
        total = d2.sum()
        if total <= 0:
            centers[i] = pts[rng.integers(len(pts))]
            continue
        probs = d2 / total
        centers[i] = pts[rng.choice(len(pts), p=probs)]

    for _ in range(iters):
        d = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if np.any(m):
                new_centers[j] = pts[m].mean(axis=0)
        if np.allclose(new_centers, centers, atol=0.25):
            centers = new_centers
            break
        centers = new_centers
    return centers


def _assign_labels(rgb_opaque: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each pixel in rgb_opaque (N, 3) to its nearest centroid.
    Done in chunks to keep memory bounded."""
    N = len(rgb_opaque)
    out = np.empty(N, dtype=np.int32)
    chunk = 200_000
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        block = rgb_opaque[start:stop].astype(np.float32)
        d = np.sum((block[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        out[start:stop] = d.argmin(axis=1)
    return out


def preprocess(input_image: Path, cfg: dict[str, Any]) -> PreprocessResult:
    """Load, alpha-mask, quantize. Cached per-file per-config-fingerprint."""
    log = get_logger(verbose=cfg.get("verbose", True))

    key = (str(input_image.resolve()), input_image.stat().st_mtime_ns, _cfg_fingerprint(cfg))
    if key in _cache:
        log.debug("preprocess: cache hit")
        return _cache[key]

    rgba = load_image_rgba(input_image)
    H, W, _ = rgba.shape
    alpha_cut = int(cfg.get("alpha_threshold", 128))
    bg_mask = rgba[..., 3] < alpha_cut

    # If the image has no real transparency, defer background detection until
    # after clustering — the background often survives as a baked-in flat fill
    # (e.g. light-grey checkerboard from an AI generator). We'll detect it by
    # finding clusters that dominate the image corners.
    alpha_bg_px = int(bg_mask.sum())
    need_corner_bg = alpha_bg_px == 0
    log.info("preprocess: alpha-based bg pixels: %d  (fallback to corner detection: %s)",
             alpha_bg_px, need_corner_bg)

    rgb = rgba[..., :3]
    opaque_flat = rgb[~bg_mask].reshape(-1, 3)
    if len(opaque_flat) == 0:
        raise RuntimeError("Input image is fully transparent — nothing to vectorize.")

    k = int(cfg.get("max_colors", 6))
    seed = int(cfg.get("kmeans_seed", 0))
    sample = int(cfg.get("kmeans_sample", 20_000))
    iters = int(cfg.get("kmeans_iters", 15))

    log.info("preprocess: k-means k=%d on %d opaque pixels (sample %d, iters %d)",
             k, len(opaque_flat), sample, iters)
    centers = _kmeans(opaque_flat, k=k, seed=seed, sample=sample, iters=iters)

    opaque_labels = _assign_labels(opaque_flat, centers)
    labels = np.full((H, W), -1, dtype=np.int32)
    labels[~bg_mask] = opaque_labels

    palette = np.clip(np.round(centers), 0, 255).astype(np.uint8)

    # Corner-based background detection when alpha didn't find any bg pixels.
    # Any cluster that covers >=corner_bg_ratio of the four corner patches is
    # treated as background and rewritten to label -1.
    bg_cluster_ids: list[int] = []
    if need_corner_bg:
        patch = int(cfg.get("corner_patch_px", 24))
        corner_labels = np.concatenate([
            labels[:patch, :patch].ravel(),
            labels[:patch, -patch:].ravel(),
            labels[-patch:, :patch].ravel(),
            labels[-patch:, -patch:].ravel(),
        ])
        corner_labels = corner_labels[corner_labels >= 0]
        corner_total = len(corner_labels)
        threshold = float(cfg.get("corner_bg_ratio", 0.10))
        uniq, cnt = np.unique(corner_labels, return_counts=True)
        for cid, c in zip(uniq.tolist(), cnt.tolist()):
            if c / corner_total >= threshold:
                bg_cluster_ids.append(int(cid))
        # Manual override
        override = cfg.get("background_cluster_ids")
        if isinstance(override, (list, tuple)) and len(override) > 0:
            bg_cluster_ids = [int(x) for x in override]
        if bg_cluster_ids:
            log.info("  corner-detected background clusters: %s", bg_cluster_ids)
            bg_drop = np.isin(labels, np.array(bg_cluster_ids, dtype=np.int32))
            labels[bg_drop] = -1
            bg_mask = bg_mask | bg_drop

    # Log the palette for sanity
    for i, c in enumerate(palette):
        count = int((labels == i).sum())
        marker = " [BG]" if i in bg_cluster_ids else ""
        log.info("  cluster %d: RGB#%02x%02x%02x  %d px (%.1f%%)%s",
                 i, c[0], c[1], c[2], count, 100.0 * count / (H * W), marker)

    result = PreprocessResult(rgba=rgba, bg_mask=bg_mask, labels=labels, palette=palette)
    _cache[key] = result
    return result
