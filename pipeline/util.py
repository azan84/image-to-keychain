"""Shared utilities: logging, path handling, image loading."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def get_logger(name: str = "i2k", verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(name)s :: %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def load_image_rgba(path: Path) -> np.ndarray:
    """Load an image as an (H, W, 4) uint8 RGBA numpy array."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.array(img)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def brightness(rgb: np.ndarray) -> np.ndarray:
    """Perceptual brightness 0-255 of an (..., 3) uint8 array."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def save_png(arr: np.ndarray, path: Path) -> None:
    """Save an RGB, RGBA, or L numpy array to PNG."""
    mode = {2: "L", 3: "RGB", 4: "RGBA"}.get(arr.ndim if arr.ndim == 2 else arr.shape[-1], None)
    if arr.ndim == 2:
        mode = "L"
    Image.fromarray(arr, mode=mode).save(path)


def describe_config(cfg: dict[str, Any]) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in cfg.items())
