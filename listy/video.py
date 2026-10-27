"""Frame extraction and image helpers built on ffmpeg and OpenCV."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    t: float  # seconds from start of video
    path: Path
    sharpness: float = 0.0


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_frames(video: Path, out_dir: Path, fps: float = 2.0, max_width: int = 1600) -> list[Frame]:
    """Sample frames at a fixed rate. Frame i (0-based) sits at t = i / fps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-vf", f"fps={fps},scale='min({max_width},iw)':-2",
            "-q:v", "3", str(out_dir / "%06d.jpg"),
        ],
        check=True,
    )
    frames = []
    for i, path in enumerate(sorted(out_dir.glob("*.jpg"))):
        frames.append(Frame(index=i, t=round(i / fps, 2), path=path, sharpness=sharpness(path)))
    return frames


def sharpness(path: Path) -> float:
    """Variance of the Laplacian: higher means less motion blur."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def load_resized(path: Path, long_edge: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def contact_grid(frames: list[Frame], cols: int = 4, tile_width: int = 360) -> np.ndarray:
    """Tile frames into one image, each stamped with its timestamp so Claude can cite times."""
    tiles = []
    for f in frames:
        img = load_resized(f.path, tile_width)
        img = cv2.resize(img, (tile_width, int(tile_width * img.shape[0] / img.shape[1])))
        label = f"{f.t:.1f}s"
        cv2.rectangle(img, (0, 0), (12 + 14 * len(label), 30), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(img)
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile[:th, :tw]
    return grid


def crop_box(img: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.04) -> np.ndarray:
    """Crop a normalized (x0, y0, x1, y1) box with a little padding, clamped to the image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(1.0, x1 + pad), min(1.0, y1 + pad)
    px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    if px1 - px0 < 32 or py1 - py0 < 32:
        return img
    return img[py0:py1, px0:px1]


def square_pad(img: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Pad to a square canvas, which every marketplace thumbnail prefers."""
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), color, dtype=np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def auto_enhance(img: np.ndarray) -> np.ndarray:
    """Truthful cleanup only: gentle local contrast on lightness.

    Color channels are left alone so the item's real color is preserved, and no
    content is added or removed, so condition stays visible.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
