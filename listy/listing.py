"""Turn one item's frames into a photo set and a draft listing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import cv2
from pydantic import BaseModel, Field

from .ai import VisionModel
from .segment import Segment
from .video import Frame, auto_enhance, crop_box, encode_jpeg, load_resized, square_pad

SYSTEM = """You write accurate resale listings for an online reseller (eBay, Poshmark, Mercari, Depop, Etsy).
You see real frames from a video of ONE item for sale, each labeled with a frame number.

Accuracy rules, because buyers return items that don't match the listing:
- Only state a brand, size, material or measurement if you can read it on a tag or label, or the
  transcript says it. Otherwise use null and add a note to needs_review.
- Describe every visible flaw (stains, pilling, holes, fading, missing buttons). Never hide wear.
- Do not invent features you cannot see.
- Ignore the clothes the seller is wearing and anything in the background.

Photos: choose the real frames that best show the item, in this order: front, back, then tag/label,
then close-ups of details and flaws. Pick 3 to 8 frames, skip blurry or near-duplicate ones.
For each, give a tight box around the item (x0, y0, x1, y1 as fractions of width and height,
origin top-left) so it can be cropped. Leave a little margin and include the whole item if visible.

Pricing: suggest a realistic used-resale price range in USD based on brand, type and condition."""

Condition = Literal["new_with_tags", "new_without_tags", "excellent", "good", "fair", "poor"]
Role = Literal["front", "back", "label", "detail", "flaw", "other"]


class Box(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class PhotoPick(BaseModel):
    frame: int = Field(description="Frame number as labeled")
    role: Role
    box: Box


class PriceRange(BaseModel):
    low: float
    high: float


class Listing(BaseModel):
    title: str = Field(description="Marketplace title, at most 80 characters, keyword-rich, no ALL CAPS")
    brand: Optional[str]
    category: str = Field(description="e.g. Women's Clothing > Tops > T-Shirts")
    item_type: str
    department: Optional[str] = Field(description="Women, Men, Unisex, Girls, Boys, Baby, Home, etc.")
    size: Optional[str]
    color: str
    material: Optional[str]
    condition: Condition
    condition_notes: str
    flaws: list[str]
    description: str = Field(description="2 to 5 short paragraphs or bullet lines, honest and specific")
    keywords: list[str]
    price: PriceRange
    price_reasoning: str
    photos: list[PhotoPick]
    visible_text: list[str] = Field(description="Any text read from tags or labels")
    needs_review: list[str] = Field(description="Things the seller should confirm before publishing")


def pick_candidates(frames: list[Frame], seg: Segment, max_frames: int = 14) -> list[Frame]:
    """Sharpest frame in each of max_frames equal time slices of the segment."""
    inside = [f for f in frames if seg.start_s - 0.25 <= f.t <= seg.end_s + 0.25]
    if len(inside) <= max_frames:
        return inside
    span = (seg.end_s - seg.start_s) or 1.0
    bins: dict[int, Frame] = {}
    for f in inside:
        b = min(max_frames - 1, int((f.t - seg.start_s) / span * max_frames))
        if b not in bins or f.sharpness > bins[b].sharpness:
            bins[b] = f
    return [bins[k] for k in sorted(bins)]


def draft_listing(candidates: list[Frame], seg: Segment, model: VisionModel, transcript: str = "") -> Listing:
    parts: list[str | bytes] = [
        f"Item: {seg.label} (shown {seg.start_s:.1f}s to {seg.end_s:.1f}s)."
    ]
    if transcript.strip():
        parts.append(f"What the seller said while showing it: \"{transcript.strip()}\"")
    for i, f in enumerate(candidates, start=1):
        parts += [f"Frame {i} (t={f.t:.1f}s):", encode_jpeg(load_resized(f.path, 1024))]
    parts.append("Write the listing and choose the photos.")
    return model.ask(parts, Listing, system=SYSTEM)


def render_photos(candidates: list[Frame], listing: Listing, out_dir: Path) -> list[dict]:
    """Crop, clean up and save the chosen real frames. Returns photo records in listing order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {"front": 0, "back": 1, "label": 2, "detail": 3, "flaw": 4, "other": 5}
    picks = sorted(
        (p for p in listing.photos if 1 <= p.frame <= len(candidates)),
        key=lambda p: order[p.role],
    )
    if not picks and candidates:
        best = max(candidates, key=lambda f: f.sharpness)
        picks = [PhotoPick(frame=candidates.index(best) + 1, role="front", box=Box(x0=0, y0=0, x1=1, y1=1))]

    photos, seen = [], set()
    for n, pick in enumerate(picks, start=1):
        if (pick.frame, pick.role) in seen:
            continue
        seen.add((pick.frame, pick.role))
        frame = candidates[pick.frame - 1]
        img = cv2.imread(str(frame.path))
        img = square_pad(auto_enhance(crop_box(img, (pick.box.x0, pick.box.y0, pick.box.x1, pick.box.y1))))
        name = f"{n:02d}-{pick.role}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        photos.append({"file": name, "role": pick.role, "source_t": frame.t})
    return photos
