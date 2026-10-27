"""Split a walkthrough video into one time range per item.

Frames are tiled into timestamped contact grids and sent to Claude in chunks.
Each chunk is told what the previous chunk ended on so items that straddle a
chunk boundary are stitched back together.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .ai import VisionModel
from .video import Frame, contact_grid, encode_jpeg

SYSTEM = """You help an online reseller turn a walkthrough video into marketplace listings.
The seller films themselves showing items for sale one at a time: holding them up, turning them,
and showing tags, labels and flaws. You see contact sheets of frames, each stamped with its time in seconds.

Your job is to find where each item for sale starts and ends.
- Only count things being presented for sale. Ignore the clothes the seller is wearing, furniture,
  boxes and anything else in the room.
- Close-ups of a tag, label or flaw belong to the item being shown at that moment.
- Gaps where no item is shown (seller out of frame, reaching for the next item) belong to no item.
- If two items look alike, use color, logos, cut and context to tell them apart. When the seller
  puts an item down and picks up a different one, that is a new item.
- Be precise with times: use the timestamps on the frames."""


class Segment(BaseModel):
    start_s: float = Field(description="Time of the first frame showing this item")
    end_s: float = Field(description="Time of the last frame showing this item")
    label: str = Field(description="Short description, e.g. 'gray zip-up hoodie with chest logo'")
    continues_previous: bool = Field(
        description="True only for the first segment of this batch, if it is the same item the previous batch ended on"
    )


class SegmentBatch(BaseModel):
    segments: list[Segment]
    open_at_end: bool = Field(description="True if the last item is still being shown at the end of this batch")


def _chunks(frames: list[Frame], size: int) -> list[list[Frame]]:
    return [frames[i:i + size] for i in range(0, len(frames), size)]


def segment_items(
    frames: list[Frame],
    model: VisionModel,
    *,
    sample_every_s: float = 1.0,
    frames_per_grid: int = 12,
    grids_per_request: int = 6,
    on_progress=None,
) -> list[Segment]:
    if not frames:
        return []
    step = max(1, round(sample_every_s / max(frames[1].t - frames[0].t, 1e-6))) if len(frames) > 1 else 1
    sampled = frames[::step]
    per_request = frames_per_grid * grids_per_request
    batches = _chunks(sampled, per_request)

    merged: list[Segment] = []
    prev_open: str | None = None
    for n, batch in enumerate(batches):
        grids = [encode_jpeg(contact_grid(g)) for g in _chunks(batch, frames_per_grid)]
        intro = (
            f"Batch {n + 1} of {len(batches)}, covering {batch[0].t:.1f}s to {batch[-1].t:.1f}s "
            f"(one frame every {sample_every_s:g}s)."
        )
        if prev_open:
            intro += f" The previous batch ended while the seller was still showing: {prev_open}."
        else:
            intro += " No item was being shown at the end of the previous batch."
        parts: list[str | bytes] = [intro]
        for i, g in enumerate(grids):
            parts += [f"Sheet {i + 1}:", g]
        parts.append("List every item shown in this batch, in order, with its start and end time.")

        result = model.ask(parts, SegmentBatch, system=SYSTEM)
        for i, seg in enumerate(sorted(result.segments, key=lambda s: s.start_s)):
            if i == 0 and seg.continues_previous and prev_open and merged:
                merged[-1].end_s = max(merged[-1].end_s, seg.end_s)
            else:
                merged.append(seg)
        prev_open = merged[-1].label if (result.open_at_end and merged) else None
        if on_progress:
            on_progress(n + 1, len(batches))

    return [s for s in merged if s.end_s >= s.start_s]
