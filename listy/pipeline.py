"""End-to-end processing of one uploaded video, persisted as a job folder.

data/jobs/<job_id>/
    video.<ext>
    frames/            sampled frames
    job.json           status + items
    items/<item_id>/   cropped photos for each item
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ai import ClaudeVision, VisionModel
from .listing import draft_listing, pick_candidates, render_photos
from .segment import segment_items
from .video import extract_frames, probe_duration

DATA_DIR = Path("data/jobs")
_locks: dict[str, threading.Lock] = {}


def job_dir(job_id: str, root: Path = DATA_DIR) -> Path:
    return root / job_id


def load_job(job_id: str, root: Path = DATA_DIR) -> dict:
    return json.loads((job_dir(job_id, root) / "job.json").read_text())


def save_job(job: dict, root: Path = DATA_DIR) -> None:
    path = job_dir(job["id"], root) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2))
    tmp.replace(path)


def update_job(job_id: str, root: Path = DATA_DIR, **changes) -> dict:
    lock = _locks.setdefault(job_id, threading.Lock())
    with lock:
        job = load_job(job_id, root)
        job.update(changes)
        save_job(job, root)
        return job


def update_item(job_id: str, item_id: str, root: Path = DATA_DIR, **changes) -> dict:
    lock = _locks.setdefault(job_id, threading.Lock())
    with lock:
        job = load_job(job_id, root)
        for item in job["items"]:
            if item["id"] == item_id:
                item.update(changes)
                save_job(job, root)
                return item
    raise KeyError(item_id)


def create_job(video_src: Path, root: Path = DATA_DIR, move: bool = False) -> str:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = job_dir(job_id, root)
    d.mkdir(parents=True)
    dest = d / f"video{video_src.suffix.lower() or '.mp4'}"
    (shutil.move if move else shutil.copyfile)(str(video_src), str(dest))
    save_job({
        "id": job_id,
        "video": dest.name,
        "created": time.time(),
        "status": "queued",
        "stage": "Waiting to start",
        "progress": 0.0,
        "error": None,
        "items": [],
    }, root)
    return job_id


def process_job(job_id: str, root: Path = DATA_DIR, model: VisionModel | None = None, workers: int = 4) -> None:
    d = job_dir(job_id, root)
    try:
        model = model or ClaudeVision()
        job = load_job(job_id, root)
        video = d / job["video"]

        update_job(job_id, root, status="running", stage="Reading video", progress=0.02,
                   duration_s=round(probe_duration(video), 1))
        frames = extract_frames(video, d / "frames")

        update_job(job_id, root, stage="Finding items in the video", progress=0.1)

        def seg_progress(done, total):
            update_job(job_id, root, progress=0.1 + 0.3 * done / total,
                       stage=f"Finding items in the video ({done}/{total})")

        segments = segment_items(frames, model, on_progress=seg_progress)
        items = [{
            "id": f"item{n:03d}",
            "status": "processing",
            "segment": s.model_dump(),
            "listing": None,
            "photos": [],
            "error": None,
        } for n, s in enumerate(segments, start=1)]
        update_job(job_id, root, items=items, stage=f"Writing {len(items)} listings", progress=0.4)

        done = 0
        done_lock = threading.Lock()

        def work(item, seg):
            nonlocal done
            try:
                cands = pick_candidates(frames, seg)
                listing = draft_listing(cands, seg, model)
                photos = render_photos(cands, listing, d / "items" / item["id"])
                update_item(job_id, item["id"], root, status="draft",
                            listing=listing.model_dump(exclude={"photos"}), photos=photos)
            except Exception as e:  # keep going; one bad item shouldn't sink the batch
                update_item(job_id, item["id"], root, status="failed", error=str(e))
            with done_lock:
                done += 1
                n = done
            update_job(job_id, root, progress=0.4 + 0.6 * n / max(len(items), 1),
                       stage=f"Writing listings ({n}/{len(items)})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, items, segments))

        update_job(job_id, root, status="done", stage="Ready to review", progress=1.0)
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, root, status="failed", stage="Failed", error=str(e))


def main() -> None:
    ap = argparse.ArgumentParser(description="Turn a reseller walkthrough video into draft listings.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    job_id = create_job(args.video, args.data)
    print(f"job {job_id}")
    process_job(job_id, args.data)
    job = load_job(job_id, args.data)
    print(f"{job['status']}: {len(job['items'])} items -> {job_dir(job_id, args.data)}")


if __name__ == "__main__":
    main()
