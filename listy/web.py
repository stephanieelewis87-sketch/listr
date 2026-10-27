"""Mobile-friendly web app: upload a video, watch progress, review and export listings."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline
from .export import build_zip

ROOT = Path(os.environ.get("LISTY_DATA", pipeline.DATA_DIR))
STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.jpg$")
EDITABLE = {
    "title", "brand", "category", "department", "size", "color", "material",
    "condition", "condition_notes", "flaws", "description", "keywords",
}

app = FastAPI(title="Listy")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _job_or_404(job_id: str) -> dict:
    if not SAFE_ID.match(job_id) or not (ROOT / job_id / "job.json").exists():
        raise HTTPException(404, "job not found")
    return pipeline.load_job(job_id, ROOT)


def _start(job_id: str) -> None:
    threading.Thread(target=pipeline.process_job, args=(job_id, ROOT), daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/job/{job_id}")
def job_page(job_id: str):
    _job_or_404(job_id)
    return FileResponse(STATIC / "job.html")


@app.post("/api/jobs")
def upload(video: UploadFile = File(...)):
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
        raise HTTPException(400, "Please upload a video file (mp4, mov, webm).")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ROOT, suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
    job_id = pipeline.create_job(Path(tmp.name), ROOT, move=True)
    _start(job_id)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    if not ROOT.exists():
        return []
    jobs = []
    for p in sorted(ROOT.glob("*/job.json"), reverse=True):
        j = pipeline.load_job(p.parent.name, ROOT)
        jobs.append({k: j.get(k) for k in ("id", "created", "status", "stage", "progress")} | {"items": len(j["items"])})
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id)


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str):
    job = _job_or_404(job_id)
    if job["status"] == "running":
        raise HTTPException(409, "already running")
    pipeline.update_job(job_id, ROOT, status="queued", stage="Waiting to start", progress=0, error=None, items=[])
    _start(job_id)
    return {"ok": True}


class ItemPatch(BaseModel):
    status: Optional[str] = None
    price_override: Optional[float] = None
    listing: Optional[dict[str, Any]] = None


@app.patch("/api/jobs/{job_id}/items/{item_id}")
def edit_item(job_id: str, item_id: str, patch: ItemPatch):
    job = _job_or_404(job_id)
    item = next((i for i in job["items"] if i["id"] == item_id), None)
    if item is None:
        raise HTTPException(404, "item not found")
    changes: dict[str, Any] = {}
    if patch.status is not None:
        if patch.status not in {"draft", "approved", "skipped"}:
            raise HTTPException(400, "bad status")
        changes["status"] = patch.status
    if patch.price_override is not None:
        changes["price_override"] = patch.price_override
    if patch.listing:
        if not item.get("listing"):
            raise HTTPException(409, "item has no listing yet")
        unknown = set(patch.listing) - EDITABLE
        if unknown:
            raise HTTPException(400, f"not editable: {', '.join(sorted(unknown))}")
        changes["listing"] = item["listing"] | patch.listing
    return pipeline.update_item(job_id, item_id, ROOT, **changes)


@app.get("/api/jobs/{job_id}/items/{item_id}/photos/{name}")
def photo(job_id: str, item_id: str, name: str):
    _job_or_404(job_id)
    if not SAFE_ID.match(item_id) or not SAFE_FILE.match(name):
        raise HTTPException(404)
    path = ROOT / job_id / "items" / item_id / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/export.zip")
def export(job_id: str, drafts: bool = False):
    job = _job_or_404(job_id)
    data = build_zip(job, ROOT / job_id, include_drafts=drafts)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="listy-{job_id}.zip"'},
    )
