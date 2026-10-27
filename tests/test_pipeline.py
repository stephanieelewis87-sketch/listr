import io
import subprocess
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from listy import pipeline
from listy.listing import Box, Listing, PhotoPick, PriceRange
from listy.segment import Segment, SegmentBatch


class FakeModel:
    """Stands in for Claude: two items, 0-3s and 4-7s."""

    def __init__(self):
        self.calls = []

    def ask(self, parts, schema, *, system, effort="medium"):
        self.calls.append((schema, parts))
        if schema is SegmentBatch:
            return SegmentBatch(
                segments=[
                    Segment(start_s=0, end_s=3, label="red tee", continues_previous=False),
                    Segment(start_s=4, end_s=7, label="blue tee", continues_previous=False),
                ],
                open_at_end=False,
            )
        label = parts[0]
        return Listing(
            title=f"Listing for {label[:30]}",
            brand=None, category="Clothing", item_type="T-shirt", department="Unisex",
            size=None, color="Red", material=None, condition="good",
            condition_notes="Light wear", flaws=["small mark on hem"],
            description="A tee.", keywords=["tee"], price=PriceRange(low=8, high=12),
            price_reasoning="Basic tee.",
            photos=[
                PhotoPick(frame=1, role="front", box=Box(x0=0.1, y0=0.1, x1=0.9, y1=0.9)),
                PhotoPick(frame=2, role="back", box=Box(x0=0, y0=0, x1=1, y1=1)),
                PhotoPick(frame=99, role="detail", box=Box(x0=0, y0=0, x1=1, y1=1)),  # out of range, dropped
            ],
            visible_text=[], needs_review=["Brand not visible"],
        )


@pytest.fixture(scope="session")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("v") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=8:size=640x360:rate=10",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_process_job_end_to_end(video, tmp_path):
    model = FakeModel()
    job_id = pipeline.create_job(video, tmp_path)
    pipeline.process_job(job_id, tmp_path, model=model, workers=2)

    job = pipeline.load_job(job_id, tmp_path)
    assert job["status"] == "done", job["error"]
    assert len(job["items"]) == 2
    for item in job["items"]:
        assert item["status"] == "draft"
        assert [p["role"] for p in item["photos"]] == ["front", "back"]
        for p in item["photos"]:
            assert (tmp_path / job_id / "items" / item["id"] / p["file"]).exists()
    # one segmentation call plus one listing call per item
    assert len(model.calls) == 3


def test_segments_stitch_across_batches():
    from listy.segment import segment_items
    from listy.video import Frame

    class TwoBatch:
        n = 0

        def ask(self, parts, schema, *, system, effort="medium"):
            self.n += 1
            if self.n == 1:
                return SegmentBatch(segments=[Segment(start_s=0, end_s=5, label="hoodie", continues_previous=False)],
                                    open_at_end=True)
            assert "still showing: hoodie" in parts[0]
            return SegmentBatch(segments=[
                Segment(start_s=6, end_s=8, label="hoodie", continues_previous=True),
                Segment(start_s=10, end_s=12, label="jeans", continues_previous=False),
            ], open_at_end=False)

    frames = [Frame(index=i, t=float(i), path=Path("missing.jpg")) for i in range(14)]
    import listy.segment as seg
    orig = seg.contact_grid
    seg.contact_grid = lambda fs: __import__("numpy").zeros((10, 10, 3), dtype="uint8")
    try:
        result = segment_items(frames, TwoBatch(), frames_per_grid=4, grids_per_request=2)
    finally:
        seg.contact_grid = orig
    assert [(s.label, s.start_s, s.end_s) for s in result] == [("hoodie", 0, 8), ("jeans", 10, 12)]


def test_web_review_and_export(video, tmp_path, monkeypatch):
    from listy import web

    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "_start", lambda job_id: pipeline.process_job(job_id, tmp_path, model=FakeModel()))
    client = TestClient(web.app)

    with open(video, "rb") as f:
        r = client.post("/api/jobs", files={"video": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200
    job_id = r.json()["id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    first, second = job["items"]

    r = client.patch(f"/api/jobs/{job_id}/items/{first['id']}",
                     json={"status": "approved", "price_override": 15, "listing": {"brand": "Gap", "size": "M"}})
    assert r.status_code == 200
    assert r.json()["listing"]["brand"] == "Gap"

    r = client.patch(f"/api/jobs/{job_id}/items/{first['id']}", json={"listing": {"price": 1}})
    assert r.status_code == 400

    photo = first["photos"][0]["file"]
    assert client.get(f"/api/jobs/{job_id}/items/{first['id']}/photos/{photo}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/items/{first['id']}/photos/..%2Fjob.json").status_code == 404

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job_id}/export.zip").content))
    csv_text = z.read("listings.csv").decode()
    assert "Gap" in csv_text and "15.00" in csv_text
    assert second["id"] not in csv_text  # still a draft, so not exported
    assert f"{first['id']}/01-front.jpg" in z.namelist()

    z_all = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job_id}/export.zip?drafts=true").content))
    assert second["id"] in z_all.read("listings.csv").decode()

    assert client.get("/").status_code == 200
    assert client.get(f"/job/{job_id}").status_code == 200
