"""Export approved items as a CSV plus a ZIP of photos, ready to paste into a crosslister."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

CONDITION_LABELS = {
    "new_with_tags": "New with tags",
    "new_without_tags": "New without tags",
    "excellent": "Pre-owned - Excellent",
    "good": "Pre-owned - Good",
    "fair": "Pre-owned - Fair",
    "poor": "Pre-owned - Poor",
}

COLUMNS = [
    "sku", "title", "description", "price", "brand", "category", "department", "size",
    "color", "material", "condition", "condition_notes", "flaws", "keywords", "photo_files",
]


def exportable(job: dict, include_drafts: bool = False) -> list[dict]:
    ok = {"approved", "draft"} if include_drafts else {"approved"}
    return [i for i in job["items"] if i["status"] in ok and i.get("listing")]


def full_description(listing: dict) -> str:
    text = listing["description"].strip()
    if listing.get("flaws"):
        text += "\n\nFlaws: " + "; ".join(listing["flaws"])
    return text


def to_rows(job: dict, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        l = item["listing"]
        price = item.get("price_override") or l["price"]["high"]
        rows.append({
            "sku": f"{job['id']}-{item['id']}",
            "title": l["title"],
            "description": full_description(l),
            "price": f"{float(price):.2f}",
            "brand": l.get("brand") or "",
            "category": l.get("category") or "",
            "department": l.get("department") or "",
            "size": l.get("size") or "",
            "color": l.get("color") or "",
            "material": l.get("material") or "",
            "condition": CONDITION_LABELS.get(l["condition"], l["condition"]),
            "condition_notes": l.get("condition_notes") or "",
            "flaws": "; ".join(l.get("flaws") or []),
            "keywords": ", ".join(l.get("keywords") or []),
            "photo_files": "; ".join(f"{item['id']}/{p['file']}" for p in item["photos"]),
        })
    return rows


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def build_zip(job: dict, job_path: Path, include_drafts: bool = False) -> bytes:
    items = exportable(job, include_drafts)
    rows = to_rows(job, items)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("listings.csv", csv_text(rows))
        for item, row in zip(items, rows):
            for p in item["photos"]:
                z.write(job_path / "items" / item["id"] / p["file"], f"{item['id']}/{p['file']}")
            text = f"{row['title']}\n\nPrice: ${row['price']}\nCondition: {row['condition']}\n\n{row['description']}\n"
            z.writestr(f"{item['id']}/listing.txt", text)
    return buf.getvalue()
