# Listy

Turn one walkthrough video of resale items into draft listings that are ready to crosslist.

A reseller films themselves showing items one at a time. Listy:

1. **Samples the video** into frames (ffmpeg) and scores each for sharpness (OpenCV).
2. **Finds each item.** Frames are tiled into timestamped contact sheets and Claude marks where each
   item starts and ends, ignoring the seller's own clothes and the room. Long videos are processed in
   batches, and items that straddle a batch boundary are stitched back together.
3. **Picks real photos.** For each item, the sharpest frame from each part of its time range is sent
   to Claude, which picks the front, back, tag and flaw shots and gives a crop box for each. Photos
   are cropped, padded to square and given a light contrast boost. Colors are never shifted and nothing
   is generated, so the photos show the actual item and its condition (eBay and other marketplaces
   require this for used goods).
4. **Drafts the listing** in the same call: title, brand, size, category, condition, flaws, description,
   keywords and a suggested price range. Anything Claude can't read off a tag goes in a
   "check before listing" note instead of being guessed.
5. **Review screen** (phone friendly): edit fields, approve or skip each item.
6. **Export** a ZIP with `listings.csv` and a folder of photos per item.

## Run it

Requires Python 3.11+, ffmpeg, and an Anthropic API key.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn listy.web:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, or your computer's address from your phone on the same Wi-Fi.

Command line, without the web app:

```bash
python -m listy.pipeline path/to/video.mp4
```

Jobs are stored under `data/jobs/` (override with `LISTY_DATA`). The model defaults to
`claude-opus-5-5` (override with `LISTY_MODEL`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a stand-in for Claude, so they need ffmpeg but no API key.

## Layout

| File | What it does |
|---|---|
| `listy/video.py` | ffmpeg frame sampling, sharpness, contact sheets, crop and cleanup |
| `listy/ai.py` | Claude vision calls with structured, validated output |
| `listy/segment.py` | Splits the video into one time range per item |
| `listy/listing.py` | Photo picks and listing draft for one item |
| `listy/pipeline.py` | Runs a job end to end and stores it on disk |
| `listy/export.py` | CSV and ZIP export |
| `listy/web.py`, `listy/static/` | Upload, progress, review and export UI |

## Not built yet

- **Narration:** speech-to-text so "Lululemon, size 6, small pull on the knee" lands in the listing.
- **Publishing:** eBay Sell API, so Nifty can import the listings and crosslist them. Nifty has no
  public API; its beta MCP connector can create drafts but can't upload photos.
- **Background removal**, accounts, and hosted storage.
