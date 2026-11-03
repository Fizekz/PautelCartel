"""Polite Wikipedia page fetcher with an on-disk cache (used by fetch_ratings.py).

Pages are fetched through the MediaWiki parse API (rendered HTML + revision id) and cached as JSON under
data/raw/ratings/. A cached page is reused unless refresh=True, so rebuilding from raw files needs no network.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "DemocracyDayForecastResearch/0.1 (Stanford Model Citizens student contest; python-requests)"
PAUSE_SECONDS = 4.0
MAX_TRIES = 6


def cache_path(raw_dir: Path, title: str) -> Path:
    return raw_dir / (title.replace(" ", "_").replace("/", "_") + ".json")


def fetch_page(title: str, raw_dir: Path, refresh: bool = False) -> dict | None:
    """Return {'title','revid','html','fetched_at'} for a page, or None if it does not exist."""
    path = cache_path(raw_dir, title)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    params = {"action": "parse", "page": title, "prop": "text|revid", "format": "json",
              "formatversion": "2", "redirects": "1"}
    for attempt in range(MAX_TRIES):
        time.sleep(PAUSE_SECONDS * (1 + attempt))
        resp = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code == 429 or "too many requests" in resp.text[:200].lower():
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            if data["error"].get("code") == "missingtitle":
                return None
            raise RuntimeError(f"{title}: {data['error']}")
        page = {"title": data["parse"]["title"], "revid": data["parse"]["revid"], "html": data["parse"]["text"],
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(page))
        return page
    raise RuntimeError(f"{title}: rate limited after {MAX_TRIES} tries")
