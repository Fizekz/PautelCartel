"""Download The Downballot / Daily Kos Elections special-election spreadsheets (all tabs) as CSV.

Sheets are listed at https://www.the-downballot.com/p/data#special-elections. Raw CSVs go to
data/specials/raw/<cycle>_<gid>.csv. Run: .venv/bin/python src/models/kavi-e/fetch_specials.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "specials" / "raw"
SHEETS = {
    "2017_18": "1C2MVeM2K7WgqmJw5RCQbWyTo2u73CX1pI8zw_G-7BJo",
    "2019_20": "1dXQhlzCAw05fC7P21eCivdfLOB-F3nykPVRPzG49U9g",
    "2021_22": "1G1SeLtlfDY9xqqVfFTsoh0jwr6h7aQzOOWpJ29aez2w",
    "2023_24": "1ajyphWQru9TgDDiBe8kvEmApBEXND2wl9MVaxi1dndk",
    "2025_26": "1JGk1r1VXnxBrAIVHz1C5HTB5jxCO6Zw4QNPivdhyWHw",
    "index_1989_2022": "1mP8z1TWsrJbWGXk_pz3QgOZQpAmQxLY6sZjPp4qzY6c",
}
UA = {"User-Agent": "Mozilla/5.0"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for cycle, sid in SHEETS.items():
        page = get(f"https://docs.google.com/spreadsheets/d/{sid}/htmlview").decode("utf-8", "ignore")
        (RAW / f"{cycle}_htmlview.html").write_text(page)
        gids = sorted(set(re.findall(r"gid=(\d+)", page)))
        for gid in gids:
            try:
                data = get(f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}")
            except Exception as exc:  # noqa: BLE001 - report and continue with the other tabs
                print(f"{cycle} {gid}: failed ({exc})")
                continue
            (RAW / f"{cycle}_{gid}.csv").write_bytes(data)
            first = data.decode("utf-8", "ignore").splitlines()[:3]
            print(cycle, gid, len(data), "|", " / ".join(line[:60] for line in first))
    return 0


if __name__ == "__main__":
    sys.exit(main())
