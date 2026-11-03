#!/usr/bin/env python3
"""Download the raw inputs for the 2010-2024 training table into data/raw/history/.

1. MIT Election Data + Science Lab (MEDSL) constituency returns
   - U.S. House 1976-2024   doi:10.7910/DVN/IG0UN2
   - U.S. Senate 1976-2024  doi:10.7910/DVN/PEJ5QU
   The Dataverse API lists the files and their MD5 checksums. The files sit behind a
   Dataverse "guestbook" (name, e-mail, institution, position) that we do not fill in
   from a script. If the direct download is refused, we fetch a public GitHub mirror
   of the same CC0 file and accept it ONLY if its MD5 equals the checksum Dataverse
   publishes for the original upload, so the bytes are provably the MEDSL release.

2. Wikipedia (MediaWiki parse API, descriptive User-Agent), rendered HTML of
   - "<year> United States House of Representatives election ratings"
   - "<year> United States House of Representatives elections"
   - "<year> United States Senate elections"
   for 2010..2024. The revision id of each page is saved next to it (revids.json).

3. Wikipedia "Cook Partisan Voting Index" wikitext at the last revision before 15 Oct 2010 and
   15 Oct 2012 (state PVIs then in use; the 2010/2012 Senate pages print no PVI column).

4. National inputs for history_national.csv (data/raw/history/national/):
   - FiveThirtyEight pollster-ratings raw_polls.csv (fivethirtyeight/data on GitHub): every national
     generic-ballot poll ("House-G-US") of the final ~60 days of each cycle 1998-2022, with the
     actual national House vote.
   - UCSB American Presidency Project tables of Gallup presidential job approval (Obama, Trump, Biden).

Usage:  python3 src/fetch_history.py [--force]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "history"
WIKI = RAW / "wikipedia"
UA = "DemocracyDay-forecast/0.1 (student project)"
DV = "https://dataverse.harvard.edu"

MEDSL = {
    "house": dict(doi="doi:10.7910/DVN/IG0UN2", stem="1976-2024-house",
                  out="mit_1976-2024-house.tab",
                  mirrors=["https://raw.githubusercontent.com/Hackquantumcpp/"
                           "snoutcounter-midterms-2026-model/main/data/1976-2024-house.tab"]),
    "senate": dict(doi="doi:10.7910/DVN/PEJ5QU", stem="1976-2024-senate-state",
                   out="mit_1976-2024-senate-state.csv",
                   mirrors=["https://raw.githubusercontent.com/Hackquantumcpp/"
                            "snoutcounter-midterms-2026-model/main/data/1976-2024-senate-state.csv"]),
}
YEARS = list(range(2010, 2026, 2))


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def fetch_medsl(force=False):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    meta = {}
    for key, cfg in MEDSL.items():
        out = RAW / cfg["out"]
        ds = s.get(f"{DV}/api/datasets/:persistentId/", params={"persistentId": cfg["doi"]},
                   timeout=60).json()
        ver = ds["data"]["latestVersion"]
        (RAW / f"dataverse_{key}_dataset.json").write_text(json.dumps(ds, indent=1))
        f = next(x["dataFile"] for x in ver["files"] if x["dataFile"]["filename"].startswith(cfg["stem"]))
        want = f["md5"]
        info = dict(doi=cfg["doi"], version=f'{ver["versionNumber"]}.{ver["versionMinorNumber"]}',
                    release=ver.get("releaseTime"), file_id=f["id"], filename=f["filename"],
                    md5=want, guestbook=ver.get("guestbookId"))
        if out.exists() and not force and md5(out.read_bytes()) == want:
            info["source"] = "cached"
            meta[key] = info
            print(f"[medsl] {key}: cached, md5 ok")
            continue
        r = s.get(f"{DV}/api/access/datafile/{f['id']}", params={"format": "original"}, timeout=120)
        body = r.content
        if r.ok and md5(body) == want:
            info["source"] = f"{DV}/api/access/datafile/{f['id']}?format=original"
        else:
            print(f"[medsl] {key}: Dataverse refused ({r.status_code} {body[:120]!r}); trying mirror")
            for m in cfg["mirrors"]:
                body = s.get(m, timeout=120).content
                if md5(body) == want:
                    info["source"] = m
                    break
            else:
                raise SystemExit(f"no copy of {cfg['filename']} with md5 {want} found")
        out.write_bytes(body)
        meta[key] = info
        print(f"[medsl] {key}: {len(body)} bytes from {info['source']} (md5 {want} verified)")
    (RAW / "medsl_provenance.json").write_text(json.dumps(meta, indent=1))


def wiki_titles():
    t = []
    for y in YEARS:
        t.append(f"{y} United States House of Representatives election ratings")
        t.append(f"{y} United States House of Representatives elections")
        t.append(f"{y} United States Senate elections")
    return t


def fetch_wiki(force=False, titles=None):
    WIKI.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    revf = WIKI / "revids.json"
    revids = json.loads(revf.read_text()) if revf.exists() else {}
    for title in titles or wiki_titles():
        out = WIKI / (title.replace(" ", "_") + ".html")
        if out.exists() and title in revids and not force:
            continue
        for attempt in range(6):
            r = s.get("https://en.wikipedia.org/w/api.php",
                      params=dict(action="parse", page=title, prop="text|revid", format="json",
                                  redirects=1, formatversion=2, disableeditsection=1,
                                  maxlag=5),
                      timeout=120)
            if r.status_code in (429, 503):
                wait = int(r.headers.get("Retry-After", 0) or 0) or 20 * (attempt + 1)
                print(f"[wiki] {title}: HTTP {r.status_code}, waiting {wait}s")
                time.sleep(wait)
                continue
            break
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            print(f"[wiki] {title}: {d['error'].get('info')}")
            continue
        out.write_text(d["parse"]["text"])
        revids[title] = d["parse"]["revid"]
        revf.write_text(json.dumps(revids, indent=1, sort_keys=True))
        print(f"[wiki] {title}: rev {d['parse']['revid']}, {len(d['parse']['text'])} chars")
        time.sleep(3)
    revf.write_text(json.dumps(revids, indent=1, sort_keys=True))


PVI_REVS = {2010: "2010-10-15T00:00:00Z", 2012: "2012-10-15T00:00:00Z"}


def fetch_cook_pvi_revisions(force=False):
    """Wikitext of 'Cook Partisan Voting Index' as it stood shortly before the 2010 / 2012 elections."""
    s = requests.Session()
    s.headers["User-Agent"] = UA
    revf = WIKI / "revids.json"
    revids = json.loads(revf.read_text()) if revf.exists() else {}
    for year, ts in PVI_REVS.items():
        out = WIKI / f"Cook_Partisan_Voting_Index_asof_{year}-10-15.wikitext"
        key = f"Cook Partisan Voting Index (as of {ts[:10]})"
        if out.exists() and key in revids and not force:
            continue
        r = s.get("https://en.wikipedia.org/w/api.php",
                  params=dict(action="query", prop="revisions", titles="Cook Partisan Voting Index",
                              rvstart=ts, rvdir="older", rvlimit=1, rvprop="ids|timestamp|content",
                              rvslots="main", format="json", formatversion=2), timeout=60)
        r.raise_for_status()
        rev = r.json()["query"]["pages"][0]["revisions"][0]
        out.write_text(rev["slots"]["main"]["content"])
        revids[key] = rev["revid"]
        print(f"[wiki] Cook PVI as of {ts[:10]}: rev {rev['revid']} ({rev['timestamp']})")
        time.sleep(3)
    revf.write_text(json.dumps(revids, indent=1, sort_keys=True))


NATIONAL = RAW / "national"
NATIONAL_FILES = {
    "538_raw_polls.csv":
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv",
    "ucsb_gallup_approval_obama.html":
        "https://www.presidency.ucsb.edu/statistics/data/barack-obama-public-approval",
    "ucsb_gallup_approval_trump1.html":
        "https://www.presidency.ucsb.edu/statistics/data/donald-j-trump-public-approval",
    "ucsb_gallup_approval_biden.html":
        "https://www.presidency.ucsb.edu/statistics/data/joseph-r-biden-public-approval",
}


def fetch_national(force=False):
    NATIONAL.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    prov_f = NATIONAL / "provenance.json"
    prov = json.loads(prov_f.read_text()) if prov_f.exists() else {}
    for name, url in NATIONAL_FILES.items():
        out = NATIONAL / name
        if out.exists() and not force:
            continue
        r = s.get(url, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        prov[name] = dict(url=url, md5=md5(r.content), fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(f"[national] {name}: {len(r.content)} bytes")
        time.sleep(2)
    prov_f.write_text(json.dumps(prov, indent=1, sort_keys=True))


if __name__ == "__main__":
    force = "--force" in sys.argv
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_medsl(force)
    fetch_wiki(force)
    fetch_cook_pvi_revisions(force)
    fetch_national(force)
