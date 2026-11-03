#!/usr/bin/env python3
"""Refresh the shared 2026 Wikipedia poll file (pau-f, Nov 2 2026).

Re-runs the shared fetcher's parser (model/src/fetch_polls.py, vendored as fetch_polls_vendored.py with the HTML
parser switched from lxml, not installed here, to html.parser) on the CURRENT revisions of the same Wikipedia
pages, through the MediaWiki API only, and writes a new, dated poll file next to (never over) the shared one.

Differences from the shared fetcher, all about politeness and dating:
  * >= 5 s between API requests (the shared one used 1.5 s; an earlier agent was rate-limited at 2.5 s),
    exponential backoff on HTTP 429 / 503, and a hard stop after the second rate-limit response.
  * generic User-Agent, no e-mail.
  * every page is parsed by revision id (oldid): one batched revisions query first records the id and the
    timestamp of the latest revision, which must be before the 2026-11-03T23:00Z cutoff (rvstart = cutoff).
  * Senate pages first, then House state pages with competitive races, then the rest, so a stop leaves the
    important pages done.
  * --offline DIR re-parses an existing cache (the Nov 1 pages) with no network, to check that the vendored
    parser reproduces the current file.

Outputs (OUT = pbase/data/refresh_1102/<tag>/)
  polls_2026.csv, polls_2026_all_versions.csv, race_poll_aggregates_2026.csv, polls_2026_district_generic.csv,
  national_2026.csv, pages.csv (title, revid, revision timestamp, fetch time, poll rows), cache/*.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

import fetch_polls_vendored as F

HERE = Path(__file__).resolve().parents[1]
SHARED = HERE / "data" / "build"
KALSHI_NOV1 = HERE / "data" / "kalshi" / "kalshi_races_2026-11-01T074812Z.csv"  # same file the Nov 1 build used
CUTOFF = "2026-11-03T23:00:00Z"
MIN_GAP = 5.0
UA = "ModelCitizensPollRefresh/1.1 (student election-forecast research) python-requests/" + requests.__version__


class RateLimited(RuntimeError):
    pass


class Wiki(F.Wiki):
    """API client: >= MIN_GAP s between calls, exponential backoff, stop at the 2nd rate-limit response."""

    def __init__(self, cache: Path, offline: bool = False):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.refresh = False
        self.offline = offline
        self.last = 0.0
        self.n_req = 0
        self.n_limited = 0
        self.gap = MIN_GAP
        self.cache = cache
        self.revs: dict[str, dict] = {}   # title -> {revid, timestamp, title(normalised)}
        self.pages: list[dict] = []
        cache.mkdir(parents=True, exist_ok=True)

    def _get(self, params: dict) -> dict:
        if self.offline:
            raise RuntimeError("offline mode: no network")
        params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
        for attempt in range(5):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.n_req += 1
            r = self.s.get(F.API, params=params, timeout=90)
            lag = r.status_code == 200 and '"code":"maxlag"' in r.text.replace(" ", "")
            if r.status_code in (429, 503) or lag:
                self.n_limited += 1
                ra = r.headers.get("retry-after")
                sleep = max(float(ra) if ra and ra.isdigit() else 0, 15 * 2 ** attempt)
                print(f"  RATE LIMIT #{self.n_limited}: HTTP {r.status_code} lag={lag}; retry-after={ra}",
                      file=sys.stderr, flush=True)
                if self.n_limited >= 2:
                    raise RateLimited(f"rate-limited twice (last: HTTP {r.status_code}); stopping")
                time.sleep(sleep)
                continue
            if r.status_code >= 500:
                time.sleep(15 * 2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up on {params}")

    def resolve(self, titles: list[str]):
        """Latest revision (id, timestamp) at or before CUTOFF for up to 50 titles per request."""
        todo = [t for t in dict.fromkeys(titles) if t not in self.revs]
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            d = self._get(dict(action="query", titles="|".join(chunk), prop="revisions", rvprop="ids|timestamp",
                               redirects=1))
            q = d.get("query", {})
            alias = {}
            for k in ("normalized", "redirects"):
                for x in q.get(k, []):
                    alias[x["to"]] = alias.get(x["from"], x["from"])
            back = {}
            for t in chunk:
                back[t] = t
            for p in q.get("pages", []):
                src = p["title"]
                orig = src
                while orig in alias:
                    orig = alias[orig]
                if p.get("missing") or not p.get("revisions"):
                    self.revs[orig] = None
                    continue
                rv = p["revisions"][0]
                if rv["timestamp"] >= CUTOFF:   # never use a revision from after the cutoff
                    raise RuntimeError(f"{src}: latest revision {rv['timestamp']} is after the cutoff")
                self.revs[orig] = dict(revid=rv["revid"], timestamp=rv["timestamp"], title=src)
            for t in chunk:
                self.revs.setdefault(t, None)

    def parse(self, title: str) -> dict | None:
        f = self.cache / f"{self.slug(title)}.json"
        if self.offline:
            if not f.exists():
                return None
            out = json.loads(f.read_text())
            self.pages.append(dict(title=title, revid=out["revid"], rev_timestamp=None, fetched=out["fetched"]))
            return out
        if title not in self.revs:
            self.resolve([title])
        rv = self.revs.get(title)
        if rv is None:
            print(f"  missing page: {title}", file=sys.stderr)
            return None
        if f.exists():
            out = json.loads(f.read_text())
            if out.get("revid") == rv["revid"]:
                self.pages.append(dict(title=title, revid=out["revid"], rev_timestamp=rv["timestamp"],
                                       fetched=out["fetched"]))
                return out
        d = self._get(dict(action="parse", oldid=rv["revid"], prop="text|revid|displaytitle",
                           disableeditsection=1, disabletoc=1))
        if "error" in d:
            print(f"  parse error: {title} ({d['error'].get('code')})", file=sys.stderr)
            return None
        out = {"title": d["parse"]["title"], "revid": d["parse"]["revid"], "rev_timestamp": rv["timestamp"],
               "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "text": d["parse"]["text"]}
        f.write_text(json.dumps(out))
        self.pages.append(dict(title=title, revid=out["revid"], rev_timestamp=rv["timestamp"],
                               fetched=out["fetched"]))
        print(f"  [{self.n_req} req] {title} rev {out['revid']} @ {rv['timestamp']}", flush=True)
        return out


def load_kalshi_nov1() -> dict:
    k = pd.read_csv(KALSHI_NOV1)
    out = {}
    for _, r in k.iterrows():
        cands = []
        for part in str(r["candidates"]).split("|"):
            bits = part.strip().split(":")
            if len(bits) < 3:
                continue
            party, name, price = bits[0], ":".join(bits[1:-1]), bits[-1]
            try:
                price = float(price)
            except ValueError:
                price = float("nan")
            placeholder = bool(re.fullmatch(r"(Democratic|Republican|Independent) party", name.strip(), re.I))
            cands.append(dict(party=party, name=name.strip(), price=price, placeholder=placeholder))
        cands.sort(key=lambda c: -(c["price"] if c["price"] == c["price"] else 0))
        out[r["race_id"]] = cands
    return out


def competitive_states(kalshi: dict) -> list[str]:
    """States with a House race whose Nov 1 Kalshi favourite is below 90%, most such races first."""
    cnt = {}
    for race, kc in kalshi.items():
        if race.endswith("-SEN") or not kc:
            continue
        tot = sum(c["price"] for c in kc if c["price"] == c["price"]) or 1
        if kc[0]["price"] / tot < 0.90:
            st = race.split("-")[0]
            cnt[st] = cnt.get(st, 0) + 1
    return sorted(cnt, key=lambda s: -cnt[s])


def house_titles_offline(cache: Path):
    titles, extra = [], []
    for f in sorted(cache.glob("2026_*.json")):
        t = json.loads(f.read_text())["title"]
        m = re.match(r"2026 United States House of Representatives elections? in (?:the )?(.+?)(?: \(districts.*\))?$", t)
        if m and F.NAME2ABBR.get(m.group(1)):
            titles.append((F.NAME2ABBR[m.group(1)], t, " election in " in t))
            continue
        m = re.match(r"2026 (.+?)'s (\d+)(?:st|nd|rd|th) congressional district election$", t)
        if m and m.group(1) in F.NAME2ABBR:
            extra.append((f"{F.NAME2ABBR[m.group(1)]}-{int(m.group(2)):02d}", t))
    return sorted(titles, key=lambda x: x[1]), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", help="re-parse this cache directory (no network)")
    ap.add_argument("--tag", required=True, help="output folder name under --out-root")
    ap.add_argument("--days-national", type=int, default=4)
    ap.add_argument("--gap", type=float, default=MIN_GAP, help="seconds between requests (>= 5)")
    ap.add_argument("--prior-limited", type=int, default=0, help="rate-limit responses already received today")
    ap.add_argument("--out-root", default=str(HERE / "data" / "pulls"),
                    help="parent folder of <tag>/ (pau-y, Nov 3: final_2026.py writes under pbase/data/pulls)")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.offline) if args.offline else out_dir / "cache"
    wiki = Wiki(cache, offline=bool(args.offline))
    wiki.gap = max(MIN_GAP, args.gap)
    wiki.n_limited = args.prior_limited
    F.RAW = cache   # generic-ballot revision cache goes here too, never into the shared cache
    kalshi = load_kalshi_nov1()
    valid = set(pd.read_csv(SHARED / "templates" / "full_congress_template.csv")["race_id"])
    rows, aggs, generic, log = [], [], [], []
    stopped = None
    t0 = dt.datetime.now(dt.timezone.utc)
    assert t0.isoformat() < CUTOFF.replace("Z", "+00:00"), "past the cutoff"

    sen = [(r, t) for r, t in F.senate_pages() if r in valid]
    try:
        if not args.offline:
            wiki.resolve([t for _, t in sen])
        for race, title in sen:
            page = wiki.parse(title)
            if page is None:
                log.append(dict(race=race, title=title, rows=None))
                continue
            n0 = len(rows)
            F.extract_race_polls(page, lambda path, r=race: r, kalshi, rows, aggs, generic)
            log.append(dict(race=race, title=title, rows=len(rows) - n0))

        if args.offline:
            titles, extra = house_titles_offline(cache)
        else:
            titles, extra = F.house_pages(wiki)
            pri = {s: i for i, s in enumerate(competitive_states(kalshi))}
            titles = sorted(titles, key=lambda x: (pri.get(x[0], 99), x[1]))
            wiki.resolve([t for _, t, _ in titles])
        done_extra = set()
        for st, title, at_large in titles:
            page = wiki.parse(title)
            if page is None:
                continue
            n0 = len(rows)
            F.extract_race_polls(page, lambda path, s=st, al=at_large: F.race_from_house_path(s, path, al),
                                 kalshi, rows, aggs, generic)
            log.append(dict(race=st + " House", title=title, rows=len(rows) - n0))
            for race, t in F.main_article_links(page):
                extra.append((race, t))
        extra = [(r, t) for r, t in extra if r in valid]
        if not args.offline:
            wiki.resolve([t for _, t in extra])
        for race, t in extra:
            if t in done_extra:
                continue
            done_extra.add(t)
            page = wiki.parse(t)
            if page is None:
                continue
            n0 = len(rows)
            F.extract_race_polls(page, lambda path, r=race: r, kalshi, rows, aggs, generic)
            log.append(dict(race=race, title=t, rows=len(rows) - n0))
    except RateLimited as e:
        stopped = str(e)
        print("STOPPED:", e, file=sys.stderr)

    allv = pd.DataFrame(rows)
    allv.drop(columns=["_tbl", "_rowkey"], errors="ignore").to_csv(out_dir / "polls_2026_all_versions.csv",
                                                                   index=False)
    polls = F.dedupe(allv)
    polls["source"] = "wikipedia"
    cols = F.POLL_COLS + ["same_party", "dates_raw", "source"]
    polls = polls[[c for c in cols if c in polls.columns]].sort_values(["race_id", "end_date"],
                                                                       ascending=[True, False])
    polls.to_csv(out_dir / "polls_2026.csv", index=False)
    pd.DataFrame(aggs).drop_duplicates().to_csv(out_dir / "race_poll_aggregates_2026.csv", index=False)
    g = pd.DataFrame(generic)
    if not g.empty:
        g = g.drop_duplicates(["race_id", "pollster", "end_date"]).sort_values(["race_id", "end_date"])
    g.to_csv(out_dir / "polls_2026_district_generic.csv", index=False)
    print(f"polls_2026.csv: {len(polls)} polls, {polls.race_id.nunique()} races ({len(allv)} version rows)")

    if not args.offline and stopped is None:
        try:
            # national: generic-ballot aggregate snapshots of the last few days (rvstart = now < cutoff) and the
            # Trump approval list
            nat = F.generic_ballot_snapshots(wiki, args.days_national) + F.approval_polls(wiki)
            nat = pd.DataFrame(nat)
            ncols = ["metric", "row_type", "pollster", "start_date", "end_date", "as_of", "sample_size",
                     "population", "dem_pct", "rep_pct", "other_pct", "approve", "disapprove", "partisan_sponsor",
                     "source_url", "wiki_revision_date", "wiki_page", "wiki_section"]
            nat[[c for c in ncols if c in nat.columns]].to_csv(out_dir / "national_2026.csv", index=False)
        except RateLimited as e:
            stopped = "national: " + str(e)

    pages = pd.DataFrame(wiki.pages)
    lg = pd.DataFrame(log)
    if len(pages):
        pages = pages.merge(lg[["title", "rows"]], on="title", how="left")
    pages.to_csv(out_dir / "pages.csv", index=False)
    meta = dict(started_utc=t0.isoformat(timespec="seconds"),
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                cutoff=CUTOFF, user_agent=UA, gap_s=wiki.gap, requests=wiki.n_req,
                rate_limited=wiki.n_limited, stopped=stopped, kalshi_names=str(KALSHI_NOV1),
                offline=args.offline)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
