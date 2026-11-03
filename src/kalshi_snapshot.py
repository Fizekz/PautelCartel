"""Snapshot Kalshi's 2026 House and Senate winner markets and map them to the organizers' race_ids.

Usage:
    python src/kalshi_snapshot.py                 # pull live, write data/kalshi/
    python src/kalshi_snapshot.py --raw FILE.json # rebuild the tidy CSV from a saved raw pull

Outputs (data/kalshi/):
    raw_<stamp>.json        every open event (with nested markets) whose ticker looks like a House/Senate race
    markets_<stamp>.csv     one row per market (candidate/party) mapped to race_id
    races_<stamp>.csv       one row per template race: favourite, its probability, coverage flags
Public market-data endpoints only; no account or key.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

API = "https://api.elections.kalshi.com/trade-api/v2/events"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "kalshi"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
PROB_FLOOR = 0.01

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
HOUSE_TITLE = re.compile(r"^([A-Z]{2})-(\d{1,2}|AL)( House)? winner\?( \(Person\))?$")
AT_LARGE_TITLE = re.compile(r"^(.+?) House winner\?$")
SENATE_TITLE = re.compile(r"^(.+?) Senate winner\?( \(Person\))?$")
RACE_TICKER = re.compile(r"^(KX)?(HOUSE|SENATE)")
RACE_TITLE = re.compile(r"(House|Senate) winner\?|^[A-Z]{2}-\d{1,2} winner\?")


def fetch_open_events() -> list[dict]:
    events, cursor = [], ""
    while True:
        url = f"{API}?status=open&limit=200&with_nested_markets=true" + (f"&cursor={cursor}" if cursor else "")
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        events += page.get("events", [])
        cursor = page.get("cursor") or ""
        if not cursor or not page.get("events"):
            return events


def race_id_for(event: dict) -> str | None:
    """Map an event to a template race_id from its title (tickers are unreliable: SENATELA-26 is Kentucky)."""
    title = event.get("title", "").strip()
    if m := HOUSE_TITLE.match(title):
        state, dist = m.group(1), m.group(2)
        return f"{state}-AL" if dist == "AL" else f"{state}-{int(dist):02d}"
    if (m := SENATE_TITLE.match(title)) and "(20" not in title:
        state = STATES.get(m.group(1).strip())
        return f"{state}-SEN" if state else None
    if m := AT_LARGE_TITLE.match(title):
        state = STATES.get(m.group(1).strip())
        return f"{state}-AL" if state else None
    return None


def is_person_event(event: dict) -> bool:
    return "(Person)" in event.get("title", "") or "PERSON" in event["event_ticker"]


def is_race_event(event: dict) -> bool:
    return bool(RACE_TICKER.match(event["event_ticker"]) or RACE_TITLE.search(event.get("title", "")))


def party_of(market: dict, person_event: bool = False) -> str:
    """DEM/REP/IND from a party market; UNK for person markets (same-party top-two races), joined later."""
    if person_event:
        return "UNK"
    suffix = market["ticker"].rsplit("-", 1)[-1]
    rules = market.get("rules_primary", "")
    if suffix == "D" or "Democratic Party" in rules or "Democratic party" in rules:
        return "DEM"
    if suffix == "R" or "Republican Party" in rules or "Republican party" in rules:
        return "REP"
    return "IND"


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def market_rows(events: list[dict], pulled_at: str) -> list[dict]:
    rows = []
    for ev in events:
        race_id = race_id_for(ev)
        if not race_id:
            continue
        for mk in ev.get("markets", []):
            bid, ask, last = (to_float(mk.get(k)) for k in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"))
            mid = (bid + ask) / 2 if bid is not None and ask is not None and ask > 0 else None
            rows.append({
                "race_id": race_id, "chamber": "Senate" if race_id.endswith("-SEN") else "House",
                "event_ticker": ev["event_ticker"], "event_kind": "person" if is_person_event(ev) else "party",
                "market_ticker": mk["ticker"], "party": party_of(mk, is_person_event(ev) or ev["event_ticker"].startswith("KXCAELECTION")),
                "candidate": mk.get("yes_sub_title", ""), "yes_bid": bid, "yes_ask": ask, "last_price": last,
                "mid": mid, "volume": to_float(mk.get("volume_fp")), "open_interest": to_float(mk.get("open_interest_fp")),
                "status": mk.get("status"), "pulled_at": pulled_at,
            })
    return rows


def race_rows(markets: list[dict], template_ids: list[str]) -> list[dict]:
    """One row per template race. Uses the event with most volume when several events cover one race."""
    by_race: dict[str, dict[str, list[dict]]] = {}
    for r in markets:
        by_race.setdefault(r["race_id"], {}).setdefault(r["event_ticker"], []).append(r)
    rows = []
    for rid in template_ids:
        events = by_race.get(rid, {})
        if not events:
            rows.append({"race_id": rid, "covered": 0, "n_events": 0})
            continue
        # Party-level events are what the template scores (predicted_winner_party); person events are a fallback.
        party_events = [e for e in events if events[e][0]["event_kind"] == "party"] or list(events)
        best_ev = max(party_events, key=lambda e: sum(m["volume"] or 0 for m in events[e]))
        mks = events[best_ev]
        # Probability per market: mid if quoted, else last trade. Normalised to sum 1 within the race.
        raw = {m["market_ticker"]: (m["mid"] if m["mid"] is not None else m["last_price"]) or 0.0 for m in mks}
        total = sum(raw.values()) or 1.0
        fav = max(mks, key=lambda m: raw[m["market_ticker"]])
        ordered = sorted(mks, key=lambda m: -raw[m["market_ticker"]])
        rows.append({
            "race_id": rid, "covered": 1, "n_events": len(events), "event_ticker": best_ev,
            "other_events": " ".join(e for e in events if e != best_ev),
            "fav_party": fav["party"], "fav_candidate": fav["candidate"],
            "fav_mid": raw[fav["market_ticker"]], "fav_norm": raw[fav["market_ticker"]] / total,
            "fav_last": fav["last_price"],
            "candidates": " | ".join(f"{m['party']}:{m['candidate']}:{raw[m['market_ticker']]:.3f}" for m in ordered),
            "event_volume": sum(m["volume"] or 0 for m in mks),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, help="rebuild from a saved raw JSON instead of pulling live")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.raw:
        payload = json.loads(args.raw.read_text())
        events, pulled_at = payload["events"], payload["pulled_at"]
    else:
        pulled_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = [e for e in fetch_open_events() if is_race_event(e)]
        raw_path = OUT_DIR / f"raw_{pulled_at.replace(':', '')}.json"
        raw_path.write_text(json.dumps({"pulled_at": pulled_at, "source": API, "events": events}))
        print(f"raw: {raw_path.relative_to(ROOT)} ({len(events)} events)")

    stamp = pulled_at.replace(":", "")
    with TEMPLATE.open() as f:
        template_ids = [r["race_id"] for r in csv.DictReader(f)]
    markets = market_rows(events, pulled_at)
    races = race_rows(markets, template_ids)
    write_csv(OUT_DIR / f"markets_{stamp}.csv", markets)
    write_csv(OUT_DIR / f"races_{stamp}.csv", races)

    covered = [r for r in races if r["covered"]]
    missing = [r["race_id"] for r in races if not r["covered"]]
    extra = sorted({m["race_id"] for m in markets} - set(template_ids))
    print(f"covered {len(covered)}/{len(template_ids)} template races; missing: {missing}")
    if extra:
        print(f"mapped but not in template: {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
