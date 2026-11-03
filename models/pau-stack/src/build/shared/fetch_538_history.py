#!/usr/bin/env python3
"""Fetch FiveThirtyEight historical polls (2018, 2022) and final 538 forecasts.

538 closed in March 2025; projects.fivethirtyeight.com now redirects to ABC News.
All files are therefore pulled from the Internet Archive (web.archive.org, raw
"id_" snapshots). For each file the script lists the archived captures with the
CDX API, tries them newest-first, and keeps the first one that is a real CSV
(some late captures are ABC News redirect pages). Files are cached in
data/raw/polls/538/ and the snapshot actually used is logged in SOURCES.txt.

Outputs
  data/features/polls_history.csv       2018 + 2022 House/Senate general-election polls,
                                        one row per (poll, race)
  data/features/national_history.csv    2018 + 2022 generic-ballot and presidential-approval polls
  data/features/forecast538_history.csv final pre-election 538 race forecasts (deluxe; classic/lite
                                        probabilities as extra columns)

race_id convention (matches data/templates/full_congress_template.csv):
  Senate: "AZ-SEN"; a special election for the state's other seat held the same day is "MN-SEN-S"
          (2018 MN, 2018 MS; 2022 OK, 2022 CA partial-term special).
  House:  "KS-01"; single-district states "WY-AL".

Run:  python3 src/fetch_538_history.py [--refresh]
Needs pandas + requests.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "polls" / "538"
FEAT = ROOT / "data" / "features"
UA = "ModelCitizensResearch/1.0 (Stanford student election-forecast project)"

YEARS = (2018, 2022)
ELECTION_DAY = {2018: pd.Timestamp("2018-11-06"), 2022: pd.Timestamp("2022-11-08")}
REGULAR_SENATE_CLASS = {2018: "Class I", 2022: "Class III"}
# 538 poll race_ids that are same-day specials sharing a class with the regular race
SPECIAL_SENATE_RACE_IDS = {9480}  # 2022 California partial-term special (Padilla)

P538 = "https://projects.fivethirtyeight.com"
# local name -> (original URL, required?, newest capture to consider)
SOURCES = {
    "senate_polls_historical.csv": (f"{P538}/polls-page/data/senate_polls_historical.csv", True, None),
    "house_polls_historical.csv": (f"{P538}/polls-page/data/house_polls_historical.csv", True, None),
    "generic_ballot_polls_historical.csv": (
        f"{P538}/polls-page/data/generic_ballot_polls_historical.csv", False, None),
    "president_approval_polls_historical.csv": (
        f"{P538}/polls-page/data/president_approval_polls_historical.csv", False, None),
    # the polls-page "current" approval file as of Jan 2025 = Biden's full term
    "president_approval_polls_biden.csv": (
        f"{P538}/polls-page/data/president_approval_polls.csv", False, "20250119"),
    "house_district_forecast_2018.csv": (f"{P538}/congress-model-2018/house_district_forecast.csv", True, None),
    "senate_seat_forecast_2018.csv": (f"{P538}/congress-model-2018/senate_seat_forecast.csv", True, None),
    "house_district_toplines_2022.csv": (
        f"{P538}/2022-general-election-forecast-data/house_district_toplines_2022.csv", True, None),
    "senate_state_toplines_2022.csv": (
        f"{P538}/2022-general-election-forecast-data/senate_state_toplines_2022.csv", True, None),
}

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
# states with a single House seat, by apportionment
AT_LARGE = {
    2018: {"AK", "DE", "MT", "ND", "SD", "VT", "WY"},
    2022: {"AK", "DE", "ND", "SD", "VT", "WY"},
}
POP_RANK = {"lv": 0, "rv": 1, "v": 2, "a": 3}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA


# --------------------------------------------------------------------------- download
def _get(url: str, tries: int = 5, timeout: int = 300) -> requests.Response:
    delay = 3.0
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code == 404:
                break
        except requests.RequestException as e:
            last = type(e).__name__
        print(f"  {last} for {url}; retry {i + 1}/{tries}", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"failed ({last}): {url}")


def _captures(url: str, before: str | None) -> list[str]:
    """Timestamps of archived 200/csv captures, newest first."""
    cdx = f"https://web.archive.org/cdx/search/cdx?url={url}&fl=timestamp,statuscode,mimetype&filter=statuscode:200"
    try:
        txt = _get(cdx, tries=3, timeout=120).text
    except RuntimeError as e:
        print(f"  CDX failed: {e}", file=sys.stderr)
        txt = ""
    ts = [ln.split()[0] for ln in txt.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    if before:
        ts = [t for t in ts if t <= before + "999999"]
    ts = sorted(set(ts), reverse=True)
    if not ts:  # fall back to the availability API / redirect
        try:
            js = _get(f"https://archive.org/wayback/available?url={url}"
                      + (f"&timestamp={before}" if before else ""), tries=3, timeout=60).json()
            snap = js.get("archived_snapshots", {}).get("closest") or {}
            if snap.get("timestamp"):
                ts = [snap["timestamp"]]
        except Exception as e:  # noqa: BLE001
            print(f"  availability API failed: {e}", file=sys.stderr)
    return ts or [before or "2025"]


def _looks_like_csv(content: bytes) -> bool:
    head = content[:2000].lstrip().lower()
    return bool(head) and not head.startswith((b"<", b"{")) and b"," in head.split(b"\n", 1)[0]


def download_all(refresh: bool = False) -> dict[str, str]:
    RAW.mkdir(parents=True, exist_ok=True)
    log = RAW / "SOURCES.txt"
    used: dict[str, str] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                used[k] = v
    for name, (url, required, before) in SOURCES.items():
        dest = RAW / name
        if dest.exists() and not refresh and _looks_like_csv(dest.read_bytes()[:2000]):
            used.setdefault(name, url)
            continue
        ok = False
        for ts in _captures(url, before)[:6]:
            wb = f"https://web.archive.org/web/{ts}id_/{url}"
            print(f"downloading {name} <- {wb}")
            try:
                r = _get(wb)
            except RuntimeError as e:
                print(f"  {e}", file=sys.stderr)
                continue
            if not _looks_like_csv(r.content):
                print("  capture is not a CSV (redirect page?), trying an older one", file=sys.stderr)
                continue
            dest.write_bytes(r.content)
            used[name] = wb
            ok = True
            break
        if not ok:
            msg = f"could not fetch {name}"
            if required:
                raise RuntimeError(msg)
            print("WARNING: " + msg, file=sys.stderr)
        log.write_text("".join(f"{k}\t{v}\n" for k, v in used.items()))
        time.sleep(2)
    return used


# --------------------------------------------------------------------------- polls
def _poll_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], format="mixed")
    df["end_date"] = pd.to_datetime(df["end_date"], format="mixed")
    df["election_date"] = pd.to_datetime(df["election_date"], format="mixed")
    df["population"] = df["population"].astype(str).str.lower()
    return df


def _race_id(row, chamber: str) -> str | None:
    st = STATE_ABBR.get(row.state)
    if st is None:
        return None
    if chamber == "Senate":
        special = (row.seat_name != REGULAR_SENATE_CLASS[row.cycle]) or (row.race_id in SPECIAL_SENATE_RACE_IDS)
        return f"{st}-SEN" + ("-S" if special else "")
    if st in AT_LARGE[row.cycle]:
        return f"{st}-AL"
    return f"{st}-{int(row.seat_number):02d}"


def _summarise_question(g: pd.DataFrame) -> dict:
    """One poll question (long format, one row per answer) -> wide D/R/other summary."""
    g = g.sort_values("pct", ascending=False)
    d = g[g.party == "DEM"]
    r = g[g.party == "REP"]
    o = g[~g.party.isin(["DEM", "REP"])]
    out = {
        "dem_candidate": d.candidate_name.iloc[0] if len(d) else None,
        "dem_pct": d.pct.iloc[0] if len(d) else None,
        "rep_candidate": r.candidate_name.iloc[0] if len(r) else None,
        "rep_pct": r.pct.iloc[0] if len(r) else None,
        "other_pct": o.pct.sum() if len(o) else None,
        "other_candidate": o.candidate_name.iloc[0] if len(o) else None,
        "other_party": o.party.iloc[0] if len(o) else None,
        "other_top_pct": o.pct.iloc[0] if len(o) else None,
        # second candidate of the same party (CA/WA top-two, LA jungle, AK top-four)
        "dem2_candidate": d.candidate_name.iloc[1] if len(d) > 1 else None,
        "dem2_pct": d.pct.iloc[1] if len(d) > 1 else None,
        "rep2_candidate": r.candidate_name.iloc[1] if len(r) > 1 else None,
        "rep2_pct": r.pct.iloc[1] if len(r) > 1 else None,
        "n_answers": len(g),
    }
    return out


def build_polls(chamber: str, path: Path) -> pd.DataFrame:
    raw = _poll_common(pd.read_csv(path, low_memory=False))
    raw = raw[raw.cycle.isin(YEARS)]
    # keep the November general election (incl. CA/WA top-two, LA/MS jungle primaries held that day)
    # and the Senate runoffs that decided the seat (GA 2022, MS-special 2018), flagged in `stage`
    gday = raw.cycle.map(lambda y: ELECTION_DAY[y])
    keep = (raw.election_date == gday) & raw.stage.isin(["general", "jungle primary"])
    if chamber == "Senate":
        keep |= raw.stage.eq("runoff") & raw.election_date.gt(gday) & raw.election_date.lt(gday + pd.Timedelta(days=40))
    raw = raw[keep].copy()
    # drop same-day House specials that duplicate the regular race (e.g. NY-25 2018): keep min 538 race_id
    if chamber == "House":
        regular = raw.groupby(["cycle", "state", "seat_number"]).race_id.transform("min")
        raw = raw[raw.race_id == regular]

    rows = []
    for qid, g in raw.groupby("question_id", sort=False):
        f = g.iloc[0]
        s = _summarise_question(g)
        s.update(
            year=int(f.cycle), race_id=_race_id(f, chamber), chamber=chamber, poll_id=int(f.poll_id),
            question_id=int(qid), pollster=f.pollster, pollster_display=f.display_name,
            sponsors=f.sponsors, start_date=f.start_date.date(), end_date=f.end_date.date(),
            sample_size=f.sample_size, population=f.population,
            partisan_sponsor=f.partisan if isinstance(f.partisan, str) else None,
            internal=bool(f.internal) if pd.notna(f.internal) else None,
            sponsor_candidate_party=f.sponsor_candidate_party,
            hypothetical=bool(f.hypothetical), ranked_choice_reallocated=bool(f.ranked_choice_reallocated),
            ranked_choice_round=f.ranked_choice_round,
            stage=f.stage, election_date=f.election_date.date(), methodology=f.methodology,
            numeric_grade=f.numeric_grade, pollscore=f.pollscore, tracking=f.tracking,
            source_url=f.url, src_538_race_id=int(f.race_id),
        )
        rows.append(s)
    q = pd.DataFrame(rows)
    q = q[q.race_id.notna()]
    n_questions = len(q)

    # one row per (poll, race, stage): prefer the real matchup over hypotheticals, likely voters, the
    # question naming both a D and an R (or, in same-party races, the most candidates), first-round
    # (not RCV-reallocated) numbers, then the largest sample
    q["_hyp"] = q.hypothetical.astype(int)
    q["_pop"] = q.population.map(POP_RANK).fillna(9)
    q["_both"] = -(q.dem_pct.notna() & q.rep_pct.notna()).astype(int)
    q["_rcv"] = q.ranked_choice_reallocated.astype(int)
    q["_n"] = -q.sample_size.fillna(0)
    q["_nans"] = -q.n_answers
    q = q.sort_values(["poll_id", "race_id", "stage", "_hyp", "_pop", "_both", "_rcv", "_n", "_nans", "question_id"])
    q["n_questions_in_poll"] = q.groupby(["poll_id", "race_id", "stage"]).question_id.transform("size")
    q = q.drop_duplicates(["poll_id", "race_id", "stage"], keep="first")
    q = q.drop(columns=[c for c in q.columns if c.startswith("_")])
    print(f"{chamber}: {n_questions} questions -> {len(q)} poll rows")
    return q


POLL_COLS = [
    "year", "race_id", "chamber", "pollster", "start_date", "end_date", "sample_size", "population",
    "dem_pct", "rep_pct", "other_pct", "dem_candidate", "rep_candidate", "partisan_sponsor", "source_url",
    "other_candidate", "other_party", "other_top_pct", "dem2_candidate", "dem2_pct", "rep2_candidate",
    "rep2_pct", "stage", "election_date", "hypothetical", "ranked_choice_reallocated", "internal",
    "sponsor_candidate_party", "sponsors", "pollster_display", "methodology", "numeric_grade", "pollscore",
    "tracking", "poll_id", "question_id", "n_questions_in_poll", "src_538_race_id",
]


# --------------------------------------------------------------------------- national
def build_national() -> pd.DataFrame:
    out = []
    gb_path = RAW / "generic_ballot_polls_historical.csv"
    if gb_path.exists():
        g = _poll_common(pd.read_csv(gb_path, low_memory=False))
        g = g[g.cycle.isin(YEARS)].copy()
        g["_pop"] = g.population.map(POP_RANK).fillna(9)
        g["_n"] = -g.sample_size.fillna(0)
        g = g.sort_values(["poll_id", "_pop", "_n", "question_id"]).drop_duplicates("poll_id")
        out.append(pd.DataFrame({
            "year": g.cycle, "measure": "generic_ballot", "pollster": g.pollster,
            "start_date": g.start_date.dt.date, "end_date": g.end_date.dt.date, "sample_size": g.sample_size,
            "population": g.population, "dem_pct": g.dem, "rep_pct": g.rep, "other_pct": g.ind,
            "approve_pct": None, "disapprove_pct": None, "president": None,
            "partisan_sponsor": g.partisan, "source_url": g.url, "tracking": g.tracking,
            "numeric_grade": g.numeric_grade, "poll_id": g.poll_id,
        }))

    def approval_rows(a: pd.DataFrame, president: str, year: int) -> pd.DataFrame:
        a = a.copy()
        a["_pop"] = a.population.map(POP_RANK).fillna(9)
        a["_n"] = -a.sample_size.fillna(0)
        a = a.sort_values(["poll_id", "_pop", "_n"]).drop_duplicates("poll_id")
        return pd.DataFrame({
            "year": year, "measure": "approval", "pollster": a.pollster,
            "start_date": a.start_date.dt.date, "end_date": a.end_date.dt.date, "sample_size": a.sample_size,
            "population": a.population, "dem_pct": None, "rep_pct": None, "other_pct": None,
            "approve_pct": a.approve, "disapprove_pct": a.disapprove, "president": president,
            "partisan_sponsor": a.get("partisan"), "source_url": a.url, "tracking": a.get("tracking"),
            "numeric_grade": a.get("numeric_grade"), "poll_id": a.poll_id,
        })

    # presidential approval, polls-page format (wide: yes = approve, no = disapprove).
    # president_approval_polls_historical.csv = Trump 2017-21; president_approval_polls.csv (Jan 2025
    # capture) = Biden 2021-25. Window: the 12 months before each election.
    for fname, who, year in (("president_approval_polls_historical.csv", "Trump", 2018),
                             ("president_approval_polls_biden.csv", "Biden", 2022)):
        path = RAW / fname
        if not path.exists():
            continue
        a = pd.read_csv(path, low_memory=False)
        if not {"politician", "yes", "no"} <= set(a.columns):
            print(f"WARNING: unexpected columns in {fname}", file=sys.stderr)
            continue
        a = a[a.politician.astype(str).str.contains(who)]
        a = a.rename(columns={"yes": "approve", "no": "disapprove"})
        a = _poll_common(a.assign(election_date=a.end_date))
        hi = ELECTION_DAY[year]
        a = a[(a.end_date > hi - pd.Timedelta(days=365)) & (a.end_date <= hi)]
        out.append(approval_rows(a, who, year))
    if not out:
        return pd.DataFrame()
    nat = pd.concat([o.dropna(axis=1, how="all") for o in out], ignore_index=True)
    return nat.sort_values(["year", "measure", "end_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- forecasts
def _forecast_2018(path: Path, chamber: str) -> pd.DataFrame:
    f = pd.read_csv(path, low_memory=False)
    f["forecastdate"] = pd.to_datetime(f.forecastdate)
    f = f[f.forecastdate <= ELECTION_DAY[2018]]
    f = f[f.forecastdate == f.forecastdate.max()]
    if chamber == "Senate":
        f["special"] = f.special.fillna(False).astype(bool)
        f["race_id"] = f.state + "-SEN" + f.special.map({True: "-S", False: ""})
    else:
        f["race_id"] = [f"{s}-AL" if s in AT_LARGE[2018] else f"{s}-{int(d):02d}" for s, d in zip(f.state, f.district)]
    rows = []
    for (rid, model), g in f.groupby(["race_id", "model"]):
        g = g.sort_values("win_probability", ascending=False)
        d, r, o = g[g.party == "D"], g[g.party == "R"], g[~g.party.isin(["D", "R"])]
        rows.append({
            "year": 2018, "race_id": rid, "chamber": chamber, "model_variant": model,
            "forecast_date": g.forecastdate.iloc[0].date(),
            "dem_win_prob": d.win_probability.sum(), "rep_win_prob": r.win_probability.sum(),
            "other_win_prob": o.win_probability.sum(),
            "dem_candidate": d.candidate.iloc[0] if len(d) else None,
            "rep_candidate": r.candidate.iloc[0] if len(r) else None,
            "other_candidate": o.candidate.iloc[0] if len(o) else None,
            "other_party": o.party.iloc[0] if len(o) else None,
            "dem_voteshare": d.voteshare.iloc[0] if len(d) else None,
            "rep_voteshare": r.voteshare.iloc[0] if len(r) else None,
            "other_voteshare": o.voteshare.iloc[0] if len(o) else None,
            "fav_candidate": g.candidate.iloc[0], "fav_party": g.party.iloc[0],
            "fav_win_prob": g.win_probability.iloc[0], "n_candidates": len(g),
            "dem_incumbent": bool(d.incumbent.iloc[0]) if len(d) else None,
            "rep_incumbent": bool(r.incumbent.iloc[0]) if len(r) else None,
        })
    return pd.DataFrame(rows)


def _forecast_2022(path: Path, chamber: str) -> pd.DataFrame:
    f = pd.read_csv(path, low_memory=False)
    f["forecastdate"] = pd.to_datetime(f.forecastdate, format="%m/%d/%y")
    f = f[f.forecastdate <= ELECTION_DAY[2022]]
    f = f[f.forecastdate == f.forecastdate.max()]
    if "timestamp" in f.columns:  # several runs per day: keep the last
        f["_ts"] = pd.to_datetime(f.timestamp, format="%H:%M:%S %d %b %Y", errors="coerce")
        f = f.sort_values("_ts").drop_duplicates(["district", "expression"], keep="last")
    f["model_variant"] = f.expression.str.strip("_")
    rows = []
    for _, x in f.iterrows():
        dist = str(x.district)
        st, num = dist.split("-", 1)
        if chamber == "Senate":
            # district codes are "<ST>-S<class>"; 2022's regular seats are Class III, so OK-S2 is the special
            rid = f"{st}-SEN" + ("" if num.upper() == "S3" else "-S")
        else:
            rid = f"{st}-AL" if st in AT_LARGE[2022] else f"{st}-{int(num):02d}"
        cands = []
        for k in ["D1", "D2", "D3", "D4", "R1", "R2", "R3", "R4", "I1", "O1"]:
            nm = x.get(f"name_{k}")
            p = pd.to_numeric(x.get(f"winner_{k}"), errors="coerce")
            if isinstance(nm, str) and nm.strip():
                cands.append((k, nm, 0.0 if pd.isna(p) else float(p),
                              pd.to_numeric(x.get(f"voteshare_mean_{k}"), errors="coerce")))
        cands.sort(key=lambda c: -c[2])
        d = [c for c in cands if c[0].startswith("D")]
        r = [c for c in cands if c[0].startswith("R")]
        o = [c for c in cands if c[0][0] in "IO"]
        # party probabilities from candidate-level win probs (538's winner_Dparty folds independents such
        # as 2022 UT McMullin into the D column); fall back to the party fields if candidates don't sum to 1
        tot = sum(c[2] for c in cands)
        if abs(tot - 1) < 0.01:
            dem_p, rep_p = sum(c[2] for c in d) / tot, sum(c[2] for c in r) / tot
        else:
            dem_p = float(pd.to_numeric(x.get("winner_Dparty"), errors="coerce"))
            rep_p = float(pd.to_numeric(x.get("winner_Rparty"), errors="coerce"))
        rows.append({
            "year": 2022, "race_id": rid, "chamber": chamber, "model_variant": x.model_variant,
            "forecast_date": x.forecastdate.date(), "src_district": dist,
            "dem_win_prob": dem_p, "rep_win_prob": rep_p,
            "other_win_prob": max(0.0, 1 - dem_p - rep_p),
            "dem_candidate": d[0][1] if d else None, "rep_candidate": r[0][1] if r else None,
            "other_candidate": o[0][1] if o else None, "other_party": ("I" if o[0][0] == "I1" else "O") if o else None,
            "dem_voteshare": d[0][3] if d else None, "rep_voteshare": r[0][3] if r else None,
            "other_voteshare": o[0][3] if o else None,
            "fav_candidate": cands[0][1] if cands else None,
            "fav_party": cands[0][0][0] if cands else None,
            "fav_win_prob": cands[0][2] if cands else None, "n_candidates": len(cands),
            "mean_netpartymargin": x.get("mean_netpartymargin"),
            "dem_party_win_prob_538": x.get("winner_Dparty"), "rep_party_win_prob_538": x.get("winner_Rparty"),
        })
    return pd.DataFrame(rows)


def build_forecasts() -> pd.DataFrame:
    parts = [
        _forecast_2018(RAW / "house_district_forecast_2018.csv", "House"),
        _forecast_2018(RAW / "senate_seat_forecast_2018.csv", "Senate"),
        _forecast_2022(RAW / "house_district_toplines_2022.csv", "House"),
        _forecast_2022(RAW / "senate_state_toplines_2022.csv", "Senate"),
    ]
    long = pd.concat(parts, ignore_index=True)
    # one row per race: deluxe (538's headline model) as the main columns; classic/lite alongside
    main = long[long.model_variant == "deluxe"].copy()
    for v in ("classic", "lite"):
        side = long[long.model_variant == v][["year", "race_id", "dem_win_prob", "rep_win_prob", "other_win_prob"]]
        side = side.rename(columns={c: f"{c}_{v}" for c in ("dem_win_prob", "rep_win_prob", "other_win_prob")})
        main = main.merge(side, on=["year", "race_id"], how="left")
    front = ["year", "race_id", "chamber", "dem_win_prob", "rep_win_prob", "other_win_prob", "forecast_date",
             "model_variant", "dem_candidate", "rep_candidate", "other_candidate", "other_party",
             "dem_voteshare", "rep_voteshare", "other_voteshare", "fav_candidate", "fav_party", "fav_win_prob"]
    main = main[front + [c for c in main.columns if c not in front]]
    return main.sort_values(["year", "chamber", "race_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    used = download_all(args.refresh)
    if args.download_only:
        return
    FEAT.mkdir(parents=True, exist_ok=True)

    polls = pd.concat([build_polls("Senate", RAW / "senate_polls_historical.csv"),
                       build_polls("House", RAW / "house_polls_historical.csv")], ignore_index=True)
    polls = polls[POLL_COLS].sort_values(["year", "chamber", "race_id", "end_date", "pollster"])
    polls.to_csv(FEAT / "polls_history.csv", index=False)
    print(f"wrote polls_history.csv: {len(polls)} rows")
    print(polls.groupby(["year", "chamber"]).agg(polls=("poll_id", "size"), races=("race_id", "nunique")))

    nat = build_national()
    if len(nat):
        nat.to_csv(FEAT / "national_history.csv", index=False)
        print(f"wrote national_history.csv: {len(nat)} rows")
        print(nat.groupby(["year", "measure"]).size())

    fc = build_forecasts()
    fc.to_csv(FEAT / "forecast538_history.csv", index=False)
    print(f"wrote forecast538_history.csv: {len(fc)} rows")
    print(fc.groupby(["year", "chamber", "model_variant", "forecast_date"]).size())
    print("sources:", *(f"  {k}: {v}" for k, v in used.items()), sep="\n")


if __name__ == "__main__":
    main()
