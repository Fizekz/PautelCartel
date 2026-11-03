"""Build expert-rating datasets (Cook, Sabato, Inside Elections + other forecasters) from Wikipedia.

Usage:
    .venv/bin/python src/data/fetch_ratings.py            # use cached raw pages, fetch only missing ones
    .venv/bin/python src/data/fetch_ratings.py --refresh  # re-download every page

Sources (cached as MediaWiki parse-API JSON under data/raw/ratings/):
    "<year> United States House of Representatives election ratings"  per-district rating table
    "<year> United States House of Representatives elections"         per-district PVI / incumbent party (imputation)
    "<year> United States Senate elections"                           "Predictions" rating table

Outputs:
    data/clean/ratings_2026.csv  one row per template race_id (470)
    data/clean/ratings_hist.csv  one row per race, 2010-2024 (House: all 435 seats; Senate: every race rated)

Scale: Safe D=-3, Likely D=-2, Lean D=-1, Tilt D=-0.5, Toss-up=0, Tilt R=0.5, Lean R=1, Likely R=2, Safe R=3.
House seats missing from Wikipedia's competitive table get Safe (+/-3) for cook/sabato/inside with imputed_safe=1.
RATING_OVERRIDES holds the few verifier corrections taken from a second source (URL recorded in imputed_basis).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ratings_parse import (CORE_RATERS, encode_rating, find_rating_table, house_districts,  # noqa: E402
                           house_race_id, parse_pvi, rating_rows, senate_label)
from wiki_fetch import fetch_page  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ratings"
CLEAN_DIR = ROOT / "data" / "clean"
TEMPLATE = ROOT / "data" / "templates" / "full_congress_template.csv"
HIST_YEARS = (2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024)
CURRENT_YEAR = 2026
SAFE = 3.0
DISTRICT_FIELDS = ("incumbent_party", "same_party_final", "n_cands_wiki")  # House only, from the elections page
PVI_OVERRIDE = 5  # an absent seat whose PVI disagrees with the incumbent party by >= this is imputed by PVI sign
# Verifier corrections taken from an independent second source, applied after imputation.
# (year, race_id) -> {rater: (raw_text, as_of, source_url)}
RATING_OVERRIDES: dict[tuple[int, str], dict[str, tuple[str, str, str]]] = {
    # Wikipedia's 2026 ratings table (Sabato column dated 2026-10-27) omits Louisiana's redrawn 5th district;
    # the Crystal Ball table mirrored by 270toWin, as of 2026-10-30, lists LA-5 (open) as Likely Republican.
    (2026, "LA-05"): {"sabato": ("Likely R", "2026-10-30",
                                 "https://www.270towin.com/2026-house-election/table/crystal-ball-2026-house-forecast")},
}


def page(title: str, refresh: bool) -> dict:
    result = fetch_page(title, RAW_DIR, refresh=refresh)
    if result is None:
        raise RuntimeError(f"Wikipedia page missing: {title}")
    return result


def rows_from_table(html: str, chamber: str, year: int, source: str) -> list[dict]:
    found = find_rating_table(html)
    if found is None:
        raise RuntimeError(f"{year} {chamber}: no rating table found")
    grid, h_idx, cols = found
    records = []
    for cells in rating_rows(grid, h_idx):
        rec = {"year": year, "chamber": chamber, "label": cells[0], "source_page": source,
               "pvi": parse_pvi(cells[1]) if len(cells) > 1 else None}
        if chamber == "House":
            rec["race_id"], rec["special"] = house_race_id(cells[0]), 0
        else:
            parsed = senate_label(cells[0])
            rec["state"], rec["kind"] = parsed if parsed else (None, None)
        if rec.get("race_id") is None and chamber == "House":
            continue
        if chamber == "Senate" and rec["state"] is None:
            continue
        for col in cols:
            raw = cells[col.index] if col.index < len(cells) else ""
            rec[f"{col.key}_raw"] = raw
            rec[col.key] = encode_rating(raw)
            rec[f"{col.key}_date"] = col.as_of
        records.append(rec)
    return records


def assign_senate_ids(records: list[dict]) -> list[dict]:
    """XX-SEN for the only race in a state; XX-SEN-S for a special held alongside a regular race."""
    by_state: dict[tuple[int, str], int] = {}
    for r in records:
        by_state[(r["year"], r["state"])] = by_state.get((r["year"], r["state"]), 0) + 1
    out = []
    for r in records:
        is_special = r["kind"] == "special"
        both = by_state[(r["year"], r["state"])] > 1
        rid = f"{r['state']}-SEN" + ("-S" if is_special and both else "")
        out.append({**r, "race_id": rid, "special": int(is_special)})
    return out


def safe_side(info: dict) -> tuple[float | None, str]:
    """Sign of the Safe rating for a seat absent from the competitive table, and the basis used."""
    party, pvi = info.get("incumbent_party", ""), info.get("pvi")
    same = {"R": 1.0, "D": -1.0}.get(info.get("same_party_final", ""))
    if same is not None:  # both finalists from one party (top-two / jungle lockout): that party is safe
        return same, "same_party_final"
    pvi_sign = None if pvi in (None, 0.0) else (1.0 if pvi > 0 else -1.0)
    inc_sign = {"R": 1.0, "D": -1.0}.get(party)
    if inc_sign is not None and (pvi_sign is None or pvi_sign == inc_sign or abs(pvi) < PVI_OVERRIDE):
        return inc_sign, "incumbent_party"
    if pvi_sign is not None:
        return pvi_sign, "pvi_sign"
    return None, "unknown"


def impute_house(rated: list[dict], districts: dict[str, dict], year: int, source: str) -> list[dict]:
    have = {r["race_id"] for r in rated}
    out = []
    for rid, info in sorted(districts.items()):
        if rid in have:
            continue
        sign, basis = safe_side(info)
        rec = {"year": year, "chamber": "House", "race_id": rid, "special": 0, "label": rid, "pvi": info["pvi"],
               "source_page": source, "imputed_basis": basis}
        for key in CORE_RATERS:
            rec[key] = None if sign is None else SAFE * sign
            rec[f"{key}_raw"] = "" if sign is None else f"Safe {'R' if sign > 0 else 'D'} (imputed)"
        out.append(rec)
    return out


def apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with RATING_OVERRIDES applied; the source is recorded in imputed_basis."""
    out = df.copy()
    for (year, rid), fixes in RATING_OVERRIDES.items():
        mask = (out["year"] == year) & (out["race_id"] == rid)
        if not mask.any():
            continue
        for key, (raw, as_of, url) in fixes.items():
            out.loc[mask, key] = encode_rating(raw)
            out.loc[mask, f"{key}_raw"] = raw
            out.loc[mask, f"{key}_date"] = as_of
            note = f"{key} override: {url}"
            prev = out.loc[mask, "imputed_basis"].fillna("").astype(str)
            out.loc[mask, "imputed_basis"] = [f"{p}; {note}" if p else note for p in prev]
    return out


def finalize(df: pd.DataFrame, rated_ids: set[tuple[int, str]]) -> pd.DataFrame:
    df = apply_overrides(df)
    core = df[list(CORE_RATERS)]
    out = df.assign(
        mean_rating=core.mean(axis=1, skipna=True).round(3),
        n_raters=core.notna().sum(axis=1),
        imputed_safe=[0 if (y, r) in rated_ids else 1 for y, r in zip(df["year"], df["race_id"])],
    )
    dates = df[[c for c in df.columns if c.endswith("_date") and c.split("_date")[0] in CORE_RATERS]]
    out["as_of"] = dates.replace("", pd.NA).max(axis=1) if not dates.empty else ""
    return out


def ordered_columns(df: pd.DataFrame, lead: list[str]) -> list[str]:
    """Lead keys, core raters (numeric + raw), other_* numeric ratings, summary, provenance, core dates."""
    core = [c for k in CORE_RATERS for c in (k, f"{k}_raw")]
    others = sorted(c for c in df.columns if c.startswith("other_") and not c.endswith(("_raw", "_date")))
    tail = ["mean_rating", "n_raters", "imputed_safe", "as_of", "imputed_basis", "pvi_wiki", *DISTRICT_FIELDS,
            "label",
            "source_page", "source_revid"] + [f"{k}_date" for k in CORE_RATERS]
    return [c for c in lead + core + others + tail if c in df.columns]


def build_year(year: int, refresh: bool) -> tuple[pd.DataFrame, set[tuple[int, str]]]:
    house_title = f"{year} United States House of Representatives election ratings"
    elec_title = f"{year} United States House of Representatives elections"
    sen_title = f"{year} United States Senate elections"
    hp, ep, sp = page(house_title, refresh), page(elec_title, refresh), page(sen_title, refresh)
    house = rows_from_table(hp["html"], "House", year, house_title)
    house = list({r["race_id"]: r for r in reversed(house)}.values())[::-1]  # keep first row per district
    senate = assign_senate_ids(rows_from_table(sp["html"], "Senate", year, sen_title))
    rated_ids = {(year, r["race_id"]) for r in house + senate}
    districts = house_districts(ep["html"])
    imputed = impute_house(house, districts, year, elec_title)
    revids = {house_title: hp["revid"], sen_title: sp["revid"], elec_title: ep["revid"]}
    df = pd.DataFrame(house + senate + imputed)
    df["source_revid"] = df["source_page"].map(revids)
    is_house = df["chamber"] == "House"
    for key in CORE_RATERS:  # imputed rows inherit the date of the table column they were imputed from
        col = f"{key}_date"
        if col in df.columns:
            table_date = next((r[col] for r in house if r.get(col)), "")
            df[col] = df[col].where(~(is_house & df[col].isna()), table_date)
    for key in DISTRICT_FIELDS:
        df[key] = df["race_id"].map(lambda rid, k=key: districts.get(rid, {}).get(k)).where(is_house)
    return df, rated_ids


def rename_others(df: pd.DataFrame) -> pd.DataFrame:
    fixed = set(CORE_RATERS) | {"year", "race_id", "chamber", "special", "label", "source_page", "pvi", "state",
                                "kind", "imputed_basis", "source_revid", "mean_rating", "n_raters",
                                "imputed_safe", "as_of", *DISTRICT_FIELDS}
    mapping = {c: f"other_{c}" for c in df.columns if c not in fixed and not c.endswith(("_raw", "_date"))}
    return df.rename(columns={**mapping, "pvi": "pvi_wiki"})


def build_current(refresh: bool) -> pd.DataFrame:
    df, rated = build_year(CURRENT_YEAR, refresh)
    df = finalize(df, rated)
    template = pd.read_csv(TEMPLATE, usecols=["race_id", "chamber"])
    merged = template.merge(df.drop(columns=["chamber"]), on="race_id", how="left", validate="one_to_one")
    missing = merged.loc[merged["year"].isna(), "race_id"].tolist()
    if missing:
        print(f"WARNING 2026 template races without ratings: {missing}", file=sys.stderr)
    extra = sorted(set(df["race_id"]) - set(template["race_id"]))
    if extra:
        print(f"WARNING 2026 rated races not in template: {extra}", file=sys.stderr)
    merged = rename_others(merged)
    return merged[ordered_columns(merged, ["race_id", "chamber"])]


def build_hist(refresh: bool) -> pd.DataFrame:
    frames, rated = [], set()
    for year in HIST_YEARS:
        df, ids = build_year(year, refresh)
        frames.append(df)
        rated |= ids
    df = finalize(pd.concat(frames, ignore_index=True), rated)
    df = rename_others(df)
    cols = ordered_columns(df, ["year", "race_id", "chamber", "special"])
    return df[cols].sort_values(["year", "chamber", "race_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="re-download every Wikipedia page")
    args = parser.parse_args()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cur = build_current(args.refresh)
    cur.to_csv(CLEAN_DIR / "ratings_2026.csv", index=False)
    hist = build_hist(args.refresh)
    hist.to_csv(CLEAN_DIR / "ratings_hist.csv", index=False)
    print(f"ratings_2026.csv rows={len(cur)} rated={int((cur['imputed_safe'] == 0).sum())}")
    print(hist.groupby(["year", "chamber"]).agg(rows=("race_id", "size"), imputed=("imputed_safe", "sum")))


if __name__ == "__main__":
    main()
