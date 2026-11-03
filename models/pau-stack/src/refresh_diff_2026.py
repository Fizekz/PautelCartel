"""pau-f: which polls are new in the refreshed 2026 file, plus sanity checks.

Usage: refresh_diff_2026.py NEW_DIR [OLD_POLLS_CSV]   (NEW_DIR = output folder of refresh_polls_2026.py;
OLD_POLLS_CSV defaults to the shared Nov 1 file; pau-y, Nov 3: final_2026.py passes the previous pull)
Writes NEW_DIR/new_polls.csv, NEW_DIR/changed_polls.csv, NEW_DIR/removed_polls.csv and prints the checks.
"""
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd

OLD = Path(__file__).resolve().parents[1] / "data" / "pulls" / "live_1102" / "polls_2026.csv"   # default previous pull
NEW_DIR = Path(sys.argv[1])
if len(sys.argv) > 2:
    OLD = Path(sys.argv[2])
CUTOFF = "2026-11-03"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s.replace("-", " "))).strip()


old, new = pd.read_csv(OLD), pd.read_csv(NEW_DIR / "polls_2026.csv")
for d in (old, new):
    d["pkey"] = d.pollster.map(norm)
K = ["race_id", "pkey", "end_date"]
VALS = ["start_date", "sample_size", "population", "dem_pct", "rep_pct", "other_pct", "partisan_sponsor", "cand_pcts"]
m = new.merge(old[K + VALS], on=K, how="outer", suffixes=("", "_old"), indicator=True)
added = m[m._merge == "left_only"].copy()
removed = m[m._merge == "right_only"].copy()
both = m[m._merge == "both"]
chg = both[[any(str(r[c]) != str(r[c + "_old"]) for c in VALS) for _, r in both.iterrows()]]

# an "added" poll with the same race, end date and numbers as a "removed" one is a renamed pollster, not new
ren = added.drop(columns=["dem_pct_old", "rep_pct_old"]).merge(removed[["race_id", "end_date", "dem_pct_old", "rep_pct_old", "pkey"]]
                  .rename(columns={"pkey": "pollster_before"}),
                  left_on=["race_id", "end_date", "dem_pct", "rep_pct"],
                  right_on=["race_id", "end_date", "dem_pct_old", "rep_pct_old"], how="left")
added["renamed_from"] = ren.pollster_before.values
added["margin_D"] = added.dem_pct - added.rep_pct
cols = ["race_id", "pollster", "start_date", "end_date", "sample_size", "population", "dem_candidate", "dem_pct",
        "rep_candidate", "rep_pct", "margin_D", "other_candidate", "other_pct", "partisan_sponsor", "sponsor",
        "same_party", "renamed_from", "wiki_section", "source_url"]
added = added.sort_values(["race_id", "end_date"])
added[cols].to_csv(NEW_DIR / "new_polls.csv", index=False)
chg.to_csv(NEW_DIR / "changed_polls.csv", index=False)
removed.to_csv(NEW_DIR / "removed_polls.csv", index=False)

trulynew = added[added.renamed_from.isna()]
print(f"old {len(old)} polls / new {len(new)} polls; added {len(added)} (renames {added.renamed_from.notna().sum()}),"
      f" removed {len(removed)}, changed {len(chg)}")
late = trulynew[trulynew.end_date >= "2026-10-25"]
print(f"truly new polls: {len(trulynew)}; ending Oct 25 or later: {len(late)}; "
      f"Senate {late.race_id.str.endswith('SEN').sum()}, House {(~late.race_id.str.endswith('SEN')).sum()}")
pd.set_option("display.width", 250)
print(trulynew[["race_id", "pollster", "start_date", "end_date", "sample_size", "population", "dem_pct", "rep_pct",
                "margin_D", "partisan_sponsor"]].to_string(index=False))
print("\nrenamed (not new):")
print(added[added.renamed_from.notna()][["race_id", "pollster", "renamed_from", "end_date"]].to_string(index=False))
print("\nremoved:")
print(removed[["race_id", "pkey", "start_date_old", "end_date", "dem_pct_old", "rep_pct_old"]].to_string(index=False))
print("\nchanged (same key, different values):")
for _, r in chg.iterrows():
    diffs = {c: (r[c + "_old"], r[c]) for c in VALS if str(r[c]) != str(r[c + "_old"])}
    print(f"  {r.race_id} {r.pollster} {r.end_date}: {diffs}")

# ---- sanity checks
print("\n== checks")
dup = new[new.duplicated(K, keep=False)]
print("duplicate (race, pollster, end date) rows:", len(dup))
near = new.merge(new, on=["race_id", "end_date", "dem_pct", "rep_pct", "other_pct"])
near = near[(near.pkey_x < near.pkey_y)]
print("same race / end date / D% / R% under two pollster names:", len(near))
if len(near):
    print(near[["race_id", "end_date", "pollster_x", "pollster_y", "dem_pct", "rep_pct"]].to_string(index=False))
# the same pollster twice in a race with overlapping field dates (e.g. one poll filed with two end dates)
ov = new.merge(new, on=["race_id", "pkey"])
ov = ov[(ov.end_date_x < ov.end_date_y) & (ov.start_date_y <= ov.end_date_x)]
print("same race + pollster with overlapping field dates:", len(ov))
if len(ov):
    print(ov[["race_id", "pollster_x", "start_date_x", "end_date_x", "start_date_y", "end_date_y", "dem_pct_x",
              "rep_pct_x", "dem_pct_y", "rep_pct_y", "population_x", "population_y"]].to_string(index=False))
print("polls ending after the cutoff (end_date > 2026-11-03):", int((new.end_date > CUTOFF).sum()),
      "| ending after the fetch day (> 2026-11-02):", int((new.end_date > "2026-11-02").sum()),
      "| start after end:", int((new.start_date > new.end_date).sum()))
print("partisan_sponsor in new polls:", trulynew.partisan_sponsor.value_counts(dropna=False).to_dict(),
      "| whole file old / new:", old.partisan_sponsor.value_counts(dropna=False).to_dict(),
      new.partisan_sponsor.value_counts(dropna=False).to_dict())
# pollsters flagged partisan anywhere in the old file but unflagged in a new poll (or vice versa)
flag_old = old.groupby("pkey").partisan_sponsor.agg(lambda s: set(s.dropna()))
for _, r in trulynew.iterrows():
    f0 = flag_old.get(r.pkey, None)
    if f0 is not None and f0 and (str(r.partisan_sponsor) not in f0):
        print(f"  flag differs from the old file: {r.race_id} {r.pollster} new={r.partisan_sponsor} old={f0}")
