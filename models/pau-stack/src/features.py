"""Build one feature table for history (2010-2024) and 2026 with identical columns.

Orientation: every feature is "D-side positive". The D-side is the Democrat, or the independent
who is the main anti-Republican candidate (Orman KS 2014, King ME, Sanders VT, McMullin UT 2022,
Osborn NE 2024/2026, Achilles ID, Bengs SD, Bodnar MT, Hill AK-AL in 2026).

Columns
  y        D-side two-party margin (points); NaN for 2026
  dwin     1 if the D-side won
  lean     -Cook PVI (on the lines in force that year)
  inc      +1 D-side incumbent running, -1 R incumbent running, 0 open
  rating   -(mean of Cook / Inside Elections / Sabato, -3 Safe D .. +3 Safe R)
  rimp     1 if the rating was imputed (race absent from the competitive tables -> Safe)
  gb       final generic-ballot average, D minus R (year-level)
  money    log((D-side receipts + 50k) / (R receipts + 50k)), clipped to +-4
  last_rel previous House result on the same lines (D-side margin) minus that year's national
           House margin; 0 with last_miss = 1 when unavailable (redistricted, new seat, Senate)
  poll_*   final-3-week poll average (see polls.py)
"""
import numpy as np
import pandas as pd

from common import FEAT, WORK, MIDTERM, PRES_PARTY, norm_race_id
import polls as P
import polls_pa as PA
import polls_v4 as P4
import topttwo
from polls_v4 import BORROWED

MONEY_SMOOTH = 5e4

# 2026 races where the main anti-Republican candidate is an independent (by name)
DSIDE_OVERRIDE_2026 = {
    "AK-AL": "Bill Hill", "MT-SEN": "Seth Bodnar", "NE-SEN": "Dan Osborn",
    "ID-SEN": "Todd Achilles", "SD-SEN": "Brian Bengs",
}
# 2026 races where the R-side is a non-Republican (Kevin Kiley, ex-R incumbent, as independent)
RSIDE_OVERRIDE_2026 = {"CA-06": "Kevin Kiley"}


def money_pa(d, r):
    """pau-a's money term: ln(1 + D/$10k) - ln(1 + R/$10k), clipped to +-6, 0 when missing."""
    d = np.nan_to_num(np.asarray(d, float))
    r = np.nan_to_num(np.asarray(r, float))
    return np.clip(np.log1p(d / 1e4) - np.log1p(r / 1e4), -6, 6)


def money_feature(d, r):
    d = np.asarray(d, float)
    r = np.asarray(r, float)
    m = np.log((np.nan_to_num(d) + MONEY_SMOOTH) / (np.nan_to_num(r) + MONEY_SMOOTH))
    miss = np.isnan(d) | np.isnan(r)
    m = np.where(miss, 0.0, np.clip(m, -4, 4))
    return m, miss.astype(int)


def national_table():
    hn = pd.read_csv(FEAT / "history_national.csv")
    nat = pd.DataFrame({
        "year": hn.year,
        "gb": -hn.generic_ballot_margin_r_minus_d,
        "house_actual": -hn.house_popular_vote_margin_two_party_r_minus_d,
        "approval_net": hn.approval_gallup_net,
    })
    n26 = pd.read_csv(WORK / "national.csv")
    r26 = n26[n26.year == 2026].iloc[0]
    nat = pd.concat([nat, pd.DataFrame([{"year": 2026, "gb": r26.gb, "house_actual": np.nan,
                                         "approval_net": r26.get("net_approval", np.nan)}])], ignore_index=True)
    return nat


def build_history():
    h = pd.read_csv(FEAT / "history_races.csv")
    nat = national_table()
    f = pd.read_csv(FEAT / "fec_history.csv")
    h = h.merge(f[["year", "race_id", "dem_receipts", "rep_receipts", "other_receipts"]],
                on=["year", "race_id"], how="left")
    h = h.merge(nat, on="year", how="left")

    ct = h.contest_type.fillna("")
    h["indep_dside"] = (ct == "R_vs_IND").astype(int)
    h["y"] = -h.margin_r_minus_d
    h["dwin"] = np.where(ct == "R_vs_IND", (h.winner_party != "REP").astype(int),
                         (h.winner_party == "DEM").astype(int))
    h["lean"] = -h.pvi_at_time
    dside_inc = np.where(h.indep_dside == 1, h.incumbent_party == "IND", h.incumbent_party == "DEM")
    h["inc"] = np.where(h.incumbent_running == 1,
                        np.where(dside_inc, 1, np.where(h.incumbent_party == "REP", -1, 0)), 0)
    h["rating"] = -h.rating_mean
    h["r_cook"], h["r_ie"], h["r_sab"] = -h.cook_final_num, -h.inside_elections_final_num, -h.sabato_final_num
    h["rimp"] = h.ratings_imputed.fillna(0).astype(int)
    dmoney = np.where(h.indep_dside == 1, h.other_receipts, h.dem_receipts)
    h["money"], h["money_miss"] = money_feature(dmoney, h.rep_receipts)
    h["money_pa"] = money_pa(dmoney, h.rep_receipts)
    # data fix borrowed from pau-a: a rating imputed Safe from the PVI sign against a running incumbent
    # who is not even on the competitive lists (MN-01 2012, Walz) belongs to the incumbent's side
    bad = (h.rimp == 1) & (h.inc != 0) & (np.sign(h.rating) != np.sign(h.inc))
    for c in ("rating", "r_cook", "r_ie", "r_sab"):
        h.loc[bad, c] = 3.0 * np.sign(h.loc[bad, "inc"])

    # previous House result on the same lines
    prev = h[["year", "race_id", "y", "contest_type"]].copy()
    prev["year"] = prev.year + 2
    prev = prev.rename(columns={"y": "y_prev", "contest_type": "ct_prev"})
    h = h.merge(prev, on=["year", "race_id"], how="left")
    nat_prev = nat[["year", "house_actual"]].assign(year=lambda d: d.year + 2).rename(
        columns={"house_actual": "nat_prev"})
    h = h.merge(nat_prev, on="year", how="left")
    ok_prev = (h.chamber == "House") & (h.redistricted == 0) & (h.ct_prev == "D_vs_R") & h.y_prev.notna()
    h["last_rel"] = np.where(ok_prev, (h.y_prev - h.nat_prev).clip(-60, 60), 0.0)
    h["last_miss"] = (~ok_prev).astype(int)

    # usable for training: a D-side vs R contest with a real margin
    h["train_ok"] = (ct.isin(["D_vs_R", "R_vs_IND"]) & h.y.notna() & (h.y.abs() < 99.9)
                     & ~((h.year == 2022) & (h.race_id == "AK-SEN"))).astype(int)

    hp = P.history_polls(h)
    hp["race_id_join"] = hp.race_id
    h["dside_party"] = np.where(h.indep_dside == 1, "IND", "DEM")
    h = h.merge(hp.drop(columns=["race_id_join"]), on=["year", "race_id", "dside_party"], how="left")

    # v3 poll averages (pau-a's method, see polls_pa.py), one column pair per training regime
    at_large = {int(y): set(g.state) for y, g in h[h.district == "AL"].groupby("year")}
    pr = PA.hist_poll_rows(at_large)
    want = dict(zip(zip(h.year, h.race_id), np.where(h.indep_dside == 1, "IND", "DEM")))
    pr = pr[[want.get((y, r), "DEM") == d for y, r, d in zip(pr.year, pr.race_id, pr.dside)]]
    res = h[(h.contest_type == "D_vs_R") & h.y.notna()][["year", "race_id", "y"]]
    bt = PA.bias_table(pr, res)
    bt.to_csv(WORK / "poll_partisan_bias_by_year.csv")
    years = sorted(h.year.unique())
    for regime in ("fwd", "loyo"):
        parts = []
        for T in years:
            yrs = [y for y in years if (y < T if regime == "fwd" else y != T)]
            parts.append(PA.average(pr[pr.year == T], PA.bias_from(bt, yrs)))
        av = pd.concat(parts).rename(columns={"pm": f"pm_{regime}", "pn": f"pn_{regime}"})
        h = h.merge(av, on=["year", "race_id"], how="left")


    # v4: presidential lean (kavi-e's lean_hist: latest presidential election before the year, on that year's lines),
    # D-side margin points: plean = 200 (district D 2-party share - national D 2-party share); plean2 = the one before
    lh = pd.read_csv(BORROWED / "lean_hist.csv")
    lh["plean"] = 200 * (lh.pres_prev_dem_2p - lh.national_pres_dem_2p)
    lh["plean2"] = 200 * (lh.pres_prev2_dem_2p - lh.national_pres_prev2_dem_2p)
    lh["plean2"] = lh.plean2.fillna(lh.plean)
    h["lean_key"] = h.race_id.str.replace("-SEN-SPECIAL", "-SEN", regex=False)
    h = h.merge(lh[["year", "race_id", "plean", "plean2"]].rename(columns={"race_id": "lean_key"}),
                on=["year", "lean_key"], how="left")
    # 53 CA 2012 and 26 FL 2016 House rows have no presidential result on those lines: use 2 x Cook PVI
    # (the fitted slope of plean on PVI is 2.05 in history, 1.98 in 2026)
    h["plean"] = h.plean.fillna(2 * h.lean)
    h["plean2"] = h.plean2.fillna(h.plean)
    # v5: presidential-lean trend (pau-a v4): latest minus previous presidential lean, share points; 0 when unknown
    h["ltrend"] = (h.plean - h.plean2) / 2.0
    # v4: top-two primary margin (CA/WA House, 2012-2024)
    pm = primary_margins(topttwo.history())
    h = h.merge(pm, on=["year", "race_id"], how="left")
    h["has_prim"] = (h.prim_margin.notna() & (h.chamber == "House")).astype(int)
    h["prim_dev"] = np.where(h.has_prim == 1, (h.prim_margin.fillna(0) - 2 * h.lean).clip(-40, 40), 0.0)
    # v4 polls (kavi-e's method, 120-day window where the archive has it)
    indep = set(zip(h.year[h.indep_dside == 1], h.race_id[h.indep_dside == 1]))
    r4 = P4.hist_rows(indep)
    h = h.merge(P4.average(r4), on=["year", "race_id"], how="left")
    for sfx, kw in P4.ALT.items():      # poll-parameter variants for the selection harness
        av = P4.average(r4, **kw).rename(columns=lambda c: c + sfx if c not in ("year", "race_id") else c)
        h = h.merge(av, on=["year", "race_id"], how="left")
    h["senate"] = (h.chamber == "Senate").astype(int)
    h["midterm"] = h.year.map(MIDTERM)
    h["pres"] = h.year.map(PRES_PARTY)
    h["race_id538"] = h.race_id.map(norm_race_id)
    return h


def primary_margins(df):
    """v4: CA/WA top-two primary, D minus R share of the two-party primary vote (points), per House district.
    Pre-election information (June/August primaries). Parties: DEM / REP labels, R-labelled variants counted R."""
    p = df[(df.stage == "primary") & df.district.notna()].copy()
    lab = p.party.fillna("").astype(str)
    p["side"] = np.where(lab == "DEM", "D", np.where((lab == "REP") | lab.str.contains("Republican"), "R", "O"))
    v = p.pivot_table(index=["year", "state", "district"], columns="side", values="votes", aggfunc="sum").fillna(0)
    v = v.reset_index()
    for c in ("D", "R"):
        if c not in v:
            v[c] = 0.0
    v = v[(v.D + v.R) > 0]
    v["prim_margin"] = 100 * (v.D - v.R) / (v.D + v.R)
    v["race_id"] = v.state + "-" + v.district.astype(int).map(lambda d: f"{d:02d}")
    return v[["year", "race_id", "prim_margin"]]


def _first(s):
    if not isinstance(s, str):
        return None
    return s.split(";")[0].strip()


def build_2026():
    r = pd.read_csv(FEAT / "race_info_2026.csv")
    nat = national_table()
    g = pd.read_csv(FEAT / "fec_2026.csv")
    r = r.merge(g[["race_id", "dem_receipts", "rep_receipts", "other_receipts", "other_cand_name_fec",
                   "same_party_race", "sp1_cand_name_fec", "sp1_receipts", "sp2_cand_name_fec", "sp2_receipts"]],
                on="race_id", how="left")
    r["year"] = 2026
    r = r.merge(nat, on="year", how="left")

    dside, rside, kind = [], [], []
    for x in r.itertuples():
        dn = _first(x.dem_candidate)
        rn = _first(x.rep_candidate)
        multi = (isinstance(x.dem_candidate, str) and ";" in x.dem_candidate) or \
                (isinstance(x.rep_candidate, str) and ";" in x.rep_candidate)
        notes = x.notes if isinstance(x.notes, str) else ""
        if x.race_id in DSIDE_OVERRIDE_2026:
            dn, k = DSIDE_OVERRIDE_2026[x.race_id], "R_vs_IND"
        elif x.race_id in RSIDE_OVERRIDE_2026:
            rn, k = RSIDE_OVERRIDE_2026[x.race_id], "D_vs_R"
        elif "same-party top-two general" in notes:
            k = "same_party"
        elif "uncontested" in notes:
            k = "unopposed"
        elif dn is None or rn is None:
            k = "vs_minor"
        elif multi:
            k = "jungle_party"   # Louisiana House: party-level contest
        else:
            k = "D_vs_R"
        dside.append(dn)
        rside.append(rn)
        kind.append(k)
    r["dside_name"], r["rside_name"], r["kind"] = dside, rside, kind
    r["indep_dside"] = (r.kind == "R_vs_IND").astype(int)

    r["lean"] = -r.pvi
    ip = r.incumbent_party.fillna("")
    run = r.incumbent_running.fillna(0) == 1
    dside_inc = np.where(r.indep_dside == 1, ip == "IND", ip == "DEM")
    rside_inc = (ip == "REP") | (r.race_id.isin(RSIDE_OVERRIDE_2026.keys()) & (ip == "IND"))
    r["inc"] = np.where(run, np.where(dside_inc, 1, np.where(rside_inc, -1, 0)), 0)
    r["rating"] = -r.rating_mean
    r["r_cook"], r["r_ie"], r["r_sab"] = -r.cook_num, -r.inside_elections_num, -r.sabato_num
    r["rating_all"] = -r.rating_mean_all   # 11 outlets; 2026 only (no history), analysis only
    r["rimp"] = r.ratings_imputed.fillna(0).astype(int)
    dmoney = np.where(r.indep_dside == 1, r.other_receipts, r.dem_receipts)
    rmoney = np.where(r.race_id.isin(RSIDE_OVERRIDE_2026.keys()), r.other_receipts, r.rep_receipts)
    r["money"], r["money_miss"] = money_feature(dmoney, rmoney)
    r["money_pa"] = money_pa(dmoney, rmoney)

    nat24 = float(nat.loc[nat.year == 2024, "house_actual"].iloc[0])
    # v4 fix: like history, only a contested D-vs-R 2024 general counts (uncontested / same-party / vs-minor = missing)
    kind = r.last_margin_kind.fillna("")
    ok_prev = (r.chamber == "House") & (r.redistricted_2026 == 0) & (r.last_margin_same_lines == 1) & \
              r.last_margin.notna() & (kind == "two-party, 2024 general")
    r["last_rel"] = np.where(ok_prev, (-r.last_margin - nat24).clip(-60, 60), 0.0)
    r["last_miss"] = (~ok_prev).astype(int)

    std = r[r.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])]
    pp = P.polls_2026(dict(zip(std.race_id, std.dside_name)), dict(zip(std.race_id, std.rside_name)))
    pp = pp.drop(columns=["dside_party"]) if len(pp) else pp
    r = r.merge(pp, on=["year", "race_id"], how="left") if len(pp) else r

    # v3 poll averages, partisan bias from all of 2010-2024
    hh = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    at_large = {int(y): set(g.state) for y, g in hh[hh.district.astype(str) == "AL"].groupby("year")}
    pr = PA.hist_poll_rows(at_large)
    want = dict(zip(zip(hh.year, hh.race_id), np.where(hh.indep_dside == 1, "IND", "DEM")))
    pr = pr[[want.get((y, rr), "DEM") == d for y, rr, d in zip(pr.year, pr.race_id, pr.dside)]]
    bt = PA.bias_table(pr, hh[(hh.contest_type == "D_vs_R") & hh.y.notna()][["year", "race_id", "y"]])
    b26 = PA.bias_from(bt, list(range(2010, 2026, 2)))
    print("2026 partisan poll bias (relative to non-partisan):", {k: round(v, 2) for k, v in b26.items()})
    p26 = PA.poll_rows_2026(dict(zip(std.race_id, std.dside_name)), dict(zip(std.race_id, std.rside_name)))
    av = PA.average(p26, b26).rename(columns={"pm": "pm_all", "pn": "pn_all"})
    r = r.merge(av, on=["year", "race_id"], how="left")


    pm = primary_margins(topttwo.primaries_2026())
    r = r.merge(pm, on=["year", "race_id"], how="left")
    r["has_prim"] = (r.prim_margin.notna() & (r.chamber == "House")).astype(int)
    r["prim_dev"] = np.where(r.has_prim == 1, (r.prim_margin.fillna(0) - 2 * r.lean).clip(-40, 40), 0.0)
    l26 = pd.read_csv(BORROWED / "lean_2026.csv")
    l26["plean"] = 200 * (l26.pres24_dem_2p - 0.49250)
    l26["plean2"] = (200 * (l26.pres20_dem_2p - 0.52268)).fillna(l26.plean)
    r = r.merge(l26[["race_id", "plean", "plean2"]], on="race_id", how="left")
    r["ltrend"] = (r.plean - r.plean2) / 2.0
    p4 = P4.rows_2026(dict(zip(std.race_id, std.dside_name)), dict(zip(std.race_id, std.rside_name)))
    p4.to_csv(WORK / "polls_v4_rows_2026.csv", index=False)
    r = r.merge(P4.average(p4), on=["year", "race_id"], how="left")
    for sfx, kw in P4.ALT.items():
        av = P4.average(p4, **kw).rename(columns=lambda c: c + sfx if c not in ("year", "race_id") else c)
        r = r.merge(av, on=["year", "race_id"], how="left")
    r["senate"] = (r.chamber == "Senate").astype(int)
    r["midterm"] = 1
    r["pres"] = PRES_PARTY[2026]
    r["special"] = r.notes.fillna("").str.contains("special election").astype(int)
    r["race_id538"] = r.race_id
    return r


if __name__ == "__main__":
    h = build_history()
    h.to_csv(WORK / "features_hist.csv", index=False)
    f = build_2026()
    f.to_csv(WORK / "features_2026.csv", index=False)
    print(h.groupby(["year", "chamber"]).train_ok.sum().unstack())
    print("polls matched (hist):", h.poll_margin.notna().groupby(h.year).sum().to_dict())
    print(f.kind.value_counts())
    print("polls matched 2026:", f.poll_margin.notna().sum())
    print(f[["race_id", "kind", "dside_name", "rside_name", "lean", "inc", "rating", "money", "last_rel",
             "poll_margin", "poll_pollsters"]][f.senate == 1].to_string())
