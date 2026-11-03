"""Honest backtests: for year T train only on 2010..T-2, predict T, score against results and 538."""
import json
import sys

import samparty

import numpy as np
import pandas as pd

from common import FEAT, WORK, OUT, norm_race_id
from model import WinModel, FINAL

EPS_SAFE = 0.995      # no major-party opponent (vs minor / same party / independent)
EPS_UNOPP = 0.999     # literally unopposed


def to_three_way(pred):
    """Map p_dside to (q_dem, q_rep, q_other) using the contest type."""
    ct = pred.contest_type.fillna("D_vs_R").values.copy()
    # Alaska 2022 Senate: an RCV final between two Republicans is labelled R_vs_IND in the history file
    two_r = pred.notes.fillna("").str.contains("between two Republicans").values if "notes" in pred else np.zeros(len(ct), bool)
    ct[two_r] = "R_vs_R"
    p = pred.p_dside.values
    qd, qr, qo = np.zeros(len(p)), np.zeros(len(p)), np.zeros(len(p))
    for i, c in enumerate(ct):
        if c in ("D_vs_R",):
            qd[i], qr[i] = p[i], 1 - p[i]
        elif c == "R_vs_IND":
            qo[i], qr[i] = p[i], 1 - p[i]
        elif c in ("D_unopposed",):
            qd[i], qr[i], qo[i] = EPS_UNOPP, 0, 1 - EPS_UNOPP
        elif c in ("D_vs_minor", "D_vs_D", "D_vs_IND"):
            qd[i], qr[i], qo[i] = EPS_SAFE, 0, 1 - EPS_SAFE
        elif c in ("R_unopposed",):
            qr[i], qd[i], qo[i] = EPS_UNOPP, 0, 1 - EPS_UNOPP
        elif c in ("R_vs_minor", "R_vs_R"):
            qr[i], qd[i], qo[i] = EPS_SAFE, 0, 1 - EPS_SAFE
        else:
            qd[i], qr[i] = p[i], 1 - p[i]
    return qd, qr, qo


def score(df, qcols, wcol="winner_slot"):
    q = df[qcols].values
    slot = df[wcol].map({"dem": 0, "rep": 1, "other": 2}).values
    qw = np.clip(q[np.arange(len(q)), slot], 0.01, 1)   # floor at 1% like the contest
    onehot = np.eye(3)[slot]
    return dict(n=len(df), logloss=float(-np.mean(np.log(qw))),
                brier=float(np.mean(np.sum((q - onehot) ** 2, axis=1))),
                acc=float(np.mean(np.argmax(q, axis=1) == slot)))


def contest_points(df, qcols, mcols):
    """Contest score with 538 as the 'market': favourite pick and EV-maximising pick."""
    q = df[qcols].values
    m = np.clip(df[mcols].values, 0.01, 1)
    slot = df.winner_slot.map({"dem": 0, "rep": 1, "other": 2}).values
    pts = -np.log2(m)
    fav = np.argmax(q, axis=1)
    ev = np.argmax(q * pts, axis=1)
    idx = np.arange(len(q))
    return dict(fav_pick_points=float(np.sum(np.where(fav == slot, pts[idx, fav], 0))),
                ev_pick_points=float(np.sum(np.where(ev == slot, pts[idx, ev], 0))),
                market_fav_points=float(np.sum(np.where(np.argmax(m, 1) == slot, pts[idx, np.argmax(m, 1)], 0))))


def run_year(hist, T, f538, cfg=FINAL):
    years = [y for y in range(2010, T, 2)]
    wm = WinModel(years, cfg).fit(hist)
    test = hist[hist.year == T].copy()
    pred = wm.predict(test)
    pred["q_dem"], pred["q_rep"], pred["q_other"] = to_three_way(pred)
    pred["winner_slot"] = pred.winner_party.map({"DEM": "dem", "REP": "rep", "IND": "other"})
    out = f538[f538.year == T][["year", "race_id", "chamber"]].merge(
        pred[["race_id538", "q_dem", "q_rep", "q_other", "p_dside", "mu", "sd", "yhat", "winner_slot",
              "contest_type", "rating", "poll_margin", "poll_pollsters", "y"]].rename(columns={"race_id538": "race_id"}),
        on="race_id", how="left")
    missing = out.q_dem.isna().sum()
    if missing:
        print(f"[{T}] {missing} races in the 538 list without a prediction:", out[out.q_dem.isna()].race_id.tolist())
    out.to_csv(WORK / f"backtest_detail_{T}.csv", index=False)
    out[["year", "race_id", "chamber", "q_dem", "q_rep", "q_other"]].to_csv(OUT / f"backtest_{T}.csv", index=False)

    s = out.merge(f538[f538.year == T][["race_id", "dem_win_prob", "rep_win_prob", "other_win_prob"]], on="race_id")
    s = s[s.winner_slot.notna() & s.q_dem.notna()]
    res = {"summary": wm.summary()}
    for ch in ["House", "Senate", "All"]:
        g = s if ch == "All" else s[s.chamber == ch]
        res[ch] = {"model": score(g, ["q_dem", "q_rep", "q_other"]),
                   "538": score(g, ["dem_win_prob", "rep_win_prob", "other_win_prob"]),
                   "points_vs_538_market": contest_points(g, ["q_dem", "q_rep", "q_other"],
                                                          ["dem_win_prob", "rep_win_prob", "other_win_prob"])}
        # competitive subset: 538 favourite < 95%
        c = g[g[["dem_win_prob", "rep_win_prob", "other_win_prob"]].max(axis=1) < 0.95]
        res[ch]["competitive_538lt95"] = {"model": score(c, ["q_dem", "q_rep", "q_other"]),
                                          "538": score(c, ["dem_win_prob", "rep_win_prob", "other_win_prob"])}
    return res, s


def loyo_all(hist, cfg=FINAL):
    """Referee request: every cycle 2010-2024 predicted by the final spec trained on all OTHER years.
    Race ids as in history_races.csv (Senate specials XX-SEN-SPECIAL)."""
    years = list(range(2010, 2026, 2))
    outs, scores = [], {}
    for T in years:
        wm = WinModel([y for y in years if y != T], cfg).fit(hist)
        pred = wm.predict(hist[hist.year == T].copy())
        pred["q_dem"], pred["q_rep"], pred["q_other"] = to_three_way(pred)
        pred["winner_slot"] = pred.winner_party.map({"DEM": "dem", "REP": "rep", "IND": "other"})
        outs.append(pred)
        sp = pred.contest_type.fillna("").isin(["D_vs_D", "R_vs_R"])
        for ch in ("House", "Senate"):
            g = pred[(pred.chamber == ch) & ~sp & pred.winner_slot.notna()]
            scores[f"{T} {ch}"] = score(g, ["q_dem", "q_rep", "q_other"])
    out = pd.concat(outs)
    out[["year", "race_id", "chamber", "q_dem", "q_rep", "q_other"]].to_csv(OUT / "loyo_backtest.csv", index=False)
    return scores


def main(years=(2018, 2022)):
    hist = pd.read_csv(WORK / "features_hist.csv", low_memory=False)
    f538 = pd.read_csv(FEAT / "forecast538_history.csv")
    allres = {}
    for T in years:
        res, s = run_year(hist, T, f538)
        allres[T] = res
        print(f"===== {T}  (train {res['summary']['train_years']})")
        sm = res["summary"]
        print("lambda", {k: round(v, 2) for k, v in sm["lam"].items()}, "lmax", {k: round(v, 3) for k, v in sm["lmax"].items()},
              "a", sm["a"], "sigma", {k: round(v, 2) for k, v in (sm["sig"] or {}).items()}, "sigma parts", sm["sig_parts"])
        print("coefs", res["summary"]["coefs"])
        for ch in ["House", "Senate", "All"]:
            m, f = res[ch]["model"], res[ch]["538"]
            cm, cf = res[ch]["competitive_538lt95"]["model"], res[ch]["competitive_538lt95"]["538"]
            print(f"{ch:6s} n={m['n']:3d}  model LL {m['logloss']:.4f} Brier {m['brier']:.4f} acc {m['acc']:.3f} | "
                  f"538 LL {f['logloss']:.4f} Brier {f['brier']:.4f} acc {f['acc']:.3f} | "
                  f"compet. n={cm['n']} model LL {cm['logloss']:.3f} 538 LL {cf['logloss']:.3f} | pts {res[ch]['points_vs_538_market']}")
    spd = samparty.hist_dataset()
    for T in years:
        sp, spm = samparty.backtest(T, spd)
        party_pred = pd.read_csv(OUT / f"backtest_{T}.csv")
        mc = samparty.multi_backtest(T, party_pred, spm)
        sp = pd.concat([sp[~sp.race_id.isin(mc.race_id)], mc], ignore_index=True)
        sp[["year", "race_id", "candidate_name", "wiki_name", "party", "q"]].to_csv(
            OUT / f"backtest_sameparty_{T}.csv", index=False)
        w = sp[sp.won == 1]
        print(f"same-party {T}: {sp.race_id.nunique()} races, LL {-np.log(np.clip(w.q, 0.01, 1)).mean():.4f}, "
              f"model {spm.summary()}")
        allres[f"sameparty_{T}"] = dict(n=int(sp.race_id.nunique()), ll=float(-np.log(np.clip(w.q, 0.01, 1)).mean()))
    lo = loyo_all(hist)
    print("LOYO (final spec, each year trained on all other years; same-party excluded):")
    for k, v in lo.items():
        print(f"  {k:12s} n={v['n']:3d} LL {v['logloss']:.4f} Brier {v['brier']:.4f} acc {v['acc']:.3f}")
    allres["loyo_all"] = lo
    json.dump(allres, open(WORK / "backtest_metrics.json", "w"), indent=1, default=float)
    return allres


if __name__ == "__main__":
    main()
