"""Pau's 2026 forecast in one command: train on 2010-2024, predict the 452 modelled 2026 races.

Adapted from pau-y's pbase/src/final_2026.py (branch pau/y-1103): same steps, same functions, paths made relative to
this folder, plus --features (use a feature table directly) as the default entry point.

  (a) optional --pull: fresh 2026 polls from Wikipedia through pau-f's MediaWiki-API fetcher (src/refresh_polls_2026.py)
      into data/pulls/<UTC tag>/, then pau-f's diff against the previous pull (src/refresh_diff_2026.py).
      THIS IS THE ONLY STEP THAT USES THE NETWORK, and it only runs with --pull.
  (b) the 2026 feature table, rebuilt from a frozen poll pull by features.build_2026(), unchanged, with its poll file
      redirected to <pull>/polls_2026.csv and the 2026 generic ballot taken from <pull>/national_2026.csv's latest
      "Average" row. Default pull: data/pulls/2026-11-03T0125Z (pau-y's last pull, 01:33Z Nov 3). Other shipped pull:
      data/pulls/live_1102 (pau-f's 23:24Z Nov 2 pull; it rebuilds data/features_2026_refresh1102.csv byte for byte).
      --features PATH skips the rebuild and uses a ready feature table instead.
  (c) the late expert-rating move: the rating change between the 2026 rating pages as of 2026-09-29 and 2026-11-02
      (data/stack/wiki), net of the poll movement (pk_2026.rchg_2026), used in unpolled rated seats.
  (d) three models trained on every House and Senate race 2010-2024 (data/features_hist.csv):
      stack_noprim = 'node+geo+field+rchg:clip2' (the main forecast), stack = 'prim+node+geo+field+rchg:clip2'
      (the same plus House primaries), v6nl = 'base' (the base model).
  (e) out/{stack_noprim,stack,v6nl}_2026.csv with columns year,race_id,chamber,q_dem,q_rep,q_other,k_pD_nov1.
      q_dem is P(the D side wins): the Democrat, or the independent in R-vs-IND races such as NE-SEN; q_other = 0.
      k_pD_nov1 is Kalshi's Nov 1 07:48Z price for the same side (data/kalshi; the benchmark price, not a model input).
      Also out/summary.json and out/detail.csv.

Usage (from any folder):
  python run_2026.py                                   # default pull data/pulls/2026-11-03T0125Z -> out/
  python run_2026.py --check-ref                       # ... and fail unless out/ equals reference/<pull>/ to --tol
  python run_2026.py --polls data/pulls/live_1102 --out out/live_1102 --check-ref   # the 0cce402 reproduction
  python run_2026.py --features PATH                   # a ready 2026 feature table (same columns), no rebuild
  python run_2026.py --pull --gap 10                   # NETWORK: fresh Wikipedia pull, rebuild, run (not after 23:00Z Nov 3)
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

DATA = HERE / "data"
PULLS = DATA / "pulls"
DEFAULT_PULL = PULLS / "2026-11-03T0125Z"       # pau-y, pages fetched 01:31-01:33Z Nov 3 (newest revision 01:23Z)
PAUF_PULL = PULLS / "live_1102"                 # pau-f, pages fetched 23:10-23:25Z Nov 2
REF = HERE / "reference"                        # reference/<pull>/: the committed outputs of that pull
                                                # (2026-11-03T0125Z: results/final/2026-11-03T0125Z at 69de6de;
                                                #  live_1102: results/joint2026/*_refreshed.csv at 0cce402)
CUTOFF = "2026-11-03T23:00:00+00:00"
FILES = {"noprim": "stack_noprim", "stack": "stack", "v6nl": "v6nl"}
T = {}


def now():
    return dt.datetime.now(dt.timezone.utc)


def log(*a):
    print(f"[{now().strftime('%H:%M:%SZ')}]", *a, flush=True)


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


# ------------------------------------------------------------------------------------------ (a) pulls
def pull_ok(d: Path) -> bool:
    m = d / "meta.json"
    if not (m.exists() and (d / "polls_2026.csv").exists() and (d / "national_2026.csv").exists()):
        return False
    meta = json.loads(m.read_text())
    return meta.get("stopped") is None and not meta.get("offline") and meta.get("requests", 0) > 0


def pulls_on_disk():
    out = []
    for d in sorted(p for p in PULLS.glob("*") if p.is_dir()):
        if pull_ok(d):
            out.append((json.loads((d / "meta.json").read_text())["finished_utc"], d))
    return [d for _, d in sorted(out)]


def do_pull(tag: str, gap: float, prev: Path, prior_limited: int):
    assert now().isoformat() < CUTOFF, "past the 23:00Z Nov 3 cutoff: no pull"
    out = PULLS / tag
    (out / "cache").mkdir(parents=True, exist_ok=True)
    n = 0
    if (prev / "cache").exists():     # the shipped pull has no page cache: every page is then downloaded
        for f in (prev / "cache").glob("*.json"):
            g = out / "cache" / f.name
            if not g.exists():
                shutil.copy2(f, g)
                n += 1
    log(f"pull {tag}: seeded {n} cached pages from {prev.name} (reused only where the revision id is unchanged)")
    cmd = [sys.executable, str(SRC / "refresh_polls_2026.py"), "--tag", tag, "--out-root", str(PULLS),
           "--gap", str(gap), "--prior-limited", str(prior_limited)]
    with open(out / "fetch.log", "a") as fh:
        r = subprocess.run(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT)
    meta = json.loads((out / "meta.json").read_text())
    if r.returncode != 0 or meta.get("stopped"):
        sys.exit(f"pull {tag} incomplete (exit {r.returncode}, stopped={meta.get('stopped')}); see {out/'fetch.log'}."
                 f" Re-run with --pull --pull-tag {tag} --prior-limited {meta.get('rate_limited', 0)} to resume"
                 " (cached pages are kept).")
    return out


def do_diff(new: Path, prev: Path):
    cmd = [sys.executable, str(SRC / "refresh_diff_2026.py"), str(new), str(prev / "polls_2026.csv")]
    with open(new / "diff.log", "w") as fh:
        fh.write(f"# new = {new.name}, previous = {prev.name}\n")
        fh.flush()
        subprocess.run(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, check=True)


# ------------------------------------------------------------------------------------------ (b) features
def build_features(polls_file: Path, nat_file: Path, out: Path):
    """final_2026.build_features (= pau-f's refresh_v6nl_2026.py build()), unchanged apart from the folders."""
    from common import WORK, FEAT
    import features
    import topttwo
    from bs4 import BeautifulSoup as _BS
    topttwo.BeautifulSoup = lambda html, *_a, **_k: _BS(html, "html.parser")   # lxml is not needed
    _read, _write = pd.read_csv, pd.DataFrame.to_csv
    natw = _read(WORK / "national.csv")
    n = _read(nat_file)
    agg = n[(n.metric == "generic_ballot") & (n.pollster == "Average")].sort_values("as_of").iloc[-1]
    gb, asof = float(agg.dem_pct - agg.rep_pct), str(agg.as_of)
    gb_old = float(natw.loc[natw.year == 2026, "gb"].iloc[0])
    natw.loc[natw.year == 2026, "gb"] = gb
    natf = out / "national.csv"
    _write(natw, natf, index=False)

    def rd(path, *a, **k):
        p = str(path)
        if p == str(FEAT / "polls_2026.csv"):
            path = polls_file
        elif p == str(WORK / "national.csv"):
            path = natf
        return _read(path, *a, **k)

    def wr(self, path=None, *a, **k):
        if path is not None and str(path).startswith(str(WORK)):
            path = out / Path(path).name
        return _write(self, path, *a, **k)

    pd.read_csv, pd.DataFrame.to_csv = rd, wr
    try:
        f = features.build_2026()
    finally:
        pd.read_csv, pd.DataFrame.to_csv = _read, _write
    _write(f, out / "features_2026.csv", index=False)
    return out / "features_2026.csv", dict(gb=gb, gb_as_of=asof, gb_frozen_national=gb_old)


# ------------------------------------------------------------------------------------------ (c, d) models
def run_models(f26: pd.DataFrame):
    """final_2026.run_models, unchanged (= pk_2026.main's model part)."""
    import pk_2026 as P26          # imports pk_stack: installs the stack hooks on WinModel
    import pk_stack as K
    import pk_run as R
    from model import WinModel
    std = f26[f26.kind.isin(["D_vs_R", "R_vs_IND", "jungle_party"])].copy()
    std["year"] = 2026
    r = P26.rchg_2026(std)
    std = std.merge(r[["race_id", "rchg_res"]], on="race_id", how="left")
    unp = ((std.poll_pollsters.fillna(0) == 0) & (std.pn4.fillna(0) == 0)).astype(float)
    std["rchg_unp"] = std.rchg_res.fillna(0) * unp * (std.rating.abs() < 2.5)
    Y = list(range(2010, 2026, 2))
    d = std[["race_id", "chamber", "senate", "rating", "poll_pollsters", "poll_margin", "pm4", "pn4", "rchg_unp",
             "dside_name", "rside_name", "kind"]].copy()
    for k, name in P26.MODELS.items():
        t0 = time.time()
        comps, kw, _, _ = R.parse(name)
        d[f"p_{k}"] = WinModel(Y, K.build(comps, **kw)).fit(K.HIST).predict(std).p_dside.values
        T[f"model_{k}_s"] = round(time.time() - t0, 1)
        log(f"model {k} ({name}) fitted in {T[f'model_{k}_s']} s")
    d["k_pD"] = d.race_id.map(P26.kalshi())
    return d


# ------------------------------------------------------------------------------------------ (e) outputs
def write_outputs(d: pd.DataFrame, out: Path):
    for k, fn in FILES.items():
        q = d[f"p_{k}"].values
        pd.DataFrame({"year": 2026, "race_id": d.race_id.values, "chamber": d.chamber.values, "q_dem": q,
                      "q_rep": 1 - q, "q_other": 0.0, "k_pD_nov1": d.k_pD.values}).to_csv(out / f"{fn}_2026.csv",
                                                                                         index=False)


def compare(out: Path, ref: Path, suffix: str, tol=1e-9):
    """max |diff| of every numeric column against the reference files; within_tol if all <= tol."""
    res = {}
    for k, fn in FILES.items():
        rf = ref / f"{fn}_2026{suffix}.csv"
        if not rf.exists():
            continue
        a, b = pd.read_csv(out / f"{fn}_2026.csv"), pd.read_csv(rf)
        m = a.merge(b, on=["year", "race_id", "chamber"], how="outer", suffixes=("", "_ref"), indicator=True)
        res[fn] = dict(n=len(a), n_ref=len(b), unmatched=int((m._merge != "both").sum()),
                       max_abs={c: float(np.nanmax(np.abs(m[c] - m[c + "_ref"]))) for c in
                                ("q_dem", "q_rep", "q_other", "k_pD_nov1")})
        res[fn]["within_tol"] = bool(res[fn]["unmatched"] == 0 and max(res[fn]["max_abs"].values()) <= tol)
    return res


def summary(d, feat_path: Path, pull, info: dict, out: Path):
    h, s, ok = d.chamber == "House", d.chamber == "Senate", d.k_pD.notna()
    pts = lambda p: -np.log2(np.clip(p, 0.01, 1))
    res = dict(created_utc=now().isoformat(timespec="seconds"), features=rel(feat_path),
               pull=rel(pull) if pull else None, **info, n_races=int(len(d)), n_kalshi=int(ok.sum()),
               races_with_21day_lv_rv_poll=int((d.poll_pollsters.fillna(0) > 0).sum()),
               kalshi_nov1=dict(expected_D_house=round(float(d.k_pD[h].sum()), 1),
                                expected_D_senate=round(float(d.k_pD[s].sum()), 2)),
               python=sys.version.split()[0], numpy=np.__version__)
    if pull is not None:
        polls = pd.read_csv(pull / "polls_2026.csv")
        res["polls"] = dict(n=int(len(polls)), n_races=int(polls.race_id.nunique()),
                            newest_end_date=str(polls.end_date.max()),
                            ending_oct25_or_later=int((polls.end_date >= "2026-10-25").sum()))
        res["pull_meta"] = json.loads((pull / "meta.json").read_text())
        nf = pull / "new_polls.csv"
        if nf.exists() and pull.resolve() != PAUF_PULL.resolve() and (pull / "removed_polls.csv").exists():
            npl = pd.read_csv(nf)
            res["polls"]["new_vs_previous_pull"] = int(npl.renamed_from.isna().sum())
            rem = pd.read_csv(pull / "removed_polls.csv")
            rn = npl[npl.renamed_from.notna()]             # final_2026.py at 69de6de: a rename is not a removal
            ren = set(zip(rn.race_id, rn.renamed_from, rn.end_date))
            rem = rem[[(r, k, e) not in ren for r, k, e in zip(rem.race_id, rem.pkey, rem.end_date)]]
            res["polls"]["renamed_vs_previous_pull"] = rn[["race_id", "renamed_from", "pollster", "end_date"]
                                                          ].values.tolist()
            res["polls"]["removed_vs_previous_pull"] = rem[["race_id", "pkey", "end_date"]].values.tolist()
            # traps pau-f flagged: untagged polls from firms tagged partisan elsewhere, '(R?)'-style names, adults
            old_flags = pd.read_csv(PAUF_PULL / "polls_2026.csv")
            tagged = set(old_flags.pollster[old_flags.partisan_sponsor.notna()].str.lower().str.split().str[0])
            npl["first"] = npl.pollster.str.lower().str.split().str[0]
            trap = npl[npl.renamed_from.isna() & (npl.partisan_sponsor.isna() & (npl["first"].isin(tagged) |
                       npl.pollster.str.contains(r"\(|\?", regex=True)) | npl.population.astype(str).str.lower().eq("a"))]
            res["polls"]["new_poll_traps"] = trap[["race_id", "pollster", "end_date", "population", "margin_D"]
                                                  ].values.tolist()
    for k in FILES:
        p = d[f"p_{k}"]
        dis = d[ok & ((p >= 0.5) != (d.k_pD >= 0.5))]
        evD, evR = p * pts(d.k_pD), (1 - p) * pts(1 - d.k_pD)
        pickD = evD > evR
        res[k] = dict(expected_D_house=round(float(p[h].sum()), 1), expected_D_senate=round(float(p[s].sum()), 2),
                      fav_disagree_kalshi=dis[["race_id", f"p_{k}", "k_pD"]].round(3).values.tolist(),
                      ev_picks_against_kalshi_fav=int((ok & (pickD != (d.k_pD >= 0.5))).sum()),
                      expected_points_own_model=round(float(np.where(pickD, evD, evR)[ok].sum()), 1))
    d.to_csv(out / "detail.csv", index=False)
    res["runtime_s"] = T
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polls", help=f"pull folder to rebuild the feature table from (default {rel(DEFAULT_PULL)})")
    ap.add_argument("--features", help="use this ready 2026 feature table instead of rebuilding one from a pull")
    ap.add_argument("--pull", action="store_true", help="NETWORK: fetch fresh polls first (MediaWiki API)")
    ap.add_argument("--pull-tag", help="pull folder name under data/pulls (default: UTC time); reuse to resume")
    ap.add_argument("--gap", type=float, default=5.0, help="seconds between API requests (the fetcher enforces >= 5)")
    ap.add_argument("--prior-limited", type=int, default=0, help="rate-limit responses already received (resume)")
    ap.add_argument("--out", default=str(HERE / "out"), help="output folder (default out/)")
    ap.add_argument("--ref", help="reference folder for --check-ref (default reference/<pull name>)")
    ap.add_argument("--check-ref", action="store_true", help="fail unless the outputs equal the reference to --tol")
    ap.add_argument("--tol", type=float, default=1e-9, help="tolerance of --check-ref (default 1e-9)")
    a = ap.parse_args()
    t_all = time.time()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pull, info = None, {}
    if a.pull:
        prevs = pulls_on_disk()
        tag = a.pull_tag or now().strftime("%Y-%m-%dT%H%MZ")
        prev = [p for p in prevs if p.name != tag][-1]
        t0 = time.time()
        pull = do_pull(tag, a.gap, prev, a.prior_limited)
        T["pull_s"] = round(time.time() - t0, 1)
        do_diff(pull, prev)
        log(f"pull done in {T['pull_s']} s; diff vs {prev.name} in {pull/'diff.log'}")
    elif not a.features:
        pull = Path(a.polls).resolve() if a.polls else DEFAULT_PULL
        assert (pull / "polls_2026.csv").exists() and (pull / "national_2026.csv").exists(), f"{pull}: not a pull"
    if pull is not None:
        t0 = time.time()
        feat_path, info = build_features(pull / "polls_2026.csv", pull / "national_2026.csv", out)
        T["features_s"] = round(time.time() - t0, 1)
        log(f"features_2026 rebuilt from {rel(pull)} in {T['features_s']} s -> {rel(feat_path)}; 2026 gb "
            f"{info['gb_frozen_national']:.2f} -> {info['gb']:.2f} (as of {info['gb_as_of']})")
    else:
        feat_path = Path(a.features).resolve()
    log(f"features: {rel(feat_path)}; outputs: {rel(out)}")
    f26 = pd.read_csv(feat_path, low_memory=False)

    t0 = time.time()
    d = run_models(f26)
    T["models_s"] = round(time.time() - t0, 1)
    write_outputs(d, out)
    T["total_s"] = round(time.time() - t_all, 1)
    res = summary(d, feat_path, pull, info, out)
    ref = Path(a.ref).resolve() if a.ref else (REF / pull.name if pull is not None else None)
    cmp_ = compare(out, ref, "", a.tol) if ref is not None and ref.exists() else {}
    res["reference"] = rel(ref) if ref is not None and ref.exists() else None
    res["compare_reference"] = cmp_
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    show = {k: v for k, v in res.items() if k not in ("pull_meta",)}
    print(json.dumps(show, indent=1, default=float))
    pl = res.get("polls", {})
    if pl.get("removed_vs_previous_pull"):
        print("WARNING: polls removed since the previous pull (check diff.log; a restructured page can drop polls):",
              pl["removed_vs_previous_pull"])
    if pl.get("new_poll_traps"):
        print("CHECK these new polls (untagged firm tagged partisan elsewhere, '(R?)' names, adult samples):",
              pl["new_poll_traps"])
    log(f"wrote {rel(out)}/{{stack_noprim,stack,v6nl}}_2026.csv ({len(d)} races) in {T['total_s']} s")
    if a.check_ref:
        bad = [k for k, v in cmp_.items() if not v["within_tol"]]
        if bad or len(cmp_) < 3:
            sys.exit(f"REFERENCE CHECK FAILED: {bad or 'reference files missing'}")
        print(f"REFERENCE CHECK OK: all three files equal {res['reference']} to {a.tol:g} (python {sys.version.split()[0]},"
              f" numpy {np.__version__})")


if __name__ == "__main__":
    main()
