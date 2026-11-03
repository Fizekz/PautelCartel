"""pau-t: graphs over House districts (and the Senate race as a state node), one edge list per cycle.

Output pbase/work/pt_edges.csv: year, i, j, graph, w (undirected; each pair once with i < j).
Graphs (all built only from geography / static Census structure; no outcome or poll enters):
  state    same state (House districts and the state's Senate race(s)), w = 1
  adj      districts whose 1:20m boundaries touch (across state lines too), w = 1
  csa      shared combined statistical area (CBSA if not in a CSA): w = sum_m s_im s_jm, s_im = share of district
           i's population in area m (county-population weights)  -- the public media-market proxy
  commute  ACS 2011-15 county-to-county commuting, carried to districts by county-population shares:
           F_ij = sum_ab s_ia s_jb flow(a->b) (both directions), w = F_ij / sqrt(W_i W_j), W_i = district workers
  knn      8 nearest districts nationally in z-scored ACS demographics (college, Hispanic, Black, white, age,
           log income, owner), union-symmetrised, w = 1
  knnl     the same with Cook-style lean added to the features
Lines per cycle (as race/features.csv): 2010 111th; 2012-2020 116th without the state-years whose lines differ
(PA 2012-16, NC 2012-14 and 2020, FL and VA 2012-14); 2022 118th; 2024 119th; 2026 119th without the 9 redrawn
states. Districts of excluded state-years get only state edges (and knn, which uses features.csv).
County-based graphs (csa, commute) need county-population weights (pau-g's pg_cd_county.csv: 116th / 118th /
119th lines), so 2010 has none."""
import zipfile
from pathlib import Path
import numpy as np, pandas as pd
import shapefile
from shapely.geometry import shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_t"
WORK = ROOT / "work"
FIPS2ST = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
           "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
           "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
           "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
           "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
           "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
SHP = {"111": ("cd111/gz_2010_us_500_11_20m", "STATE", "CD"), "116": ("cd116/cb_2018_us_cd116_20m", "STATEFP", "CD116FP"),
       "118": ("cd118/cb_2022_us_cd118_20m", "STATEFP", "CD118FP"), "119": ("cd119/cb_2024_us_cd119_20m", "STATEFP", "CD119FP")}
LINES = {2010: "111", 2012: "116", 2014: "116", 2016: "116", 2018: "116", 2020: "116", 2022: "118", 2024: "119", 2026: "119"}
EXCL = {2012: {"PA", "NC", "FL", "VA"}, 2014: {"PA", "NC", "FL", "VA"}, 2016: {"PA"}, 2020: {"NC"},
        2026: {"CA", "TX", "MO", "NC", "OH", "UT", "FL", "VA", "IN"}}
AL = {"AK", "DE", "MT", "ND", "SD", "VT", "WY"}


def rid(st, cd, lines):
    if cd in ("00", "98") or (st in AL and not (st == "MT" and lines in ("118", "119"))) or (st == "MT" and lines == "116"):
        return f"{st}-AL"
    return f"{st}-{int(cd):02d}"


def polygons(lines):
    f, sf, cf = SHP[lines]
    r = shapefile.Reader(str(RAW / f), encoding="utf-8", encodingErrors="replace")
    names = [x[0] for x in r.fields[1:]]
    out = {}
    for sr in r.iterShapeRecords():
        rec = dict(zip(names, sr.record))
        st = FIPS2ST.get(str(rec[sf]).zfill(2))
        if st is None or st == "DC":
            continue
        g = shape(sr.shape.__geo_interface__)
        k = rid(st, str(rec[cf]).zfill(2), lines)
        out[k] = g if k not in out else out[k].union(g)
    return out


def adjacency(lines):
    P = polygons(lines)
    keys = sorted(P)
    geoms = [P[k].buffer(0.01) for k in keys]
    tree = STRtree(geoms)
    E = set()
    for a, g in enumerate(geoms):
        for b in tree.query(g, predicate="intersects"):
            if b > a:
                E.add((keys[a], keys[b]))
    return keys, E


def cbsa_map():
    d = pd.read_excel(RAW / "cbsa_list1_2023.xlsx", header=2, dtype=str)
    d = d[d["FIPS State Code"].notna() & d["FIPS County Code"].notna()]
    f = (d["FIPS State Code"].str.zfill(2) + d["FIPS County Code"].str.zfill(3)).astype(int)
    area = np.where(d["CSA Code"].notna(), "C" + d["CSA Code"].fillna(""), "M" + d["CBSA Code"].fillna(""))
    m = dict(zip(f, area))
    # Connecticut: the 2023 file uses planning regions; old counties mapped by hand to their 2018-era areas
    m.update({9001: "C408", 9009: "C408", 9003: "C278", 9007: "C278", 9013: "C278", 9011: "C278", 9015: "C278",
              9005: "C278"})
    return m


def commuting():
    d = pd.read_excel(RAW / "commuting_2015_table1.xlsx", header=None, skiprows=7, dtype=str)
    d = d.iloc[:, [0, 1, 4, 5, 8]]
    d.columns = ["rs", "rc", "ws", "wc", "n"]
    d = d[d.rs.notna() & d.ws.notna() & d.rc.notna() & d.wc.notna()]
    d = d[d.ws.str.fullmatch(r"\d{3}") & d.rs.str.fullmatch(r"\d{2}") & d.wc.str.fullmatch(r"\d{3}")]
    d = d[(d.rs.astype(int) <= 56) & (d.ws.astype(int) <= 56)]
    d["a"] = (d.rs.str[-2:] + d.rc.str.zfill(3)).astype(int)
    d["b"] = (d.ws.str[-2:] + d.wc.str.zfill(3)).astype(int)
    d["n"] = pd.to_numeric(d.n, errors="coerce").fillna(0)
    for c in ("a", "b"):
        d[c] = d[c].where(d[c] // 1000 != 2, 2000).replace({51515: 51019, 46113: 46102, 51560: 51005, 15005: 15009})
    return d.groupby(["a", "b"]).n.sum().reset_index()


def county_graphs(year, lines, keep):
    w = pd.read_csv(WORK / "pg_cd_county.csv")
    w = w[(w.lines == int(lines)) & w.race_id.isin(keep)].copy()
    w["s_c"] = w["pop"] / w.groupby("fips")["pop"].transform("sum")       # share of the county in the district
    w["s_d"] = w["pop"] / w.groupby("race_id")["pop"].transform("sum")    # share of the district in the county
    out = []
    cm = cbsa_map()
    w["area"] = w.fips.map(cm)
    a = w[w.area.notna()].groupby(["race_id", "area"]).s_d.sum().reset_index()
    A = a.pivot(index="race_id", columns="area", values="s_d").fillna(0)
    M = A.values @ A.values.T
    ids = A.index.values
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if M[i, j] > 0.01:
                out.append((year, ids[i], ids[j], "csa", float(M[i, j])))
    fl = COMM
    fl = fl[fl.a.isin(w.fips) & fl.b.isin(w.fips)]
    S = w.pivot_table(index="fips", columns="race_id", values="s_c", aggfunc="sum").fillna(0)
    fips = S.index.values
    ix = {f: k for k, f in enumerate(fips)}
    Fc = np.zeros((len(fips), len(fips)))
    Fc[[ix[x] for x in fl.a], [ix[x] for x in fl.b]] = fl.n.values
    Fd = S.values.T @ Fc @ S.values
    Wk = Fd.sum(1)
    Fs = Fd + Fd.T
    np.fill_diagonal(Fs, 0)
    Wn = Fs / np.sqrt(np.outer(Wk, Wk))
    ids = S.columns.values
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if Wn[i, j] > 0.002:
                out.append((year, ids[i], ids[j], "commute", float(Wn[i, j])))
    return out


def knn(year, feats, cols, name, k=8):
    f = feats[(feats.year == year) & (feats.senate == 0)].dropna(subset=cols)
    if len(f) < 50:
        return []
    X = f[cols].values
    X = (X - X.mean(0)) / X.std(0)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    ids = f.race_id.values
    E = set()
    for i in range(len(ids)):
        for j in np.argsort(D[i])[:k]:
            E.add(tuple(sorted((ids[i], ids[j]))))
    return [(year, a, b, name, 1.0) for a, b in E]


def main():
    global COMM
    COMM = commuting()
    print("commuting flows", len(COMM), COMM.n.sum())
    h = pd.read_csv(WORK / "features_hist.csv", low_memory=False, usecols=["year", "race_id", "state", "senate"])
    f26 = pd.read_csv(WORK / "features_2026.csv", low_memory=False) if (WORK / "features_2026.csv").exists() else None
    races = h[["year", "race_id", "state", "senate"]]
    if f26 is not None and "race_id" in f26:
        r26 = f26[["race_id"]].assign(year=2026)
        r26["state"] = r26.race_id.str[:2]
        r26["senate"] = r26.race_id.str.endswith("SEN").astype(int)
        races = pd.concat([races, r26[["year", "race_id", "state", "senate"]]])
    feats = pd.read_csv(ROOT.parent / "race" / "features.csv")
    adj_cache = {}
    rows = []
    for y in sorted(races.year.unique()):
        R = races[races.year == y]
        for st, g in R.groupby("state"):
            ids = sorted(g.race_id)
            rows += [(y, ids[a], ids[b], "state", 1.0) for a in range(len(ids)) for b in range(a + 1, len(ids))]
        lines = LINES[y]
        if lines not in adj_cache:
            adj_cache[lines] = adjacency(lines)
        keys, E = adj_cache[lines]
        ex = EXCL.get(y, set())
        keep = set(R[(R.senate == 0) & ~R.state.isin(ex)].race_id)
        miss = keep - set(keys)
        if miss:
            print(y, "no shape for", sorted(miss)[:10])
        rows += [(y, a, b, "adj", 1.0) for a, b in E if a in keep and b in keep]
        if lines != "111":
            rows += county_graphs(y, lines, keep)
        rows += knn(y, feats, ["college", "hispanic", "black", "white", "age", "loginc", "owner"], "knn")
        rows += knn(y, feats, ["college", "hispanic", "black", "white", "age", "loginc", "owner", "lean"], "knnl")
        print(y, pd.Series([r[3] for r in rows if r[0] == y]).value_counts().to_dict(), flush=True)
    e = pd.DataFrame(rows, columns=["year", "i", "j", "graph", "w"])
    e.to_csv(WORK / "pt_edges.csv", index=False)
    print(e.groupby(["graph"]).w.describe())


if __name__ == "__main__":
    main()
