"""Same-party generals (CA/WA top-two, LA jungle, CA Senate 2016/2018): candidate-level sub-model.

Latent variable: two-way general margin of candidate 1 (the leader of the two finalists in the
top-two primary) over candidate 2, in points.
  prior : m = B_PRIM * two-way primary margin + B_INC * (inc1 - inc2) + e,  e ~ N(0, SD^2)
          fitted by OLS on past same-party generals (CA/WA 2012..T-2, primary and general results
          parsed from Wikipedia, see topttwo.py), so a backtest for year T uses earlier years only.
  polls : polls of the two finalists ending in the last 75 days (two-way margin, weighted by
          sqrt(n) and recency), error sd_p^2 = TAU^2 + 8^2 / n_pollsters, TAU fitted by maximum
          likelihood on the polled training races (falls back to 10 with fewer than 5).
  result: Gaussian posterior, P(cand1) = Phi(mu / sd)
Races without a primary (Louisiana jungle generals) use the same fitted model with primary
margin 0, i.e. incumbency only.
"""
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from common import FEAT, RAW, WORK, ELECTION_DATES, norm_name, surname
import polls as P
import topttwo

WINDOW = 75
S1 = 8.0
TAU_PRIOR = 10.0
QCAP = 0.995     # same cap as other no-contest races: never more sure than 99.5%


def same_name(a, b):
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    return ta[-1] == tb[-1] and ta[0][0] == tb[0][0]


def _find(rows, name):
    for r in rows:
        if same_name(r["candidate"], name):
            return r
    for r in rows:   # surname only, if unique
        pass
    cand = [r for r in rows if surname(r["candidate"]) == surname(name)]
    return cand[0] if len(cand) == 1 else None


# ---------------------------------------------------------------- history
def hist_polls():
    rp = pd.read_csv(RAW / "history/national/538_raw_polls.csv", low_memory=False)
    rp = rp[rp.type_simple.isin(["House-G", "Sen-G"]) & (rp.cand1_party == rp.cand2_party) & (rp.cycle % 2 == 0)]
    rp = rp[~rp.race.str.contains("_House-R_|_Sen-R_|_Sen-SR_")]
    out = []
    for r in rp.itertuples():
        loc = str(r.location)
        st = loc.split("-")[0]
        dist = int(loc.split("-")[1]) if "-" in loc else 0
        days = (pd.Timestamp(ELECTION_DATES[int(r.cycle)]) - pd.Timestamp(r.polldate)).days
        out.append(dict(year=int(r.cycle), state=st, district=dist, pollster=r.pollster, days=days,
                        n=r.samplesize, c1=r.cand1_name, v1=r.cand1_pct, c2=r.cand2_name, v2=r.cand2_pct))
    return pd.DataFrame(out)


def poll_avg(pp, n1, n2):
    ms, ws, pollsters = [], [], set()
    for r in pp.itertuples():
        if same_name(r.c1, n1) and same_name(r.c2, n2):
            v1, v2 = r.v1, r.v2
        elif same_name(r.c1, n2) and same_name(r.c2, n1):
            v1, v2 = r.v2, r.v1
        else:
            continue
        if v1 + v2 <= 0 or r.days < 0 or r.days > WINDOW:
            continue
        ms.append(100 * (v1 - v2) / (v1 + v2))
        n = r.n if r.n == r.n else 500
        ws.append(np.sqrt(min(n, 1500)) * np.exp(-r.days / 21))
        pollsters.add(r.pollster)
    if not ms:
        return np.nan, 0
    return float(np.average(ms, weights=ws)), len(pollsters)


def hist_dataset():
    t = topttwo.history()
    hp = hist_polls()
    rows = []
    for (y, st, d), g in t.groupby(["year", "state", "district"]):
        gen = g[g.stage == "general"].sort_values("votes", ascending=False)
        pri = g[g.stage == "primary"].to_dict("records")
        if len(gen) < 2:
            continue
        a, b = gen.iloc[0], gen.iloc[1]
        if a.party != b.party or a.party not in ("DEM", "REP"):
            continue
        ra, rb = _find(pri, a.candidate), _find(pri, b.candidate)
        pa = ra["pct"] if ra else np.nan
        pb = rb["pct"] if rb else np.nan
        ia = max(int(a.incumbent), ra["incumbent"] if ra else 0)
        ib = max(int(b.incumbent), rb["incumbent"] if rb else 0)
        # orient: candidate 1 = primary leader
        if pb == pb and (pa != pa or pb > pa):
            (a, pa, ia), (b, pb, ib) = (b, pb, ib), (a, pa, ia)
        gm = 100 * (a.votes - b.votes) / (a.votes + b.votes)
        pm = 100 * (pa - pb) / (pa + pb) if pa == pa and pb == pb else 0.0
        pol, npol = poll_avg(hp[(hp.year == y) & (hp.state == st) & (hp.district == d)], a.candidate, b.candidate)
        rid = f"{st}-SEN" if d == 0 else (f"{st}-{d:02d}")
        rows.append(dict(year=y, race_id=rid, party=a.party, cand1=a.candidate, cand2=b.candidate,
                         prim1=pa, prim2=pb, prim_m=pm, inc1=ia, inc2=ib, poll=pol, npoll=npol,
                         gen_m=gm, c1_won=int(gm > 0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model
class SamePartyModel:
    def fit(self, d):
        A = np.column_stack([d.prim_m.values, (d.inc1 - d.inc2).values])
        self.b_ = np.linalg.lstsq(A, d.gen_m.values, rcond=None)[0]
        e = d.gen_m.values - A @ self.b_
        self.sd_ = float(np.sqrt(np.sum(e ** 2) / max(len(d) - 2, 1)))
        g = d[d.npoll > 0]
        if len(g) >= 5:
            mu0 = np.column_stack([g.prim_m, g.inc1 - g.inc2]) @ self.b_

            def nll(tau):
                sp2 = tau ** 2 + S1 ** 2 / g.npoll.values
                prec = 1 / self.sd_ ** 2 + 1 / sp2
                mu = (mu0 / self.sd_ ** 2 + g.poll.values / sp2) / prec
                s = np.sqrt(1 / prec)
                return -np.sum(norm.logpdf(g.gen_m.values, mu, s))

            self.tau_ = float(minimize_scalar(nll, bounds=(2, 30), method="bounded").x)
        else:
            self.tau_ = TAU_PRIOR
        # races without a primary: margin of the incumbent when exactly one finalist is an incumbent
        one = d[(d.inc1 + d.inc2) == 1]
        mi = np.where(one.inc1 == 1, one.gen_m, -one.gen_m)
        self.inc_mu_, self.inc_sd_ = float(np.mean(mi)), float(np.std(mi, ddof=1))
        self.n_ = len(d)
        return self

    def p_incumbent_no_primary(self):
        return float(np.clip(norm.cdf(self.inc_mu_ / self.inc_sd_), 1 - QCAP, QCAP))

    def predict(self, prim_m, inc_diff, poll, npoll):
        mu0 = self.b_[0] * prim_m + self.b_[1] * inc_diff
        if npoll and poll == poll:
            sp2 = self.tau_ ** 2 + S1 ** 2 / npoll
            prec = 1 / self.sd_ ** 2 + 1 / sp2
            mu = (mu0 / self.sd_ ** 2 + poll / sp2) / prec
            sd = np.sqrt(1 / prec)
        else:
            mu, sd = mu0, self.sd_
        return float(mu0), float(mu), float(sd), float(np.clip(norm.cdf(mu / sd), 1 - QCAP, QCAP))

    def summary(self):
        return dict(n=self.n_, b_prim=round(float(self.b_[0]), 3), b_inc=round(float(self.b_[1]), 2),
                    sd=round(self.sd_, 2), tau=round(self.tau_, 2), inc_only_mu=round(self.inc_mu_, 1),
                    inc_only_sd=round(self.inc_sd_, 1))


# ---------------------------------------------------------------- backtests 2018 / 2022
def mit_names(year):
    h = pd.read_csv(RAW / "history/mit_1976-2024-house.tab", low_memory=False)
    h = h[(h.year == year) & (h.stage == "GEN")]
    s = pd.read_csv(RAW / "history/mit_1976-2024-senate-state.csv", low_memory=False)
    s = s[(s.year == year) & (s.stage == "gen")]
    out = {}
    for r in h.itertuples():
        d = int(r.district)
        rid = f"{r.state_po}-AL" if d == 0 else f"{r.state_po}-{d:02d}"
        out.setdefault(rid, []).append((str(r.candidate), str(r.party)))
    for r in s.itertuples():
        out.setdefault(f"{r.state_po}-SEN", []).append((str(r.candidate), str(r.party_simplified)))
    return out


def backtest(T, data=None):
    d = hist_dataset() if data is None else data
    m = SamePartyModel().fit(d[d.year < T])
    test = d[d.year == T]
    names = mit_names(T)
    rows = []
    for x in test.itertuples():
        mu0, mu, sd, p1 = m.predict(x.prim_m, x.inc1 - x.inc2, x.poll, x.npoll)
        mit = names.get(x.race_id, [])
        for cand, q in ((x.cand1, p1), (x.cand2, 1 - p1)):
            mn = next((n for n, _ in mit if same_name(n, cand)), None)
            if mn is None:
                sm = [n for n, _ in mit if surname(n) == surname(cand)]
                mn = sm[0] if len(sm) == 1 else cand
            rows.append(dict(year=T, race_id=x.race_id, candidate_name=mn, wiki_name=cand, party=x.party, q=q,
                             mu=mu, sd=sd, won=int((cand == x.cand1) == bool(x.c1_won))))
    # same-party generals without a top-two primary (Louisiana jungle, Alaska RCV 2022): every
    # candidate on the MIT ballot list is kept (choosing "the two finalists" from the result would
    # leak); the incumbent / polled pair get the fitted model, the rest share a 2% remainder.
    hr = pd.read_csv(FEAT / "history_races.csv")
    flag = hr.contest_type.isin(["R_vs_R", "D_vs_D"]) | hr.notes.fillna("").str.contains(
        "same-party general|between two Republicans|between two Democrats")
    extra = hr[(hr.year == T) & flag & ~hr.state.isin(["CA", "WA"])]
    hp = hist_polls()
    for x in extra.itertuples():
        rid = x.race_id.replace("-SPECIAL", "")
        mit = [c for c, _ in names.get(rid, []) if c.upper() not in ("WRITEIN", "OTHER", "BLANK", "NAN")]
        mit = list(dict.fromkeys(mit))
        if not mit:
            continue
        party = "REP" if ("R_vs_R" in str(x.contest_type) or "Republicans" in str(x.notes)) else "DEM"
        winner = str(x.winner_name)
        pp = hp[(hp.year == T) & (hp.state == x.state) & (hp.district == (0 if x.chamber == "Senate" else int(x.district)))]
        pair = None
        if len(pp):
            a, b = pp.iloc[0].c1, pp.iloc[0].c2
            ca = next((c for c in mit if surname(c) == surname(a)), None)
            cb = next((c for c in mit if surname(c) == surname(b)), None)
            if ca and cb:
                pol, npol = poll_avg(pp, a, b)
                inc = str(x.incumbent_name) if isinstance(x.incumbent_name, str) else ""
                i_a, i_b = int(surname(inc) == surname(a)), int(surname(inc) == surname(b))
                _, _, _, pa = m.predict(0.0, i_a - i_b, pol, npol)
                pair = {ca: pa, cb: 1 - pa}
        if pair is None:
            inc = str(x.incumbent_name) if isinstance(x.incumbent_name, str) else ""
            inc_c = next((c for c in mit if surname(c) == surname(inc)), None) if x.incumbent_running == 1 else None
            if inc_c is None:
                continue
            p_inc = m.p_incumbent_no_primary()
            pair = {inc_c: p_inc}
            rest = [c for c in mit if c != inc_c]
            for c in rest:
                pair[c] = (1 - p_inc) / len(rest)
            qs = pair
        else:
            rest = [c for c in mit if c not in pair]
            qs = {c: v * (0.98 if rest else 1.0) for c, v in pair.items()}
            for c in rest:
                qs[c] = 0.02 / len(rest)
        for c, q in qs.items():
            rows.append(dict(year=T, race_id=x.race_id, candidate_name=c, wiki_name=c, party=party, q=q,
                             mu=np.nan, sd=np.nan, won=int(surname(c) == surname(winner))))
    return pd.DataFrame(rows), m


# ---------------------------------------------------------------- multi-candidate ballots
# Jungle rounds (Louisiana in November, Mississippi 2018 special), Alaska top-four/RCV ballots,
# fusion lines (NY-14 2018). Party-level probabilities come from the main model's honest backtest
# for that year; inside a party the incumbent (or the poll leader) is set against the strongest
# rival with the same-party machinery (incumbent-only prior, updated by polls); everybody else in
# the party shares the rest; minor-party candidates share 0.4%. For a jungle round q is the
# probability of finishing first, which the party-level D-vs-R margin approximates.
REFEREE_LIST = {2018: ["LA-01", "LA-03", "LA-06", "MS-02", "MS-SEN-SPECIAL", "NY-14"],
                2022: ["AK-AL", "AK-SEN", "LA-03", "LA-05", "LA-SEN"]}
MINOR_TOTAL = 0.004
JUNK = ("WRITEIN", "WRITE-IN", "OTHER", "BLANK", "OVERVOTES", "UNDERVOTES", "SCATTERING", "NAN")


def _party3(p):
    p = str(p).upper()
    if p.startswith("DEMOCRAT"):
        return "DEM"
    if p.startswith("REPUBLICAN"):
        return "REP"
    return "OTH"


def ballot(T, rid):
    """[(name, party)] from MIT, fusion lines merged (a name keeps its D/R line if it has one)."""
    st = rid.split("-")[0]
    if "-SEN" in rid:
        s = pd.read_csv(RAW / "history/mit_1976-2024-senate-state.csv", low_memory=False)
        s = s[(s.year == T) & (s.state_po == st) & (s.stage.str.lower() == "gen") &
              (s.special == rid.endswith("SPECIAL"))]
        rows = [(str(r.candidate), _party3(r.party_simplified), r.candidatevotes) for r in s.itertuples()
                if not bool(r.writein)]
    else:
        h = pd.read_csv(RAW / "history/mit_1976-2024-house.tab", low_memory=False)
        d = 0 if rid.endswith("-AL") else int(rid.split("-")[1])
        h = h[(h.year == T) & (h.state_po == st) & (h.district == d) & (h.stage == "GEN")]
        rows = [(str(r.candidate), _party3(r.party), r.candidatevotes) for r in h.itertuples()
                if not bool(r.writein)]
    out = {}
    for n, p, _ in rows:
        if n.upper().strip('\\" ') in JUNK:
            continue
        if n not in out or (out[n] == "OTH" and p != "OTH"):
            out[n] = p
    return list(out.items())


def _polls_hist(T, rid):
    p = pd.read_csv(FEAT / "polls_history.csv", low_memory=False)
    rid538 = rid.replace("-SEN-SPECIAL", "-SEN-S")
    p = p[(p.year == T) & (p.race_id == rid538) & (p.stage.isin(["general", "jungle primary"]))]
    ed = pd.Timestamp(ELECTION_DATES[T])
    p = p[(ed - pd.to_datetime(p.end_date)).dt.days.between(0, WINDOW)]
    rows = []
    for r in p.itertuples():
        for c, v, party in ((r.dem_candidate, r.dem_pct, "DEM"), (r.dem2_candidate, r.dem2_pct, "DEM"),
                            (r.rep_candidate, r.rep_pct, "REP"), (r.rep2_candidate, r.rep2_pct, "REP")):
            if isinstance(c, str) and v == v:
                rows.append(dict(pollster=r.pollster, end=r.end_date, cand=c, pct=v, party=party))
    return pd.DataFrame(rows)


def _flip(n):
    """'MURKOWSKI, LISA' -> 'LISA MURKOWSKI' (MIT's Alaska 2022 style); other names unchanged."""
    if "," in n and not re.search(r",\s*(JR|SR|II|III|IV)\.?$", n.strip(), flags=re.I):
        last, first = n.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return n


def jungle_incumbent_rate(T):
    """Share of Louisiana November-jungle incumbents (House, MIT ballots) who finished first among
    their own party's candidates, cycles 2012..T-2, with a Jeffreys prior: (k + 0.5) / (n + 1)."""
    hr = pd.read_csv(FEAT / "history_races.csv")
    h = pd.read_csv(RAW / "history/mit_1976-2024-house.tab", low_memory=False)
    h = h[(h.state_po == "LA") & (h.stage == "GEN") & (h.year >= 2012) & (h.year < T) & (h.writein == False)]
    k = n = 0
    for (y, d), g in h.groupby(["year", "district"]):
        x = hr[(hr.year == y) & (hr.race_id == f"LA-{int(d):02d}")]
        if not len(x) or x.iloc[0].incumbent_running != 1:
            continue
        pty = "REPUBLICAN" if x.iloc[0].incumbent_party == "REP" else "DEMOCRAT"
        gg = g[g.party == pty].groupby("candidate").candidatevotes.sum().sort_values(ascending=False)
        if len(gg) < 2:
            continue
        n += 1
        k += int(surname(gg.index[0]) in {surname(v) for v in str(x.iloc[0].incumbent_name).split(";")})
    return (k + 0.5) / (n + 1), k, n


def multi_candidate(T, rid, party_q, inc_name, m):
    """party_q: dict DEM/REP/OTH -> probability (party-level, honest backtest)."""
    cands = ballot(T, rid)
    key = {n: surname(_flip(n)) for n, _ in cands}
    pol = _polls_hist(T, rid)
    # parties from the polls when MIT has no party (Alaska 2022 Senate lists everyone as OTHER)
    if len(pol):
        pm = {surname(c): p for c, p in zip(pol.cand, pol.party)}
        cands = [(n, pm.get(key[n], p) if p == "OTH" else p) for n, p in cands]
    q = {}
    minors = [n for n, p in cands if p == "OTH"]
    scale = 1 - (MINOR_TOTAL if minors else 0)
    for party in ("DEM", "REP"):
        members = [n for n, p in cands if p == party]
        pq = party_q.get(party, 0.0) * scale
        if not members:
            continue
        if len(members) == 1:
            q[members[0]] = pq
            continue
        avg = {}
        if len(pol):
            for n in members:
                v = pol[pol.cand.map(surname) == key[n]].groupby("pollster").pct.mean()
                if len(v):
                    avg[n] = (float(v.mean()), len(v))
        inc = next((n for n in members if key[n] == surname(inc_name)), None) if inc_name else None
        lead = inc or (max(avg, key=lambda k: avg[k][0]) if avg else None)
        if lead is None:
            for n in members:
                q[n] = pq / len(members)
            continue
        rivals = [n for n in members if n != lead]
        rival = max(rivals, key=lambda k: avg.get(k, (-1, 0))[0])
        mu0 = m.inc_mu_ * (int(lead == inc) - 0)
        sd0 = m.inc_sd_ if lead == inc else m.sd_
        # head-to-head (final-round, for RCV) polls of the pair from 538 raw_polls beat first-round shares
        st = rid.split("-")[0]
        dist = 0 if ("-SEN" in rid or rid.endswith("-AL")) else int(rid.split("-")[1])
        hp = HIST_POLLS[(HIST_POLLS.year == T) & (HIST_POLLS.state == st) & (HIST_POLLS.district == dist)]
        h2h, nh2h = poll_avg(hp, _flip(lead), _flip(rival)) if len(hp) else (np.nan, 0)
        if nh2h:
            poll, npol = h2h, nh2h
        elif lead in avg and rival in avg:
            a, b = avg[lead][0], avg[rival][0]
            poll = 100 * (a - b) / (a + b)
            npol = min(avg[lead][1], avg[rival][1])
        else:
            poll, npol = np.nan, 0
        if npol:
            sp2 = m.tau_ ** 2 + S1 ** 2 / npol
            prec = 1 / sd0 ** 2 + 1 / sp2
            mu = (mu0 / sd0 ** 2 + poll / sp2) / prec
            sd = np.sqrt(1 / prec)
        else:
            mu, sd = mu0, sd0
        p_lead = float(np.clip(norm.cdf(mu / sd), 1 - QCAP, QCAP)) if (mu0 != 0 or lead in avg) else 1 / len(members)
        if rid.startswith("LA-") and lead == inc and not npol:
            # jungle round, incumbent vs unpolled same-party also-rans: the CA/WA finalist-based
            # incumbent prior is far too pessimistic; use the earlier LA jungle record instead (v3)
            p_lead = jungle_incumbent_rate(T)[0]
        q[lead] = pq * p_lead
        for n in rivals:
            q[n] = pq * (1 - p_lead) / len(rivals)
    for n in minors:
        q[n] = MINOR_TOTAL / len(minors) + party_q.get("OTH", 0.0) * scale / len(minors)
    tot = sum(q.values())
    party = {n: p for n, p in cands}
    return [dict(candidate_name=n, wiki_name=n, party=party[n], q=v / tot) for n, v in q.items()]


HIST_POLLS = None


def multi_backtest(T, party_pred, m):
    """party_pred: the main model's backtest rows for year T (race_id in 538 form, q_dem/q_rep/q_other)."""
    hr = pd.read_csv(FEAT / "history_races.csv")
    hr = hr[hr.year == T]
    # the referee's list (the harness's same-party races outside the CA/WA top-two set); a scan of
    # MIT ballots for two same-party names also catches unflagged write-ins (AZ 2022), so it is not used
    ids = set(REFEREE_LIST.get(T, []))
    global HIST_POLLS
    if HIST_POLLS is None:
        HIST_POLLS = hist_polls()
    rows = []
    pp = party_pred.set_index("race_id")
    for rid in sorted(ids):
        x = hr[hr.race_id == rid].iloc[0]
        r538 = rid.replace("-SEN-SPECIAL", "-SEN-S")
        if r538 not in pp.index:
            continue
        pq = dict(DEM=float(pp.loc[r538, "q_dem"]), REP=float(pp.loc[r538, "q_rep"]),
                  OTH=float(pp.loc[r538, "q_other"]))
        inc = x.incumbent_name if (isinstance(x.incumbent_name, str) and x.incumbent_running == 1) else None
        winner = str(x.winner_name)
        for r in multi_candidate(T, rid, pq, inc, m):
            rows.append(dict(year=T, race_id=rid, **r, mu=np.nan, sd=np.nan,
                             won=int(surname(_flip(r["candidate_name"])) == surname(winner))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2026
def _names(row):
    """(cand1, cand2): the listed party nominee first, the other_major_candidate second."""
    oth = row.other_major_candidate
    oth = oth.split("(")[0].strip() if isinstance(oth, str) else None
    first = row.dem_candidate if isinstance(row.dem_candidate, str) else row.rep_candidate
    return first.split(";")[0].strip(), oth


def polls_2026_rows():
    p = pd.read_csv(FEAT / "polls_2026.csv")
    ed = pd.Timestamp(ELECTION_DATES[2026])
    p["days"] = (ed - pd.to_datetime(p.end_date)).dt.days
    out = []
    for r in p.itertuples():
        cp = P._parse_cand_pcts(r.cand_pcts)
        if len(cp) < 2:
            continue
        items = sorted(cp.items(), key=lambda kv: -kv[1][1])[:2]
        out.append(dict(race_id=r.race_id, pollster=r.pollster, days=r.days, n=r.sample_size,
                        c1=items[0][0], v1=items[0][1][1], c2=items[1][0], v2=items[1][1][1]))
    return pd.DataFrame(out)


def predict_2026(df):
    data = hist_dataset()
    m = SamePartyModel().fit(data)
    prim = topttwo.primaries_2026()
    pp = polls_2026_rows()
    rows = []
    for x in df.itertuples():
        n1, n2 = _names(x)
        pr = prim[(prim.state == x.state) & (prim.district == int(x.district))].to_dict("records")
        r1, r2 = _find(pr, n1), _find(pr, n2)
        p1 = r1["pct"] if r1 else np.nan
        p2 = r2["pct"] if r2 else np.nan
        i1 = r1["incumbent"] if r1 else 0
        i2 = r2["incumbent"] if r2 else 0
        if p2 == p2 and p1 == p1 and p2 > p1:     # orient on the primary leader, like the training data
            n1, n2, p1, p2, i1, i2 = n2, n1, p2, p1, i2, i1
        pm = 100 * (p1 - p2) / (p1 + p2) if p1 == p1 and p2 == p2 else 0.0
        pol, npol = poll_avg(pp[pp.race_id == x.race_id], n1, n2)
        mu0, mu, sd, pc1 = m.predict(pm, i1 - i2, pol, npol)
        rows.append(dict(race_id=x.race_id, sp_name1=n1, sp_name2=n2, sp_prim1=p1, sp_prim2=p2, sp_prim_margin=pm,
                         sp_inc1=i1, sp_inc2=i2, sp_mu0=mu0, sp_poll=pol, sp_npollsters=npol, sp_mu=mu, sp_sd=sd,
                         sp_p1=pc1))
    out = pd.DataFrame(rows)
    out.attrs["model"] = m.summary()
    return out


if __name__ == "__main__":
    d = hist_dataset()
    d.to_csv(WORK / "sameparty_hist.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.round(1).to_string())
    for T in (2018, 2022, 2026):
        print(T, SamePartyModel().fit(d[d.year < T]).summary())
