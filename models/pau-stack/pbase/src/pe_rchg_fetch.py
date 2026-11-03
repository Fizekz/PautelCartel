"""pau-e: fetch the expert-rating pages as of E-35 days (same pages as pau-r's election-eve revisions in
ratings/asof/), via the MediaWiki API only: generic UA without email, >= 6 s between calls, maxlag=5, back-off on
429 (curl, since Python's TLS does not pass the sandbox proxy). Output: pbase/data/raw_e/wiki_early/<key>.wikitext
+ revisions_e<days>.json. Content CC BY-SA."""
import json, subprocess, sys, time, urllib.parse
from pathlib import Path
import pandas as pd
from common import ELECTION_DATES  # noqa: E402
R = Path(__file__).resolve().parents[2] / "ratings"
UA = "ModelCitizensResearch/1.0 (student election-forecast project)"
_last = [0.0]


def api(params, gap=6.0):
    params = dict(params, format="json", maxlag=5, formatversion=2)
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    for k in range(5):
        w = _last[0] + gap - time.time()
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        r = subprocess.run(["curl", "-sS", "-m", "90", "-A", UA, "-w", "\n%{http_code}", url], capture_output=True, text=True)
        body, code = r.stdout.rsplit("\n", 1)
        if code == "429" or code.startswith("5"):
            time.sleep(60 * (k + 1)); continue
        d = json.loads(body)
        if "error" in d and d["error"].get("code") == "maxlag":
            time.sleep(10 * (k + 1)); continue
        return d
    raise RuntimeError("gave up")


def rev_asof(title, ts, content=True):
    d = api(dict(action="query", prop="revisions", titles=title, rvlimit=1, rvdir="older", rvstart=ts,
                 rvprop="ids|timestamp|size|content", rvslots="main"))
    pg = d["query"]["pages"][0]
    if "revisions" not in pg:
        return None
    r = pg["revisions"][0]
    return dict(revid=r["revid"], timestamp=r["timestamp"], content=r["slots"]["main"]["content"])
def main():
    OUT = Path(__file__).resolve().parents[1] / "data" / "raw_e" / "wiki_early"
    OUT.mkdir(parents=True, exist_ok=True)
    DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    eve = json.load(open(R / "asof" / "revisions.json"))
    pages = {k: v["title"] for k, v in eve.items()}
    pages["2018_house_template"] = "Template:United States House of Representatives elections, 2018 election ratings"
    pages["2018_senate_template"] = "Template:United States Senate elections, 2018 election predictions"
    revs = {}
    for k, t in sorted(pages.items()):
        y = int(k[:4])
        ts = (pd.Timestamp(ELECTION_DATES[y]) - pd.Timedelta(days=DAYS)).strftime("%Y-%m-%dT00:00:00Z")
        r = rev_asof(t, ts, content=True)
        if r is None:
            print(k, "no revision", flush=True); continue
        (OUT / f"{k}_e{DAYS}.wikitext").write_text(r["content"])
        revs[k] = dict(title=t, revid=r["revid"], timestamp=r["timestamp"])
        print(k, r["revid"], r["timestamp"], len(r["content"]), flush=True)
    json.dump(revs, open(OUT / f"revisions_e{DAYS}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
