"""Download the Wikipedia article pages with past California/Washington top-two primary results.

Only /wiki/ article URLs (allowed by robots.txt for generic agents), generic User-Agent, one
request every 2 seconds, cached on disk so it runs once.
"""
import time
import urllib.parse
import urllib.request

from common import ROOT

UA = "ModelCitizensResearch/1.0 (student election-forecast project)"
RAWDIR = ROOT / "data" / "raw" / "wiki"
TITLES = [f"{y}_United_States_House_of_Representatives_elections_in_{st}"
          for y in range(2012, 2026, 2) for st in ("California", "Washington")]
TITLES += ["2016_United_States_Senate_election_in_California", "2018_United_States_Senate_election_in_California"]


def fetch(title):
    RAWDIR.mkdir(parents=True, exist_ok=True)
    path = RAWDIR / f"{title}.html"
    if path.exists() and path.stat().st_size > 10000:
        return path
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())
    time.sleep(2)
    return path


if __name__ == "__main__":
    for t in TITLES:
        try:
            p = fetch(t)
            print(t, p.stat().st_size)
        except Exception as e:  # noqa: BLE001
            print("FAILED", t, e)
