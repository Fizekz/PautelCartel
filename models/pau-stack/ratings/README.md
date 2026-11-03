# ratings/: election-eve expert ratings (pau-r integrity check)

- `history_ratings_eve.csv`: one row per race-year of the shared `history_races.csv` (2010-2024).
  - Columns: `cook`, `sabato`, `inside` on the layer's scale (Safe D -3 .. Tossup 0 .. Safe R +3, Tilt +-0.5,
    independents as D); `*_src` (`eve_revision` or `layer`); `ratings_imputed` (layer flag); `rating_mean_eve`.
  - Each cell is the outlet's rating as shown on the Wikipedia revision in force on election day where that
    revision's table lists the race. Otherwise it is the data layer's value.
- `ratings_eve_parsed.csv`: every rating parsed from the eve revisions (year, race_id, outlet, text, num).
- `ratings_compare.csv`: eve vs Pau's layer (`layer`) vs Kavi's layer (`kavi`), with the winner.
- `asof/`: the eve revisions' wikitext; `revisions.json` gives title, revid and timestamp. All are timestamped 1-2
  days before each election.
  - Pages: the main House elections article (2010-2016), the 2018 ratings/predictions templates, the ratings pages
    (2020-2024), and the Senate elections articles.
- Source: Wikipedia via the MediaWiki API (`prop=revisions`, `rvstart=<election day>T00:00:00Z`, `rvdir=older`).
  Generic UA, at least 6 s between calls, about 45 calls on 2026-11-02 06:00-06:10Z. Content CC BY-SA.
- Leakage: eve values cannot contain post-election edits. Cells without an eve value keep the layer's
  retrospective value (see RESEARCH_RATINGS.md for its small lean toward winners).
- Rebuild offline: `python compare.py` (parses `asof/`, compares), then the snippet in RESEARCH_RATINGS.md section
  5 / `rescore.py`.
