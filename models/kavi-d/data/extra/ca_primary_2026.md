# ca_primary_2026.csv: California June 2, 2026 top-two primary results by district

Built by `src/models/kavi-d/fetch_ca_primary.py` (kavi-d). Pulled 2026-11-01 from the MediaWiki parse API:
- "2026 United States House of Representatives elections in California (districts 1–26)", revid 1383079181
- "2026 United States House of Representatives elections in California (districts 27–52)", revid 1383079672
Raw page JSON is cached in `data/extra/raw/` (not committed; the script re-downloads it if missing).

Columns: race_id (CA-NN, 2026 lines), candidate (as Wikipedia prints it, "(incumbent)" removed), party (Wikipedia
label), votes, pct (share of the primary vote). 297 rows, all 52 districts. Parsed from each district's
"Primary results" table (the text between "Primary results" and "Total votes").

Use in kavi-d: only for the 9 same-party general elections (CA-04/07/11/12/14/29/34/37/40), as a hand-weighted term
(0.04 logit per point above the pair mean) in the within-party candidate split. Not used for any D-vs-R race and not
available for the backtests (no historical primary file), so its weight is not fitted.

Caveats: Wikipedia is crowd-edited; the counts were not checked against the CA Secretary of State certified
Statement of Vote. Spot check: CA-34 Gomez 54,993 (45.99%) / Gonzales-Torres 36,708 (30.70%) as printed. A
primary share is not a general-election share: voters of eliminated candidates (often the other party) decide
same-party generals (e.g. CA-40, where Calvert led Kim 34.9 to 20.6 but Democrats make up ~30% of the primary vote).
