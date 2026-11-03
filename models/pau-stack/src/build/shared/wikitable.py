"""Small helpers to turn Wikipedia HTML tables into grids (rowspan/colspan expanded)."""
import re

from bs4 import BeautifulSoup

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
STATE_RE = re.compile("|".join(sorted(STATES, key=len, reverse=True)))


def clean(el):
    """Visible text of a cell, without footnote markers and hidden sort keys."""
    for x in el.find_all(["sup"]):
        x.decompose()
    for x in el.find_all(style=re.compile(r"display:\s*none")):
        x.decompose()
    for x in el.find_all(class_=["sortkey", "mw-ref", "reference"]):
        x.decompose()
    t = el.get_text(" ", strip=True)
    t = t.replace("\xa0", " ").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", " ", t).strip()


def grid(table):
    """Return list of rows; each row a list of (text, is_header, cell_element)."""
    out = []
    pending = {}  # col -> (remaining_rows, value)
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        row = []
        col = 0
        cells = [c for c in tr.find_all(["th", "td"], recursive=False)]
        ci = 0
        while ci < len(cells) or col in pending:
            if col in pending:
                n, val = pending[col]
                row.append(val)
                if n <= 1:
                    del pending[col]
                else:
                    pending[col] = (n - 1, val)
                col += 1
                continue
            c = cells[ci]
            ci += 1
            raw = c  # keep the element for links / colours
            txt = clean(BeautifulSoup(str(c), "html.parser"))
            val = (txt, c.name == "th", raw)
            try:
                cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
            except ValueError:
                cs = 1
            try:
                rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            except ValueError:
                rs = 1
            for k in range(cs):
                row.append(val)
                if rs > 1:
                    pending[col] = (rs - 1, val)
                col += 1
        out.append(row)
    return out


def soup(path):
    return BeautifulSoup(open(path, encoding="utf-8").read(), "html.parser")


def parse_district_label(text):
    """'Arizona 1' / 'Alaska at-large' / 'California 52nd' -> ('AZ', 1) / ('AK', 0)."""
    t = text.replace("\xa0", " ")
    m = STATE_RE.search(t)
    if not m:
        return None
    st = STATES[m.group(0)]
    rest = t[m.end():]
    if re.search(r"at[- ]large|AL\b", rest, re.I):
        return st, 0
    d = re.search(r"(\d+)", rest)
    if not d:
        return None
    return st, int(d.group(1))
