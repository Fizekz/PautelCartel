"""Parse expert-rating tables (Cook, Rothenberg / Inside Elections, Sabato) out of Wikipedia wikitext revisions.
Output rows: race_id, outlet, raw text, numeric (Safe D -3 .. Tossup 0 .. Safe R +3, Tilt +-0.5; I counts as D),
following the shared data layer's convention (history_races.csv *_final_num)."""
import re

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"}
ABBR = set(STATES.values())
AL = {"AK", "DE", "MT", "ND", "SD", "VT", "WY"}
LV = {"safe": 3, "solid": 3, "likely": 2, "favored": 2, "lean": 1, "leans": 1, "tilt": 0.5, "tilts": 0.5}


def rating_num(cell):
    """Return (canonical, numeric) for a rating cell, else (None, None)."""
    c = cell
    m = re.search(r"\{\{\s*USRaceRating\s*\|\s*([A-Za-z]+)\s*(?:\|\s*([A-Za-z]+))?", c)
    if m:
        lvl, side = m.group(1).lower(), (m.group(2) or "").upper()
        if lvl in ("tossup", "toss-up"):
            return "Tossup", 0.0
        if lvl in LV and side[:1] in ("D", "R", "I"):
            v = LV[lvl]
            return f"{lvl.title()} {side[:1]}", float(v if side[:1] == "R" else -v)
        return None, None
    t = re.sub(r"\{\{\s*(?:sort|sortname|nts|hs)\s*\|[^|{}]*\|([^{}]*)\}\}", r" \1 ", c, flags=re.I)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"(style|class|bgcolor|align|data-sort-value)\s*=\s*\"?[^|\"]*\"?\s*\|?", " ", t)
    t = t.replace("[[", "").replace("]]", "").strip(" |!")
    low = t.lower()
    m = re.search(r"(tossup|toss-up|toss up)\s*/?\s*(tilts?)\s*(d|r|dem|rep|democrat|republican)\b", low)
    if m:
        s = m.group(3)[0].upper()
        return f"Tilt {s}", 0.5 if s == "R" else -0.5
    m = re.search(r"\b(safe|solid|likely|favored|leans?|tilts?)\s+(d|r|i|dem|rep|democratic|republican|independent)\b", low)
    if m:
        s = m.group(2)[0].upper()
        v = LV[m.group(1)]
        return f"{m.group(1).rstrip('s').title()} {s}", float(v if s == "R" else -v)
    if re.search(r"\b(pure\s+)?(tossup|toss-up|toss up)\b", low):
        return "Tossup", 0.0
    return None, None


def outlet_of(h):
    hl = h.lower()
    if "pvi" in hl or "partisan voting" in hl:
        return None
    if "cook" in hl:
        return "cook"
    if "sabato" in hl or "crystal ball" in hl:
        return "sabato"
    if "rothenberg" in hl or "inside elections" in hl or re.search(r"\|\s*ie\s*\]\]|\bie\b", hl) or "roth." in hl:
        return "inside"
    return "other"


def tables(text):
    """Line-based: a table opens on a line starting with '{|' and closes on a line starting with '|}'
    (so template endings such as '|}}' inside cells do not close it)."""
    out, cur, depth = [], [], 0
    for ln in text.split("\n"):
        s = re.sub(r"^\s*(\{\{[^{}]*\}\}\s*)*", "", ln)
        if s.startswith("{|"):
            depth += 1
        if depth > 0:
            cur.append(ln)
        if s.startswith("|}") and depth > 0:
            depth -= 1
            if depth == 0:
                out.append("\n".join(cur)); cur = []
    return out


def split_cells(line):
    s = line[1:]
    parts = re.split(r"\|\||!!", s)
    cells = []
    for p in parts:
        m = re.match(r"\s*((?:colspan|rowspan|style|class|scope|align|bgcolor|data-sort-value)[^|]*)\|(?!\|)(.*)$", p, flags=re.S)
        attr, val = (m.group(1), m.group(2)) if m else ("", p)
        cs = re.search(r"colspan\s*=\s*\"?(\d+)", attr)
        rs = re.search(r"rowspan\s*=\s*\"?(\d+)", attr)
        cells.append(dict(v=val.strip(), cs=int(cs.group(1)) if cs else 1, rs=int(rs.group(1)) if rs else 1))
    return cells


def row_race(rowtext, senate, year):
    m = re.search(r"ushr\|([A-Z]{2})\|([A-Za-z0-9]+)", rowtext)
    if m and not senate:
        st, d = m.group(1), m.group(2)
        return f"{st}-AL" if (d.upper() in ("AL", "AT-LARGE", "0") or st in AL) else f"{st}-{int(re.sub(r'[^0-9]', '', d) or 0):02d}"
    m = re.search(r"\[\[([A-Za-z ]+?)'s ([A-Za-z0-9\-]+) congressional district", rowtext)
    if m and not senate and m.group(1).strip() in STATES:
        st = STATES[m.group(1).strip()]
        dd = m.group(2)
        if "large" in dd.lower() or st in AL:
            return f"{st}-AL"
        n = re.sub(r"[^0-9]", "", dd)
        return f"{st}-{int(n):02d}" if n else None
    m = re.search(r"\b([A-Z]{2})[- ](\d{1,2}|AL)\b", rowtext)
    if m and not senate and m.group(1) in ABBR:
        st, d = m.group(1), m.group(2)
        return f"{st}-AL" if (d == "AL" or st in AL) else f"{st}-{int(d):02d}"
    if senate:
        head = rowtext[:400]
        for name, ab in sorted(STATES.items(), key=lambda kv: -len(kv[0])):
            if re.search(r"\b" + name + r"\b", head):
                sp = bool(re.search(r"special|\(S\)|\bclass\s*(i|ii|iii)\b", head, flags=re.I))
                return f"{ab}-SEN-SPECIAL" if sp else f"{ab}-SEN"
    return None


def parse(text, senate, year):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\n[ \t]+([|!])", r"\n\1", text)
    rows = []
    for tb in tables(text):
        lines = tb.split("\n")
        # header: the LAST header-only row before the first data row (handles grouped two-row headers)
        hdr = []
        for rt in re.split(r"\n\|-", "\n".join(lines[1:])):
            cl = [ln for ln in rt.split("\n") if (ln.startswith("|") or ln.startswith("!")) and not ln.startswith("|-") and not ln.startswith("|+")]
            if not cl:
                continue
            if all(ln.startswith("!") for ln in cl):
                h = []
                for ln in cl:
                    for c in split_cells(ln):
                        if c["rs"] >= 50:
                            continue
                        h += [c["v"]] * c["cs"]
                if len(h) >= 3:
                    hdr = h
                continue
            break
        outs, seen = [], set()
        for hh in hdr:
            o = outlet_of(hh)
            if o in seen:
                o = "other"
            if o in ("cook", "sabato", "inside"):
                seen.add(o)
            outs.append(o)
        if not any(o in ("cook", "sabato", "inside") for o in outs):
            continue
        body = tb.split("\n|-")[1:]
        carry = {}
        for rt in body:
            cells = []
            for ln in rt.split("\n"):
                if ln.startswith("|}") or ln.startswith("|+"):
                    continue
                if ln.startswith("|") or ln.startswith("!"):
                    cells += split_cells(ln)
                elif cells and ln.strip():
                    cells[-1]["v"] += "\n" + ln
            if not cells or all(c["v"].startswith(("Cook", "Sabato")) for c in cells[:1]):
                continue
            rid = row_race(rt, senate, year)
            if rid is None:
                continue
            # expand colspans and rowspan carries
            flat, k, ci = [], 0, 0
            while ci < len(cells) or k in carry:
                if k in carry and carry[k][1] > 0:
                    flat.append(carry[k][0]); carry[k] = (carry[k][0], carry[k][1] - 1); k += 1; continue
                if ci >= len(cells):
                    break
                c = cells[ci]; ci += 1
                for _ in range(c["cs"]):
                    if c["rs"] > 1:
                        carry[k] = (c["v"], c["rs"] - 1)
                    flat.append(c["v"]); k += 1
            for col, o in enumerate(outs):
                if o in ("cook", "sabato", "inside") and col < len(flat):
                    can, num = rating_num(flat[col])
                    if num is not None:
                        rows.append(dict(year=year, race_id=rid, outlet=o, text=can, num=num))
    return rows
