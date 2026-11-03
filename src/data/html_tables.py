"""Minimal stdlib HTML table extractor (handles rowspan/colspan, nested tables skipped).

Used by fetch_ratings.py because the project venv has no lxml/bs4. Each table becomes a list of rows; each row is a
list of cell texts, with spanned cells repeated so every row has the same column positions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

SKIP_TAGS = {"style", "script", "sup"}
WS = re.compile(r"\s+")


@dataclass
class _Cell:
    text: list[str] = field(default_factory=list)
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass
class RawTable:
    rows: list[list[_Cell]] = field(default_factory=list)
    offset: int = 0  # character offset of <table> in the source (to find the enclosing section)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[RawTable] = []
        self._stack: list[RawTable] = []
        self._row: list[list[_Cell]] = []
        self._cell: list[_Cell | None] = []
        self._skip_depth = 0
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        style = (a.get("style") or "").replace(" ", "")
        if "display:none" in style and tag == "span":
            self._hidden_depth += 1
        if tag == "table":
            self._stack.append(RawTable(offset=self.getpos()[1]))
            self._row.append([])
            self._cell.append(None)
        elif not self._stack:
            return
        elif tag == "tr":
            self._row[-1] = []
        elif tag in ("td", "th"):
            cell = _Cell(rowspan=_int(a.get("rowspan")), colspan=_int(a.get("colspan")), is_header=tag == "th")
            self._cell[-1] = cell
            self._row[-1].append(cell)
        elif tag == "br" and self._cell and self._cell[-1] is not None:
            self._cell[-1].text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "span" and self._hidden_depth:
            self._hidden_depth -= 1
        if not self._stack:
            return
        if tag == "table":
            table = self._stack.pop()
            self._row.pop()
            self._cell.pop()
            self.tables.append(table)
        elif tag == "tr":
            if self._row[-1]:
                self._stack[-1].rows.append(self._row[-1])
            self._row[-1] = []
        elif tag in ("td", "th"):
            self._cell[-1] = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._hidden_depth or not self._stack:
            return
        cell = self._cell[-1]
        if cell is not None:
            cell.text.append(data)


def _int(value: str | None) -> int:
    try:
        return max(1, int(re.sub(r"[^0-9]", "", value or "1") or 1))
    except ValueError:
        return 1


def _expand(table: RawTable) -> list[list[str]]:
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    for r, cells in enumerate(table.rows):
        row: list[str] = []
        col = 0
        queue = list(cells)
        while queue or (r, col) in pending:
            if (r, col) in pending:
                row.append(pending.pop((r, col)))
                col += 1
                continue
            cell = queue.pop(0)
            text = WS.sub(" ", "".join(cell.text)).strip()
            for dc in range(cell.colspan):
                row.append(text)
                for dr in range(1, cell.rowspan):
                    pending[(r + dr, col + dc)] = text
            col += cell.colspan
        grid.append(row)
    return grid


def extract_tables(html: str) -> list[list[list[str]]]:
    """Return every table (outermost and nested, in closing order) as a grid of cell strings."""
    parser = _TableParser()
    parser.feed(html)
    return [_expand(t) for t in parser.tables]
