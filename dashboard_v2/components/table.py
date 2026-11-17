"""HTML tables so the page does not depend on pyarrow."""
from __future__ import annotations

import html


def table(rows, columns):
    if not rows:
        return '<div class="gap">DATA_UNAVAILABLE</div>'
    head = "".join(f"<th>{html.escape(str(name))}</th>" for _key, name in columns)
    body = []
    for row in rows:
        cells = []
        for key, _name in columns:
            cells.append(f"<td>{html.escape(str(row.get(key, '')))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def cards(items):
    blocks = []
    for item in items:
        blocks.append(
            '<article class="card"><p>'
            + html.escape(item["label"])
            + '</p><strong>'
            + html.escape(str(item["value"]))
            + '</strong><small>'
            + html.escape(item.get("note", ""))
            + "</small></article>"
        )
    return '<section class="cards">' + "".join(blocks) + "</section>"
