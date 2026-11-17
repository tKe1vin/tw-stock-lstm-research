"""Small SVG charts. They do not require pyarrow."""
from __future__ import annotations

COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2"]


def _extent(values):
    low = min(values)
    high = max(values)
    if low == high:
        high = low + 1
    return low, high


def bars(items, title, unit=""):
    if not items:
        return unavailable(title)
    width, height, left, top = 760, 260, 150, 28
    values = [float(item["value"]) for item in items]
    low, high = _extent(values + [0])
    span = high - low
    plot_w = width - left - 20
    gap = plot_w / len(items)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">',
             f'<text x="12" y="18" fill="#d7dee7" font-size="14">{title}</text>']
    zero = top + (high - 0) / span * (height - top - 30)
    parts.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{width-16}" y2="{zero:.1f}" stroke="#667"/>')
    for index, item in enumerate(items):
        value = float(item["value"])
        y = top + (high - value) / span * (height - top - 30)
        bar_h = abs(zero - y)
        y0 = min(y, zero)
        x = left + index * gap + 8
        parts.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{max(gap-16, 8):.1f}" height="{bar_h:.1f}" fill="{COLORS[index % len(COLORS)]}"/>'
        )
        parts.append(f'<text x="{x:.1f}" y="{height-8}" fill="#c5ced8" font-size="11">{item["label"]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y0-4:.1f}" fill="#eef3f8" font-size="11">{value:.2f}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def lines(series, title):
    usable = [item for item in series if item.get("points")]
    if not usable:
        return unavailable(title)
    width, height, left, top = 760, 280, 48, 28
    values = [point[1] for item in usable for point in item["points"]]
    low, high = _extent(values)
    span = high - low
    longest = max(len(item["points"]) for item in usable)
    plot_w = width - left - 16
    plot_h = height - top - 36
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">',
             f'<text x="12" y="18" fill="#d7dee7" font-size="14">{title}</text>']
    for index, item in enumerate(usable):
        coords = []
        for step, (_label, value) in enumerate(item["points"]):
            x = left + (step / max(longest - 1, 1)) * plot_w
            y = top + (high - value) / span * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        color = COLORS[index % len(COLORS)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}"/>')
        parts.append(f'<text x="{left + index * 140}" y="{height-8}" fill="{color}" font-size="11">{item["name"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def unavailable(title):
    return (
        '<div class="gap">DATA_UNAVAILABLE<br>'
        f'{title} 沒有與目前協議對得上的逐日序列。</div>'
    )
