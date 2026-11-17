"""Shared, read-only charts for paired research books.

Inputs are already derived wealth indexes, never raw provider quotes. Keeping
the common initial capital at 100 preserves entry costs and makes relative
wealth different from a simple difference between ROI percentages.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

VIEWS = ("成本後淨值", "相對基準", "回撤路徑")
MODEL = "#2563eb"
BENCHMARK = "#7b8799"
PRICE = "#c48b29"
EQUAL = "#8063c5"


class ChartDataError(ValueError):
    """A visual comparison must not silently repair or discard evidence."""


def _color(arm, name, benchmark_arm=None, *, fallback=MODEL):
    if arm == benchmark_arm:
        return BENCHMARK
    text = f"{arm} {name}".lower()
    if "price" in text or "價格" in text:
        return PRICE
    if "equal" in text or "等權" in text:
        return EQUAL
    if "基準" in text or "0%" in text or "zero" in text:
        return BENCHMARK
    if any(token in text for token in ("news", "新聞", "model", "模型")):
        return MODEL
    return fallback


def _wealth_frame(curves, arms, benchmark_arm):
    required = {"Date", "Arm", "Display_Name", "Wealth_Index", "Drawdown_Pct"}
    if not isinstance(curves, pd.DataFrame) or not required.issubset(curves):
        raise ChartDataError("財富圖表缺少日期、組別、名稱、淨值或回撤欄位")
    arms = list(arms)
    if not arms or len(arms) != len(set(arms)):
        raise ChartDataError("請選擇不同且非空的比較組")
    involved = list(dict.fromkeys([benchmark_arm, *arms]))
    if not set(involved).issubset(set(curves.Arm)):
        raise ChartDataError("所選比較組或基準缺少已保存資料")
    work = curves.loc[curves.Arm.isin(involved), list(required)].copy()
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(work.Date, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ChartDataError("圖表日期無法解析") from exc
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ChartDataError("圖表日期必須是完整、無時區的市場日期")
    work["Date"] = dates
    if work.duplicated(["Date", "Arm"]).any():
        raise ChartDataError("同一比較組有重複日期")
    try:
        values = work[["Wealth_Index", "Drawdown_Pct"]].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ChartDataError("圖表淨值或回撤含非數值") from exc
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any():
        raise ChartDataError("圖表淨值必須有限且為正，回撤不得缺值")
    work[["Wealth_Index", "Drawdown_Pct"]] = values
    names = {}
    expected_dates = None
    for arm in involved:
        part = work.loc[work.Arm.eq(arm)].sort_values("Date")
        actual_dates = pd.DatetimeIndex(part.Date)
        if expected_dates is not None and not actual_dates.equals(expected_dates):
            raise ChartDataError("比較組日期未完整配對；不可混合不同期間或刪除缺值")
        expected_dates = actual_dates
        if part.Display_Name.isna().any() or part.Display_Name.nunique() != 1:
            raise ChartDataError("每個比較組必須有唯一顯示名稱")
        name = part.Display_Name.iloc[0]
        if not isinstance(name, str) or not name.strip() or name in names.values():
            raise ChartDataError("比較組顯示名稱必須非空且互不相同")
        names[arm] = name
        wealth = part.Wealth_Index.to_numpy(float)
        peak = np.maximum.accumulate(np.r_[100., wealth])[1:]
        expected_drawdown = (wealth / peak - 1) * 100
        if not np.allclose(part.Drawdown_Pct, expected_drawdown, atol=1e-7, rtol=0):
            raise ChartDataError("回撤必須由相同淨值及初始資金100計算，包含進場成本")
    benchmark = work.loc[work.Arm.eq(benchmark_arm), ["Date", "Wealth_Index"]].rename(
        columns={"Wealth_Index": "Benchmark_Wealth"})
    # Date equality was checked above; merging cannot delete unmatched rows.
    work = work.merge(benchmark, on="Date", how="left", validate="many_to_one")
    work["Relative_Wealth_Pct"] = (work.Wealth_Index / work.Benchmark_Wealth - 1) * 100
    return work.loc[work.Arm.isin(arms)].sort_values(["Date", "Arm"]), arms, names


def wealth_chart(curves, view, arms, benchmark_arm, *, benchmark_label="基準"):
    """One common-period chart with the same units, palette and hover for V2/V3.

    ``Wealth_Index`` equals saved NAV divided by common initial capital times 100.
    It is never divided by the first observed NAV. All requested books must
    have complete identical dates and drawdowns that include initial wealth 100.
    """
    if view not in VIEWS:
        raise ChartDataError("未知的財富圖表視角")
    frame, selected, names = _wealth_frame(curves, arms, benchmark_arm)
    field, title = {
        "成本後淨值": ("Wealth_Index", "淨值指數（起始資金＝100）"),
        "相對基準": ("Relative_Wealth_Pct", f"相對{benchmark_label}財富增長（%）"),
        "回撤路徑": ("Drawdown_Pct", "回撤（%）"),
    }[view]
    domain = [names[arm] for arm in selected]
    palette = alt.Scale(domain=domain, range=[_color(arm, names[arm], benchmark_arm) for arm in selected])
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(frame).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%m/%d", grid=False)),
        y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("Display_Name:N", title=None, scale=palette,
                        legend=alt.Legend(orient="bottom", labelLimit=220)),
        tooltip=[alt.Tooltip("Date:T", title="日期", format="%Y-%m-%d"),
                 alt.Tooltip("Display_Name:N", title="組別"),
                 alt.Tooltip(f"{field}:Q", title=title, format=".3f")])
    line = base.mark_line(strokeWidth=2.4)
    points = base.mark_circle(size=55).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))).add_params(hover)
    reference = alt.Chart(pd.DataFrame({"value": [100 if view == "成本後淨值" else 0]})).mark_rule(
        color="#bdc8d7", strokeDash=[4, 5]).encode(y="value:Q")
    return (reference + line + points).properties(width="container", height=330).configure_view(
        strokeWidth=0).configure_axis(labelColor="#6b7a90", titleColor="#6b7a90",
                                     gridColor="#e9edf4", titleFontWeight="normal")


def comparison_bars(frame, category, value, title, *, series=None, unit="", height=250,
                    benchmark_label=None):
    """Small native comparison bars; duplicate or incomplete keys are rejected.

    With ``series``, bars are grouped side by side, not stacked. Different
    measures must already have a shared unit before callers put them together.
    An explicit benchmark label overrides other semantics. Unclassified labels
    use distinct fallback colors while retaining known model/control colors.
    """
    fields = list(dict.fromkeys([category, value, *([series] if series else [])]))
    if not isinstance(frame, pd.DataFrame) or frame.empty or not set(fields).issubset(frame):
        raise ChartDataError("長條比較缺少完整資料")
    work = frame[fields].copy()
    keys = [category, *([series] if series else [])]
    if work[keys].isna().any().any() or work.duplicated(keys).any():
        raise ChartDataError("長條比較的分類缺值或鍵重複")
    for key in keys:
        if not work[key].map(lambda item: isinstance(item, str) and bool(item.strip())).all():
            raise ChartDataError("長條分類必須是非空名稱")
    try:
        values = pd.to_numeric(work[value], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ChartDataError("長條比較含非數值") from exc
    if not np.isfinite(values).all():
        raise ChartDataError("長條比較不得含缺值或非有限值")
    work[value] = values
    palette_key = series or category
    names = list(dict.fromkeys(work[palette_key]))
    colors = [_color(name, name, benchmark_label, fallback=None) for name in names]
    color_order = (MODEL, PRICE, EQUAL, BENCHMARK)
    fallbacks = [color for color in color_order if color not in colors]
    fallbacks += [color for color in color_order if color in colors]
    unclassified = 0
    for index, color in enumerate(colors):
        if color is None:
            colors[index] = fallbacks[unclassified % len(fallbacks)]
            unclassified += 1
    palette = alt.Scale(domain=names, range=colors)
    encoding = {
        "x": alt.X(f"{category}:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=200)),
        "y": alt.Y(f"{value}:Q", title=f"{title}（{unit}）" if unit else title, scale=alt.Scale(zero=True)),
        "color": alt.Color(f"{palette_key}:N", title=None, scale=palette,
                           legend=alt.Legend(orient="bottom") if series else None),
        "tooltip": [alt.Tooltip(f"{key}:N", title=key) for key in keys] + [
            alt.Tooltip(f"{value}:Q", title=title, format=".3f")],
    }
    if series:
        encoding["xOffset"] = alt.XOffset(f"{series}:N")
    return alt.Chart(work).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        **encoding).properties(width="container", height=height).configure_view(strokeWidth=0).configure_axis(
            labelColor="#6b7a90", titleColor="#6b7a90", gridColor="#e9edf4", titleFontWeight="normal")
