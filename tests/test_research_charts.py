"""The shared visual basis cannot hide entry fees or unmatched study dates."""
import numpy as np
import pandas as pd
import pytest

from dashboard.research_charts import ChartDataError, comparison_bars, wealth_chart


def curves():
    rows = []
    for arm, label, wealth in [("B0", "同曝險等權", [99.8, 110., 108.]),
                              ("model", "凍結模型", [99.9, 115., 113.]),
                              ("price", "價格模型", [99.9, 111., 109.])]:
        wealth = np.array(wealth)
        drawdown = (wealth / np.maximum.accumulate(np.r_[100., wealth])[1:] - 1) * 100
        rows.extend({"Date": day, "Arm": arm, "Display_Name": label, "Wealth_Index": float(value),
                     "Drawdown_Pct": float(dd)} for day, value, dd in zip(pd.date_range("2026-11-06", periods=3), wealth, drawdown))
    return pd.DataFrame(rows)


def plotted_rows(chart):
    spec = chart.to_dict()
    return next(records for records in spec["datasets"].values()
                if records and "Wealth_Index" in records[0])


def test_shared_chart_keeps_entry_fee_and_uses_relative_wealth_not_roi_difference():
    source = curves()
    before = source.copy(deep=True)
    chart = wealth_chart(source, "相對基準", ["model", "B0"], "B0", benchmark_label="同曝險等權")
    rows = pd.DataFrame(plotted_rows(chart))
    model = rows[rows.Arm.eq("model")].sort_values("Date")
    assert model.Wealth_Index.iloc[0] == 99.9
    assert model.Relative_Wealth_Pct.iloc[-1] == pytest.approx((113 / 108 - 1) * 100)
    assert model.Relative_Wealth_Pct.iloc[-1] != pytest.approx(13 - 8)
    assert model.Drawdown_Pct.iloc[0] == pytest.approx(-.1)
    pd.testing.assert_frame_equal(source, before)
    spec = chart.to_dict()
    assert spec["height"] == 330 and spec["width"] == "container"
    palette = spec["layer"][1]["encoding"]["color"]["scale"]
    assert palette == {"domain": ["凍結模型", "同曝險等權"], "range": ["#2563eb", "#7b8799"]}
    assert "價格模型" not in palette["domain"]


@pytest.mark.parametrize("change", ["duplicate", "missing-date", "different-period", "nan", "bad-drawdown"])
def test_invalid_comparisons_never_drop_or_rebase_rows(change):
    frame = curves()
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif change == "missing-date":
        frame = frame.drop(frame[frame.Arm.eq("model")].index[0])
    elif change == "different-period":
        frame.loc[frame.Arm.eq("model"), "Date"] += pd.Timedelta(days=5)
    elif change == "nan":
        frame.loc[frame.Arm.eq("model"), "Wealth_Index"] = np.nan
    else:
        frame.loc[frame.Arm.eq("model"), "Drawdown_Pct"] = 0.
    with pytest.raises(ChartDataError):
        wealth_chart(frame, "成本後淨值", ["model", "B0"], "B0")


def test_all_three_views_share_complete_points_and_benchmark_reference():
    for view, field in [("成本後淨值", "Wealth_Index"), ("相對基準", "Relative_Wealth_Pct"), ("回撤路徑", "Drawdown_Pct")]:
        chart = wealth_chart(curves(), view, ["B0", "model", "price"], "B0")
        assert len(plotted_rows(chart)) == 9
        assert chart.to_dict()["layer"][1]["encoding"]["y"]["field"] == field
    with pytest.raises(ChartDataError):
        wealth_chart(curves(), "未知", ["model"], "B0")


def test_bars_preserve_signed_values_and_require_complete_unique_classifications():
    frame = pd.DataFrame({"Measure": ["MAE", "MAE"], "Forecast": ["凍結模型", "零預測基準"], "Value": [4.7, 4.8]})
    chart = comparison_bars(frame, "Measure", "Value", "同列預測誤差", series="Forecast", unit="百分點")
    spec = chart.to_dict()
    assert spec["encoding"]["xOffset"]["field"] == "Forecast"
    assert "百分點" in spec["encoding"]["y"]["title"]
    with pytest.raises(ChartDataError):
        comparison_bars(pd.concat([frame, frame.iloc[:1]]), "Measure", "Value", "誤差", series="Forecast")
    frame.loc[0, "Value"] = np.inf
    with pytest.raises(ChartDataError):
        comparison_bars(frame, "Measure", "Value", "誤差", series="Forecast")


def test_actual_and_target_holdings_have_distinct_colors_with_unchanged_semantics():
    frame = pd.DataFrame({
        "股票": ["2330", "2330", "2330", "2330"],
        "類型": ["末日實際持倉", "最新訊號目標", "新聞模型", "價格控制"],
        "權重": [21.4, 19.8, 20.0, 20.0],
    })
    chart = comparison_bars(frame, "股票", "權重", "持倉與目標", series="類型", unit="%")
    scale = chart.to_dict()["encoding"]["color"]["scale"]
    colors = dict(zip(scale["domain"], scale["range"]))
    assert colors["末日實際持倉"] != colors["最新訊號目標"]
    assert colors["新聞模型"] == "#2563eb"
    assert colors["價格控制"] == "#c48b29"
    assert chart.to_dict()["encoding"]["xOffset"]["field"] == "類型"


def test_explicit_benchmark_is_gray_while_other_equal_weight_control_stays_purple():
    frame = pd.DataFrame({"組別": ["凍結模型", "相同政策等權", "另一等權控制"], "成本": [71.2, 71.2, 72.0]})
    chart = comparison_bars(frame, "組別", "成本", "成本", benchmark_label="相同政策等權")
    scale = chart.to_dict()["encoding"]["color"]["scale"]
    colors = dict(zip(scale["domain"], scale["range"]))
    assert colors == {"凍結模型": "#2563eb", "相同政策等權": "#7b8799", "另一等權控制": "#8063c5"}
    default = comparison_bars(frame, "組別", "成本", "成本").to_dict()["encoding"]["color"]["scale"]
    assert dict(zip(default["domain"], default["range"]))["相同政策等權"] == "#8063c5"
