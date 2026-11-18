"""Read-only native views for the separately trained V2.12 historical study.

Only the public, derived report and its model archive are read here. Loading a
page never imports a model runtime, fetches a quote or changes a frozen signal.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import io
import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import ROOT, DataIntegrityError
from dashboard.research_charts import comparison_bars, wealth_chart
from dashboard.research_layout import RESEARCH_TABS

REPORT_DIR = Path("reports/research_v2_common/2026-11-17")
SCHEMA = "v2-common-public-v1"
ARMS = (
    "static_equal444", "dynamic_news_risk555", "v2_news_lstm",
    "v2_no_news_lstm", "v2_price_lstm", "v2_news_no_risk",
    "v2_matched_equal", "full_equal_5",
)
TABLES = (
    "metrics", "curves", "nav", "weights", "fills", "thresholds",
    "prediction_metrics", "prediction_timeline", "paired_effects", "daily_returns",
)
REQUIRED_FILES = {"protocol.json", "summary.json", "summary.md", *[name + ".csv" for name in TABLES]}
GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"


class V2HistoryError(DataIntegrityError):
    """Missing or inconsistent historical evidence must remain visible."""


def _path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise V2HistoryError("歷史來源路徑不安全")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise V2HistoryError(f"缺少歷史來源或路徑不安全：{name}")
    return path


def _json(raw: bytes, name: str) -> dict:
    try:
        result = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise V2HistoryError(f"歷史JSON無法讀取：{name}") from exc
    if not isinstance(result, dict):
        raise V2HistoryError(f"歷史JSON必須是物件：{name}")
    return result


def _finite(frame: pd.DataFrame, columns, name: str) -> None:
    if frame.empty or not set(columns).issubset(frame):
        raise V2HistoryError(f"{name}缺必要欄位或資料")
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            good = frame[column].map(lambda value: math.isfinite(float(value))).all()
        except (ValueError, TypeError, OverflowError) as exc:
            raise V2HistoryError(f"{name}含非數值：{column}") from exc
        if not good:
            raise V2HistoryError(f"{name}含缺值或非有限值：{column}")


def _date_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    if column not in frame or not frame[column].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise V2HistoryError(f"{name}日期格式不完整")
    try:
        result = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise V2HistoryError(f"{name}日期不合法") from exc
    return result


def _close(left, right, message, *, tolerance=1e-6):
    if not math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0):
        raise V2HistoryError(message)


def history_available(root=ROOT) -> bool:
    """Availability chooses the initial scope; validity is checked separately."""
    return (Path(root) / REPORT_DIR / "manifest.json").is_file()


def _validate_predictions(tables, summary, protocol, valuation_dates):
    models = {"price_lstm", "no_news_lstm", "news_lstm"}
    diagnostics, timeline = tables["prediction_metrics"], tables["prediction_timeline"]
    counts = ("Rows", "Signal_Days", "Matured_Rows", "Matured_Signal_Days", "Pending_Rows", "Pending_Signal_Days")
    _finite(diagnostics, (*counts, "MAE_Pp", "RMSE_Pp"), "預測診斷")
    if (not {"Model", "Fold", "Split", "Forecast"}.issubset(diagnostics)
            or set(diagnostics.Model) != models
            or set(diagnostics.Forecast) != {"model", "zero_return_baseline"}
            or diagnostics.duplicated(["Model", "Fold", "Split", "Forecast"]).any()):
        raise V2HistoryError("預測診斷缺完整三候選／基準或有重複")
    _finite(timeline, ("Rows", "Actual_Count"), "預測時間線")
    if (not {"Model", "Fold", "Split", "Signal_Date", "Execution_Date", "Label_End", "Label_State"}.issubset(timeline)
            or set(timeline.Model) != models
            or timeline.duplicated(["Model", "Fold", "Split", "Signal_Date"]).any()):
        raise V2HistoryError("預測時間線缺三候選、日期或有重複")
    for column in ("Signal_Date", "Execution_Date"):
        timeline[column] = _date_column(timeline, column, "預測時間線")
    if not set(timeline.Label_State).issubset({"matured", "pending"}):
        raise V2HistoryError("預測時間線缺明確成熟狀態")
    if set(timeline.Split) != {"validation", "test"} or not set(diagnostics.Split).issubset({"validation", "test"}):
        raise V2HistoryError("預測診斷必須明確區分驗證與測試")
    if not timeline.Rows.eq(5).all() or not timeline.Actual_Count.eq(timeline.Label_State.eq("matured").astype(int) * 5).all():
        raise V2HistoryError("已成熟列須有五筆實際標籤，待成熟不得當成0報酬")
    if not timeline.Signal_Date.lt(timeline.Execution_Date).all():
        raise V2HistoryError("歷史預測進場必須晚於訊號收盤")
    cutoff = pd.Timestamp(summary["period_end"])
    mature = timeline.Label_State.eq("matured")
    label_end = _date_column(timeline.loc[mature], "Label_End", "成熟標籤")
    if (not label_end.le(cutoff).all() or not label_end.gt(timeline.loc[mature, "Execution_Date"]).all()
            or not timeline.loc[timeline.Split.eq("validation"), "Label_State"].eq("matured").all()):
        raise V2HistoryError("預測成熟日期與估值截止日不符")
    pending_ends = pd.to_datetime(timeline.loc[~mature, "Label_End"], errors="coerce")
    if pending_ends.le(cutoff).any():
        raise V2HistoryError("待成熟標籤日期不能早於估值截止日")
    expected_days = {
        "test_signal_days": 150, "matured_signal_days": 145,
        "pending_signal_days": 5, "forecasts_per_model": 750, "fold_count": 3,
    }
    if any(summary.get(key) != value for key, value in expected_days.items()):
        raise V2HistoryError("歷史摘要的訊號／成熟／待成熟日數與登錄150日設計不符")
    test = timeline.loc[timeline.Split.eq("test")]
    shared_signals = protocol.get("shared_signal_dates")
    if not isinstance(shared_signals, list) or len(shared_signals) != 150:
        raise V2HistoryError("歷史協議缺150個共同訊號日期")
    shared_signals = pd.DatetimeIndex(_date_column(pd.DataFrame({"Date": shared_signals}), "Date", "共同訊號"))
    if (shared_signals.duplicated().any() or not shared_signals.is_monotonic_increasing
            or not shared_signals[1:].equals(valuation_dates[:-1])
            or not shared_signals[0] < valuation_dates[0]):
        raise V2HistoryError("共同訊號與次日收盤估值日期不一致")
    expected_dates = None
    expected_folds = None
    for model in sorted(models):
        part = test.loc[test.Model.eq(model)].sort_values("Signal_Date")
        dates = pd.DatetimeIndex(part.Signal_Date)
        if (not dates.equals(shared_signals)
                or not pd.DatetimeIndex(part.Execution_Date).equals(valuation_dates)):
            raise V2HistoryError("預測訊號／進場日期與共同帳本不一致")
        if (len(part) != 150 or dates.duplicated().any()
                or len(part.loc[part.Label_State.eq("matured")]) != 145
                or len(part.loc[part.Label_State.eq("pending")]) != 5
                or part.Fold.nunique() != 3
                or not part.groupby("Fold").size().eq(50).all()):
            raise V2HistoryError("每模型必須保留150訊號／145成熟／5待成熟及三折各50訊號")
        if expected_dates is not None and not dates.equals(expected_dates):
            raise V2HistoryError("三模型測試訊號日期未完整配對")
        assignments = part.Fold.astype(str).tolist()
        if expected_folds is not None and assignments != expected_folds:
            raise V2HistoryError("三模型測試折分配不一致")
        expected_dates = dates
        expected_folds = assignments
        aggregate = diagnostics.loc[diagnostics.Model.eq(model) & diagnostics.Fold.astype(str).eq("all") & diagnostics.Split.eq("test")]
        if len(aggregate) != 2:
            raise V2HistoryError("每模型需保留完整測試彙總與零預測基準")
        for field, value in zip(counts, (750, 150, 725, 145, 25, 5)):
            if not aggregate[field].eq(value).all():
                raise V2HistoryError("預測診斷成熟列數與時間線不一致")
    for (model, fold, split), part in timeline.groupby(["Model", "Fold", "Split"]):
        scored = diagnostics.loc[diagnostics.Model.eq(model) & diagnostics.Fold.astype(str).eq(str(fold)) & diagnostics.Split.eq(split)]
        mature_part = part.loc[part.Label_State.eq("matured")]
        pending_part = part.loc[part.Label_State.eq("pending")]
        values = (part.Rows.sum(), part.Signal_Date.nunique(), mature_part.Rows.sum(),
                  mature_part.Signal_Date.nunique(), pending_part.Rows.sum(), pending_part.Signal_Date.nunique())
        if len(scored) != 2 or any(not scored[field].eq(value).all() for field, value in zip(counts, values)):
            raise V2HistoryError("逐折預測診斷列數與時間線不一致")
    thresholds = tables["thresholds"]
    _finite(thresholds, ("Tau", "Validation_Rows", "Validation_Signal_Days"), "逐折門檻")
    if (not {"Model", "Fold", "Ticker", "Latest_Validation_Label_End", "Test_First_Signal"}.issubset(thresholds)
            or len(thresholds) != 45 or set(thresholds.Model) != models
            or thresholds.duplicated(["Model", "Fold", "Ticker"]).any()
            or not thresholds.Validation_Rows.eq(60).all()
            or not thresholds.Validation_Signal_Days.eq(60).all()):
        raise V2HistoryError("逐折門檻必須保留45組模型／股票及60日驗證")
    last_label = _date_column(thresholds, "Latest_Validation_Label_End", "逐折門檻")
    first_signal = _date_column(thresholds, "Test_First_Signal", "逐折門檻")
    if not last_label.lt(first_signal).all():
        raise V2HistoryError("驗證標籤尚未在測試前成熟，不可使用未來門檻")
    effects = tables["paired_effects"]
    _finite(effects, ("N_Days", "Mean_Daily_Difference_Pp", "CI_Lower_Pp", "CI_Upper_Pp"), "配對效果")
    if ("Contrast" not in effects or set(effects.Contrast) != {"news_vs_no_news", "risk_with_news", "selection_at_same_gross"}
            or effects.Contrast.duplicated().any() or not effects.N_Days.eq(149).all()
            or not effects.CI_Lower_Pp.le(effects.CI_Upper_Pp).all()):
        raise V2HistoryError("配對效果缺共同日期、對照或正確區間")
    daily = tables["daily_returns"]
    _finite(daily, ("Net_Return_Pct",), "配對日報酬")
    if "Arm" not in daily or set(daily.Arm) != set(ARMS):
        raise V2HistoryError("配對日報酬缺八個共同組別")
    daily["Date"] = _date_column(daily, "Date", "配對日報酬")
    nav = tables["nav"]
    for arm in ARMS:
        part = daily.loc[daily.Arm.eq(arm)].sort_values("Date")
        book = nav.loc[nav.Arm.eq(arm)].sort_values("Date")
        if not pd.DatetimeIndex(part.Date).equals(pd.DatetimeIndex(book.Date.iloc[1:])):
            raise V2HistoryError("配對日報酬必須保留149個共同持有期日期")
        previous = book.NAV.iloc[0]
        for saved, value in zip(part.Net_Return_Pct, book.NAV.iloc[1:]):
            _close(saved, (value / previous - 1) * 100, "配對日報酬與NAV不一致")
            previous = value


def load_v2_history(root=ROOT) -> dict:
    """Verify the complete derived report before exposing any historical ROI.

    ``sources`` records the inputs used at export. It is provenance, not a
    promise that a growing live source will forever retain its current hash.
    All report files and the immutable model archive are verified directly.
    """
    root = Path(root).resolve()
    report = _path(root, (REPORT_DIR / "manifest.json").as_posix()).parent
    manifest_path = report / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = _json(manifest_raw, "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise V2HistoryError("V2歷史manifest版本不符")
    files = manifest.get("files")
    if not isinstance(files, dict) or not REQUIRED_FILES.issubset(files):
        raise V2HistoryError("V2歷史manifest缺完整報告清單")
    recorded, contents = {}, {}
    for name, expected in files.items():
        path = _path(report, name)
        digest = expected.get("sha256") if isinstance(expected, dict) else expected
        raw = path.read_bytes()
        if not isinstance(digest, str) or hashlib.sha256(raw).hexdigest() != digest:
            raise V2HistoryError(f"V2歷史報告hash不符：{name}")
        contents[name] = raw
        recorded[path.relative_to(root).as_posix()] = digest
    protocol = _json(contents["protocol.json"], "protocol.json")
    summary = _json(contents["summary.json"], "summary.json")
    if summary.get("study_status") != "historical_exploratory":
        raise V2HistoryError("V2歷史報告必須明示historical_exploratory")
    if (summary.get("benchmark_arm") != "static_equal444"
            or summary.get("model_arm") != "v2_news_lstm"
            or summary.get("matched_arm") != "v2_matched_equal"):
        raise V2HistoryError("V2歷史比較組與登錄協議不符")
    for key in ("period_start", "period_end", "initial_capital"):
        if manifest.get(key) != summary.get(key):
            raise V2HistoryError(f"V2歷史manifest與摘要不一致：{key}")
    try:
        capital = float(summary["initial_capital"])
        sessions = int(summary["valuation_sessions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V2HistoryError("V2歷史摘要缺共同期間或初始資金") from exc
    if capital != 100000 or sessions != 150:
        raise V2HistoryError("V2歷史協議必須保留100000元及150個共同估值日")
    models = manifest.get("models")
    if not isinstance(models, dict) or summary.get("reproducible_model_pack") is not True:
        raise V2HistoryError("V2歷史報告缺新訓練模型包")
    model_path = _path(root, models.get("archive_path"))
    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_digest != models.get("sha256"):
        raise V2HistoryError("V2歷史新訓練模型包hash不符")
    recorded[model_path.relative_to(root).as_posix()] = model_digest
    tables = {}
    for name in TABLES:
        try:
            tables[name] = pd.read_csv(io.BytesIO(contents[name + ".csv"]), encoding="utf-8-sig")
        except (ValueError, OSError, UnicodeError) as exc:
            raise V2HistoryError(f"V2歷史CSV無法讀取：{name}") from exc
    if "news_coverage.csv" in contents:
        try:
            coverage = pd.read_csv(io.BytesIO(contents["news_coverage.csv"]), encoding="utf-8-sig")
        except (ValueError, OSError, UnicodeError) as exc:
            raise V2HistoryError("V2歷史新聞覆蓋CSV無法讀取") from exc
        _finite(coverage, ("Signal_Days", "Stock_Rows", "Nonzero_Scores", "Unknown_Source_Status_Rows",
                           "Observed_Article_Rows", "Verified_No_Article_Rows", "Publication_Timestamp_Rows"), "新聞覆蓋")
        if (len(coverage) != 3 or not coverage.Signal_Days.eq(50).all() or not coverage.Stock_Rows.eq(250).all()
                or (coverage.Unknown_Source_Status_Rows < 0).any()
                or (coverage.Unknown_Source_Status_Rows > coverage.Stock_Rows).any()):
            raise V2HistoryError("新聞覆蓋必須保留完整三折及750個股票觀測列")
        tables["news_coverage"] = coverage
    metrics, curves, nav = (tables[name] for name in ("metrics", "curves", "nav"))
    numeric_metrics = (
        "ROI_Pct", "MDD_Pct", "Sharpe", "Total_Transaction_Cost",
        "Average_Equity_Weight_Pct", "Average_Cash_Weight_Pct", "Rebalance_Days", "Sessions",
        "Gap_Vs_444_Pp", "Gap_Vs_555_Pp", "Gap_Vs_Matched_Equal_Pp",
    )
    _finite(metrics, numeric_metrics, "歷史指標")
    _finite(curves, ("Wealth_Index", "Drawdown_Pct"), "歷史曲線")
    _finite(nav, ("NAV", "Cash", "Equity_Weight", "Transaction_Cost"), "歷史NAV")
    for frame, name in ((metrics, "歷史指標"), (curves, "歷史曲線"), (nav, "歷史NAV")):
        if not {"Arm", "Display_Name"}.issubset(frame) or set(frame.Arm) != set(ARMS):
            raise V2HistoryError(f"{name}必須包含八個登錄組別")
    if (metrics.Arm.duplicated().any() or metrics.Display_Name.duplicated().any()
            or not metrics.Display_Name.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise V2HistoryError("歷史指標組別或顯示名稱重複／缺值")
    for frame, name in ((curves, "歷史曲線"), (nav, "歷史NAV")):
        frame["Date"] = _date_column(frame, "Date", name)
        if frame.duplicated(["Date", "Arm"]).any():
            raise V2HistoryError(f"{name}同組同日重複")
    if (curves.Wealth_Index <= 0).any() or (nav.NAV <= 0).any() or (nav.Cash < -1e-6).any():
        raise V2HistoryError("歷史淨值或現金範圍不合法")
    if ((nav.Equity_Weight < -1e-8) | (nav.Equity_Weight > 1 + 1e-8)).any() or (nav.Transaction_Cost < 0).any():
        raise V2HistoryError("歷史曝險或成本範圍不合法")
    metric = metrics.set_index("Arm")
    dates = None
    for arm in ARMS:
        curve = curves[curves.Arm.eq(arm)].sort_values("Date")
        book = nav[nav.Arm.eq(arm)].sort_values("Date")
        arm_dates = pd.DatetimeIndex(curve.Date)
        if dates is None:
            dates = arm_dates
        if not arm_dates.equals(dates) or not pd.DatetimeIndex(book.Date).equals(dates) or len(curve) != sessions:
            raise V2HistoryError("歷史比較組必須有完整150個共同估值日，不可刪除缺值")
        for part in (curve, book):
            if part.Display_Name.isna().any() or not part.Display_Name.eq(metric.loc[arm, "Display_Name"]).all():
                raise V2HistoryError("歷史顯示名稱與指標不一致")
        for wealth, value in zip(curve.Wealth_Index, book.NAV):
            _close(wealth, value / capital * 100, "歷史曲線不能重設首日基準或偏離NAV")
        _close(book.NAV.iloc[0] + book.Transaction_Cost.iloc[0], capital,
               "歷史首筆進場不得取得訊號日至成交日已完成的報酬")
        for cash, equity_weight, value in zip(book.Cash, book.Equity_Weight, book.NAV):
            _close(cash / value + equity_weight, 1, "歷史現金與股票曝險無法對帳")
        _close(book.Equity_Weight.iloc[-1], 0, "歷史末日須真正清算，不以未平倉報酬替代")
        _close(book.Cash.iloc[-1], book.NAV.iloc[-1], "歷史末日清算後現金與NAV不一致")
        peak, drawdowns = 100., []
        for wealth, saved_dd in zip(curve.Wealth_Index, curve.Drawdown_Pct):
            peak = max(peak, wealth)
            dd = (wealth / peak - 1) * 100
            _close(saved_dd, dd, "歷史回撤必須保留起始100及進場成本")
            drawdowns.append(dd)
        row = metric.loc[arm]
        _close(row.ROI_Pct, curve.Wealth_Index.iloc[-1] - 100, "歷史ROI與末日淨值不一致")
        _close(row.MDD_Pct, min(drawdowns), "歷史MDD與曲線不一致")
        _close(row.Sessions, sessions, "歷史Sessions與共同估值日數不一致")
        _close(row.Total_Transaction_Cost, book.Transaction_Cost.sum(), "歷史成本與NAV帳本不一致")
        _close(row.Average_Equity_Weight_Pct, book.Equity_Weight.mean() * 100, "歷史平均股票曝險不一致")
        _close(row.Average_Cash_Weight_Pct, (1 - book.Equity_Weight).mean() * 100, "歷史平均現金曝險不一致")
        for field, benchmark in (("Gap_Vs_444_Pp", "static_equal444"), ("Gap_Vs_555_Pp", "dynamic_news_risk555"), ("Gap_Vs_Matched_Equal_Pp", "v2_matched_equal")):
            _close(row[field], row.ROI_Pct - metric.loc[benchmark, "ROI_Pct"], "歷史配對差額與ROI不一致")
    if dates[0].strftime("%Y-%m-%d") != summary["period_start"] or dates[-1].strftime("%Y-%m-%d") != summary["period_end"]:
        raise V2HistoryError("歷史共同估值期間與摘要不一致")
    target_dates = protocol.get("shared_target_dates")
    if (not isinstance(target_dates, list) or len(target_dates) != sessions
            or not pd.DatetimeIndex(_date_column(pd.DataFrame({"Date": target_dates}), "Date", "共同估值")).equals(dates)):
        raise V2HistoryError("歷史NAV日期與協議共同估值日不一致")
    _validate_predictions(tables, summary, protocol, dates)
    # Check the exact bytes again to reject a partial deployment during a read.
    for name, raw in contents.items():
        if _path(report, name).read_bytes() != raw:
            raise V2HistoryError("歷史來源在載入期間變動，等待完整同步")
    if manifest_path.read_bytes() != manifest_raw:
        raise V2HistoryError("歷史manifest在載入期間變動")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_digest:
        raise V2HistoryError("歷史模型包在載入期間變動")
    return {
        **tables, "available": True, "mode": "saved", "is_demo": False,
        "initial_capital": capital, "period_start": summary["period_start"],
        "period_end": summary["period_end"], "model_arm": summary["model_arm"],
        "benchmark_arm": summary["benchmark_arm"], "matched_arm": summary["matched_arm"],
        "benchmark_label": metric.loc[summary["benchmark_arm"], "Display_Name"],
        "protocol": protocol, "summary": summary, "manifest": manifest,
        "sources": recorded, "report_dir": REPORT_DIR.as_posix(),
        "summary_markdown": contents["summary.md"].decode("utf-8-sig"),
    }


def _table(frame):
    st.dataframe(frame, hide_index=True, width="stretch")


def _bars(frame, category, value, title, **kwargs):
    st.altair_chart(comparison_bars(frame, category, value, title, **kwargs), use_container_width=True)


def _overview(data):
    metric = data["metrics"].set_index("Arm")
    model = metric.loc[data["model_arm"]]
    benchmark = metric.loc[data["benchmark_arm"]]
    matched = metric.loc[data["matched_arm"]]
    a, b, c, d = st.columns(4)
    a.metric("V2.12新聞模型淨報酬", f"{model.ROI_Pct:.2f}%")
    b.metric("444固定持股淨報酬", f"{benchmark.ROI_Pct:.2f}%")
    c.metric("新聞模型對444差額", f"{model.Gap_Vs_444_Pp:+.2f} 個百分點")
    d.metric("新聞模型最大回撤", f"{model.MDD_Pct:.2f}%")
    st.caption("成本後淨值包含首日進場與末日真正清算；100000元、小數股、次日收盤成交、手續費0.1425%、賣稅0.3%、最低費用0、滑價0、股利0。")
    with st.container(border=True):
        left, right = st.columns([3, 2])
        left.subheader("150個共同估值日的資金路徑")
        with right:
            view = st.radio("圖表視角", ["成本後淨值", "相對基準", "回撤路徑"], horizontal=True,
                            key="v2_curve_view", label_visibility="collapsed")
        names = metric.Display_Name.to_dict()
        optional = [arm for arm in ARMS if arm != data["benchmark_arm"]]
        selected = st.multiselect("加入比較組（444固定持股始終保留）", optional,
                                 default=[data["model_arm"], data["matched_arm"]],
                                 format_func=lambda arm: names[arm], key="v2_history_arms")
        st.altair_chart(wealth_chart(data["curves"], view, [data["benchmark_arm"], *selected],
                                    data["benchmark_arm"], benchmark_label="444固定持股"), use_container_width=True)
        st.caption("起始資金＝100，保留進場成本；相對財富＝該組淨值÷444淨值−1。150日為估值觀測，並非150個獨立樣本。")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("八組完整結果")
        _table(data["metrics"])
    with right:
        st.subheader("主對照：相同政策曝險等權")
        st.metric("V2同政策等權淨報酬", f"{matched.ROI_Pct:.2f}%")
        st.metric("新聞模型對同政策等權差額", f"{model.Gap_Vs_Matched_Equal_Pp:+.2f} 個百分點")
        st.write(f"新聞模型對444為 **{model.Gap_Vs_444_Pp:+.3f} 個百分點**，對同政策等權為 **{model.Gap_Vs_Matched_Equal_Pp:+.3f} 個百分點**。配置與現金曝險不同，需連同成本和風險解讀。")
        news_gap = model.ROI_Pct - metric.loc["v2_no_news_lstm", "ROI_Pct"]
        st.write(f"同任務的新聞模型對無新聞模型為 **{news_gap:+.3f} 個百分點**；配對不確定區間另見「優化實驗」，不據此改選每日模型。")
        st.caption("V2每5個交易日再平衡；原555每日再平衡。原555保留原canonical結果，頻率差異不能當成單純模型／新聞增益。")
        st.caption("原444／555是一日預測，V2預測五日報酬；共同估值與成本不會使預測任務相同。預測誤差僅在三個V2候選及各自零預測基準內比較。")
        st.write("本研究未納入同期0050含息帳本，不能宣稱超越0050。")


def _predictions(data):
    summary = data["summary"]
    st.subheader("新訓練模型的預測驗證")
    a, b, c, d = st.columns(4)
    a.metric("測試訊號日", str(summary["test_signal_days"]))
    b.metric("每模型已成熟訊號日", str(summary["matured_signal_days"]))
    c.metric("每模型待成熟訊號日", str(summary["pending_signal_days"]))
    count = summary.get("forecasts_per_model")
    d.metric("每模型預測列", str(count))
    st.caption("五日標籤＝Close[i+6]÷Close[i+1]−1；待成熟標籤保留缺值。同日五檔與重疊標籤相關，725個已成熟股票預測列不是725個獨立樣本。")
    metrics = data["prediction_metrics"]
    st.subheader("MAE、RMSE與簡單預測基準")
    aggregate = metrics.loc[metrics.Fold.astype(str).eq("all") & metrics.Split.eq("test")].copy()
    plot = aggregate.assign(候選=aggregate.Model.replace({"price_lstm": "價格", "no_news_lstm": "五特徵無新聞", "news_lstm": "六特徵新聞"}),
                            預測=aggregate.Forecast.replace({"model": "模型預測", "zero_return_baseline": "零報酬預測"}))
    left, right = st.columns(2)
    with left:
        _bars(plot, "候選", "MAE_Pp", "共同成熟標籤MAE", series="預測", unit="百分點", benchmark_label="零報酬預測")
    with right:
        _bars(plot, "候選", "RMSE_Pp", "共同成熟標籤RMSE", series="預測", unit="百分點", benchmark_label="零報酬預測")
    _table(aggregate)
    st.caption("三個候選完整呈現：price_lstm、去新聞情緒的no_news_lstm及六特徵news_lstm；歷史測試報酬未用來挑選勝出者。")
    st.caption("零報酬基準只評MAE／RMSE；預測0視為方向棄權，方向準確率需與覆蓋率及總是看多基準一起看，並非交易勝率。")
    st.subheader("每折預測驗證")
    _table(metrics.loc[~metrics.Fold.astype(str).eq("all")])
    with st.expander("逐折驗證門檻與完整預測時間線", expanded=False):
        _table(data["thresholds"])
        _table(data["prediction_timeline"])
    st.info("3個測試折各50訊號日；每折訓練756日、驗證60日，依完整交易日曆purge 6日。兩層50單元LSTM、10日輸入視窗的新訓練，並非把V2.11凍結權重倒填到過去。")


def _costs(data):
    st.subheader("曝險、成本與相同交易頻率對照")
    metric = data["metrics"].set_index("Arm")
    model = metric.loc[data["model_arm"]]
    a, b, c = st.columns(3)
    a.metric("新聞模型總成交成本", f"{model.Total_Transaction_Cost:,.2f} 元")
    b.metric("平均股票曝險", f"{model.Average_Equity_Weight_Pct:.2f}%")
    c.metric("平均現金曝險", f"{model.Average_Cash_Weight_Pct:.2f}%")
    left, right = st.columns(2)
    with left:
        _bars(data["metrics"], "Display_Name", "Total_Transaction_Cost", "含末日清算的成交成本", unit="元")
    with right:
        _bars(data["metrics"], "Display_Name", "Average_Equity_Weight_Pct", "平均股票曝險", unit="%")
    chosen = st.selectbox("曝險路徑組別", list(ARMS), index=ARMS.index(data["model_arm"]),
                          format_func=lambda arm: metric.loc[arm, "Display_Name"], key="v2_history_exposure")
    nav = data["nav"].loc[data["nav"].Arm.eq(chosen)].copy()
    exposure = pd.concat([
        nav[["Date"]].assign(Series="股票曝險", Weight_Pct=nav.Equity_Weight * 100),
        nav[["Date"]].assign(Series="現金曝險", Weight_Pct=(1 - nav.Equity_Weight) * 100),
    ], ignore_index=True)
    chart = alt.Chart(exposure).mark_area(opacity=.85).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Weight_Pct:Q", title="資金比例（%）", stack="zero", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=["股票曝險", "現金曝險"], range=["#2563eb", "#d9e1ec"]), legend=alt.Legend(orient="bottom")),
        tooltip=["Date:T", "Series:N", alt.Tooltip("Weight_Pct:Q", format=".2f")])
    st.altair_chart(chart.properties(width="container", height=240), use_container_width=True)
    st.subheader("V2相同5日頻率的對照")
    controls = ["v2_news_lstm", "v2_news_no_risk", "v2_matched_equal", "full_equal_5"]
    _table(data["metrics"].loc[data["metrics"].Arm.isin(controls)])
    st.caption("同政策等權跟隨新聞模型的事前風險總曝險；full_equal_5維持全額股票等權，分離降曝險與選股配置。末日已執行清算，末日曝險會降為現金。")
    with st.expander("成交、目標權重與逐日NAV"):
        _table(data["fills"])
        _table(data["weights"])
        _table(data["nav"])


def _experiments(data):
    st.subheader("配對效果與不確定性")
    effects = data["paired_effects"].copy()
    effects["對照"] = effects.Contrast.replace({
        "news_vs_no_news": "新聞模型－無新聞模型",
        "risk_with_news": "新聞有風控－新聞無風控",
        "selection_at_same_gross": "新聞模型－同政策等權",
    })
    base = alt.Chart(effects).encode(y=alt.Y("對照:N", title=None), tooltip=[
        "對照:N", alt.Tooltip("Mean_Daily_Difference_Pp:Q", format=".4f"),
        alt.Tooltip("CI_Lower_Pp:Q", format=".4f"), alt.Tooltip("CI_Upper_Pp:Q", format=".4f"), "N_Days:Q"])
    interval = base.mark_rule(color="#7b8799", strokeWidth=3).encode(
        x=alt.X("CI_Lower_Pp:Q", title="配對日報酬差額與95%區間（百分點）"), x2="CI_Upper_Pp:Q")
    point = base.mark_circle(size=90, color="#2563eb").encode(x="Mean_Daily_Difference_Pp:Q")
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#bdc8d7", strokeDash=[4, 4]).encode(x="zero:Q")
    st.altair_chart((zero + interval + point).properties(width="container", height=230), use_container_width=True)
    _table(data["paired_effects"])
    st.caption("配對統計使用149個持有期日報酬，排除沒有先前持倉報酬的首日；首筆成本仍保留在ROI。區塊長20日、2000次重抽樣；95%區間跨0表示本樣本未排除雜訊，不代表證明沒有任何效果。")
    crosses_zero = effects.CI_Lower_Pp.le(0) & effects.CI_Upper_Pp.ge(0)
    if crosses_zero.all():
        st.write("本次三個配對區間皆涵蓋0，沒有足夠證據宣稱新聞、風控或選股配置取得顯著增益。無風控組的歷史報酬較高，也不據此替換正式策略。")
    st.write("新聞有／無比較news_lstm與no_news_lstm；價格特徵對照price_lstm；風控比較news_lstm與news_no_risk；同曝險等權控制現金比例與交易頻率。不同模型與曝險的合成差額不稱為純新聞因果效果。")
    st.info("這是已查看歷史的探索性研究，不是前瞻驗證。三個預先固定候選與各折結果全部保留，未以這段測試報酬改選模型或回填V2.11。")
    with st.expander("共同日期的配對日報酬"):
        _table(data["daily_returns"])


def _sources(data, archive_renderer: Callable | None):
    st.subheader("已補證據與仍有的缺口")
    _table(pd.DataFrame([
        {"項目": "共同150日NAV／成本／曝險", "狀態": "已補", "說明": "八組同日期、保留首筆成本及末日清算"},
        {"項目": "新訓練各折權重與Scaler", "狀態": "已補", "說明": "45組fold×model×ticker；獨立模型包及SHA-256"},
        {"項目": "預測成熟狀態／各折驗證門檻", "狀態": "已補", "說明": "已成熟與待成熟分開，完整候選不挑勝出"},
        {"項目": "歷史新聞point-in-time證據", "狀態": "仍未證明", "說明": "hash證明完整性，不認證過去已可取得"},
        {"項目": "舊私人Ridge前瞻原始鏈", "狀態": "仍未取得", "說明": "完整帳本、receipt及CHAIN不能由DEMO替代"},
        {"項目": "同期0050可交易含息基準", "狀態": "未納入", "說明": "本帳本不含息，不能宣稱勝過0050"},
        {"項目": "前瞻樣本／獨立市場樣本", "狀態": "非本研究", "說明": "150估值日、重疊標籤及同日五檔均非獨立試驗"},
    ]))
    st.write("新歷史研究獨立於V2.11每日凍結追蹤。新模型包補齊本輪訓練證據，舊歷史結果缺失的權重與發布時點不會因此獲得認證。")
    coverage = data.get("news_coverage")
    if coverage is not None:
        st.subheader("本輪歷史新聞的實際覆蓋")
        unknown, total = int(coverage.Unknown_Source_Status_Rows.sum()), int(coverage.Stock_Rows.sum())
        st.warning(f"{total}個訊號日×股票觀測列中，{unknown}列缺來源狀態；零情緒分數不等於當時已完成查詢或確認沒有文章。三折歷史可用時點皆未認證。")
        _table(coverage)
    if archive_renderer is not None:
        section = st.selectbox("歷史與其他來源", ["本輪來源", "原 555 與 V2", "歷史實驗", "Ridge 前瞻", "限制"], key="v2_archive_section")
        if section != "本輪來源":
            archive_renderer(section)
    with st.expander("下載與研究來源"):
        a, b, c = st.columns(3)
        a.download_button("下載共同回測指標CSV", data["metrics"].to_csv(index=False).encode("utf-8-sig"), "v212_common_metrics.csv", "text/csv", width="stretch")
        b.download_button("下載共同淨值曲線CSV", data["curves"].to_csv(index=False).encode("utf-8-sig"), "v212_common_curves.csv", "text/csv", width="stretch")
        c.link_button("本輪研究摘要", GITHUB + data["report_dir"] + "/summary.md", width="stretch")
        st.caption("下載僅含衍生摘要與淨值；不重散布完整行情。報告與模型包逐檔校驗SHA-256，完整性不等於歷史可用性。")
        st.json(data["summary"], expanded=False)
        st.json(data["protocol"], expanded=False)
        st.json(data["manifest"], expanded=False)


def render_v2_history(data: dict, *, daily_renderer: Callable, archive_renderer: Callable | None = None):
    """Render six native research tabs; daily operations retain their own book."""
    st.info("V2.12同標準歷史回測｜HISTORICAL_EXPLORATORY。獨立新訓練、共同150日帳本；不是V2.11前瞻績效，也不回填已發布訊號。")
    st.caption(f"估值期間 {data['period_start']} — {data['period_end']}；固定基準：444固定持股；初始資金 {data['initial_capital']:,.0f} 元。")
    tabs = st.tabs(RESEARCH_TABS)
    with tabs[0]: _overview(data)
    with tabs[1]: _predictions(data)
    with tabs[2]: _costs(data)
    with tabs[3]:
        st.info("以下每日運作讀取V2.11正式凍結追蹤，日期及績效獨立於上方歷史回測。")
        daily_renderer()
    with tabs[4]: _experiments(data)
    with tabs[5]: _sources(data, archive_renderer)
