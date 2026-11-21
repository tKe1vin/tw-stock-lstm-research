"""Calendar and recovery scenarios exercise the real session/cache boundaries."""
from datetime import datetime
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.v2 import v211
from scripts import daily_reports, run_all_daily as daily, run_model as runner
from scripts.common import TICKERS, current_completed_session, sessions
from scripts.unified_csv import COLUMNS, blank_row


@pytest.mark.parametrize("clock,expected", [
    ("2026-11-16T23:59:59+08:00", "2026-11-16"),
    ("2026-11-17T00:00:00+08:00", "2026-11-16"),
    ("2026-11-17T14:59:59+08:00", "2026-11-16"),
    ("2026-11-17T15:00:00+08:00", "2026-11-17"),
    ("2026-11-16T16:00:00+00:00", "2026-11-16"),
    ("2026-11-14T09:17:00+08:00", "2026-11-13"),
    ("2026-11-16T09:17:00+08:00", "2026-11-13"),
    ("2022-02-04T21:37:00+08:00", "2022-01-26"),
    ("2022-02-07T09:17:00+08:00", "2022-01-26"),
    ("2022-02-07T15:00:00+08:00", "2022-02-07"),
])
def test_latest_completed_session_uses_exchange_days_and_taipei_cutoff(clock, expected):
    assert current_completed_session(datetime.fromisoformat(clock)) == pd.Timestamp(expected)


@pytest.fixture
def isolated_daily(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    monkeypatch.setattr(daily, "LOCK", tmp_path / "logs/orchestrator/run.lock")
    monkeypatch.setattr(daily, "_RUN_CONTEXT", {})
    # Report persistence has its own integration tests; these scenarios isolate
    # process scheduling and use actual cached model statuses on disk.
    monkeypatch.setattr(daily_reports, "build_reports", lambda *_: {"status": "success"})
    return tmp_path


def save_export(root, model, day):
    """Write a complete small daily export, with the actual cache file contract."""
    folder = root / "outputs" / model / day
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, ticker in enumerate(TICKERS, 1):
        row = blank_row()
        row.update(model_id=model, signal_date=day, data_asof=day, stock_id=ticker,
                   predicted_return_pct=0.1, target_weight=0.2, cash_weight=0.,
                   prediction_horizon_days=5 if model == "555_v2" else 1)
        if model == "555_v2":
            row['rank'] = rank
        rows.append(row)
    forecast = pd.DataFrame(rows, columns=COLUMNS)
    unified = folder / "unified_signal.csv"
    forecast.to_csv(unified, index=False)
    status = {"model": model, "status": "success", "exit_code": 0,
              "completed_session": day, "unified_csv": unified.relative_to(root).as_posix()}
    if model == "555_v2":
        status["weight_sha256"] = runner.EXPECTED_WEIGHTS
    if model == "555":
        status["research_protocol"] = "paired-444-core-allocation-v1"
        core = root / "outputs/444" / day / "native/test_predictions.csv"
        if not core.exists():
            save_export(root, "444", day)
        status["primary_research"] = {"source_prediction_sha256": hashlib.sha256(core.read_bytes()).hexdigest()}
    if model in {"444", "555"}:
        native = folder / "native"
        native.mkdir(exist_ok=True)
        previous = str(sessions(pd.Timestamp(day) - pd.Timedelta(days=10), day)[-2].date())
        target = str(sessions(pd.Timestamp(day) + pd.Timedelta(days=1), pd.Timestamp(day) + pd.Timedelta(days=10))[0].date())
        (native / "evaluation_protocol.json").write_text(json.dumps({"as_of_date": day,
            "test_days": 1, "test_target_dates": [day]}))
        pd.DataFrame([{"Ticker": ticker, "Signal_Date": day, "Target_Date": target,
                       "Predicted_Return_Pct": .1, "Target_Weight": 0.2}
                      for ticker in TICKERS]).to_csv(native / "next_day_recommendation.csv", index=False)
        pd.DataFrame([{"Ticker": ticker, "Signal_Date": previous, "Date": day,
                       "Previous_Close": 100., "Actual_Close": 100.1, "Predicted_Close": 100.1,
                       "Actual_Return_Pct": .1, "Predicted_Return_Pct": .1}
                      for ticker in TICKERS]).to_csv(native / "test_predictions.csv", index=False)
        pd.DataFrame([{"Ticker": ticker, "Price_MAE": .1, "Price_RMSE": .1,
                       "Return_MAE_pp": .1, "Return_RMSE_pp": .1, "Direction_Accuracy_Pct": 50.}
                      for ticker in TICKERS]).to_csv(native / "model_metrics.csv", index=False)
        pd.DataFrame([{"As_Of_Date": day, "ROI_Pct": .1, "Sharpe": .1, "MDD_Pct": -.1,
                       "Direction_Accuracy_Pct": 50., "Test_Days": 1}]).to_csv(native / "strategy_metrics.csv", index=False)
        pd.DataFrame([{"Date": previous, "Portfolio_Net" if model == "444" else "NAV": 100000., "Market_Index": 100000.},
                      {"Date": day, "Portfolio_Net" if model == "444" else "NAV": 100100., "Market_Index": 100100.}]).to_csv(
            native / "backtest_nav.csv", index=False)
        if model == "555":
            pd.DataFrame([{"Sharpe": .1, "MDD_Pct": -.1}]).to_csv(native / "market_metrics.csv", index=False)
        status["daily_native_dir"] = native.relative_to(root).as_posix()
    (folder / "status.json").write_text(json.dumps(status))
    return status


def summary(root, run_date):
    return json.loads((root / "outputs/daily_summary" / run_date / "summary.json").read_text())


def test_source_recovers_then_midnight_reuses_session_and_next_close_requires_new_exports(
        isolated_daily, monkeypatch):
    root = isolated_daily
    observed = []
    source = {"ready": False}

    def process(command, name, extra_env=None):
        context = dict(extra_env or {})
        day = context["TW_LSTM_TARGET_SESSION"]
        actual = datetime.fromisoformat(context["TW_LSTM_RUN_STARTED_AT"])
        observed.append((name, day, actual.date().isoformat()))
        assert context["TW_LSTM_OUTPUT_DAY"] == day
        if name == "market":
            payload = {"completed_session": day, "prices_ready": True,
                       "sentiment_ready": source["ready"]}
            code = 0 if source["ready"] else 2
        else:
            assert context["TW_LSTM_SKIP_MARKET_UPDATE"] == "1"
            assert command[-2:] == ["--mode", "full"]
            if name == "v211" and day != actual.date().isoformat():
                payload = {"model": name, "status": "maintenance_only",
                           "reason": "delayed_session_maintenance_only",
                           "maintenance_successful": True, "signal_published": False}
                code = 2
            elif name == "444" or source["ready"]:
                payload = save_export(root, name, day)
                code = 0
            else:
                payload = {"model": name, "status": "waiting", "reason": "sentiment_not_ready"}
                code = 2
        return {"exit_code": code, "status": "success" if code == 0 else "waiting",
                "output": json.dumps(payload)}

    monkeypatch.setattr(daily, "_run_process", process)
    assert daily.main([], now=datetime.fromisoformat("2026-11-16T17:37:00+08:00")) == 2
    waiting = summary(root, "2026-11-16")
    assert waiting["status"] == "waiting"
    assert waiting["steps"]["444"]["status"] == "success"
    assert waiting["steps"]["555"]["reason"] == "sentiment_not_ready"
    core = root / "outputs/444/2026-11-16/native/test_predictions.csv"
    original_core = (core.read_bytes(), core.stat().st_mtime_ns)

    source["ready"] = True
    observed.clear()
    assert daily.main([], now=datetime.fromisoformat("2026-11-16T21:37:00+08:00")) == 0
    assert [name for name, *_ in observed] == ["market", "v211", "555_v2", "555"]
    assert summary(root, "2026-11-16")["steps"]["444"]["reason"] == "already_completed_for_this_session"

    observed.clear()
    assert daily.main([], now=datetime.fromisoformat("2026-11-17T00:03:00+08:00")) == 0
    midnight = summary(root, "2026-11-17")
    assert observed == [("market", "2026-11-16", "2026-11-17"),
                        ("v211", "2026-11-16", "2026-11-17")]
    assert midnight["status"] == "research_complete_tracker_deferred"
    assert midnight["completed_session"] == midnight["output_day"] == "2026-11-16"
    assert (core.read_bytes(), core.stat().st_mtime_ns) == original_core

    observed.clear()
    assert daily.main([], now=datetime.fromisoformat("2026-11-17T17:37:00+08:00")) == 0
    assert [name for name, *_ in observed] == ["market", *daily.ORDER]
    assert all(day == "2026-11-17" for _, day, _ in observed)
    assert summary(root, "2026-11-17")["status"] == "complete"
    assert (root / "outputs/444/2026-11-17/native/test_predictions.csv").is_file()
    assert (core.read_bytes(), core.stat().st_mtime_ns) == original_core
    assert not daily._RUN_CONTEXT
    fd = daily._acquire()
    assert fd is not None
    daily._release(fd)


@pytest.mark.parametrize("failed_model", ["v211", "555_v2", "444", "555"])
def test_failed_system_retry_reuses_successful_peers_and_retries_dependent_555(
        isolated_daily, monkeypatch, failed_model):
    root = isolated_daily
    day = "2026-11-16"
    calls = []
    state = {"retry": False}
    monkeypatch.setattr(daily, "_update_market", lambda: {"exit_code": 0})

    def run(model, mode):
        calls.append(model)
        if not state["retry"]:
            if model == failed_model:
                return {"model": model, "status": "failed", "exit_code": 1,
                        "reason": "controlled_model_failure"}
            if failed_model == "444" and model == "555":
                return {"model": model, "status": "waiting", "exit_code": 2,
                        "reason": "same_session_444_core_missing"}
        return save_export(root, model, day)

    monkeypatch.setattr(daily, "_run_one", run)
    assert daily.main([], now=datetime.fromisoformat("2026-11-16T17:37:00+08:00")) == 1
    assert calls == daily.ORDER
    first = summary(root, day)
    assert first["status"] == "failed" and first["steps"][failed_model]["exit_code"] == 1
    retained = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                for model in ("444", "555", "555_v2")
                for path in (root / "outputs" / model / day).rglob("*") if path.is_file()}
    assert retained

    state["retry"] = True
    calls.clear()
    assert daily.main([], now=datetime.fromisoformat("2026-11-16T21:37:00+08:00")) == 0
    expected = ["v211"]
    if failed_model != "v211":
        expected.append(failed_model)
    if failed_model == "444":
        expected.append("555")
    assert calls == expected
    assert summary(root, day)["status"] == "complete"
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in retained} == retained


@pytest.mark.parametrize("broken_record", ["status_list", "status_null", "protocol_list", "protocol_null"])
def test_malformed_cached_record_is_retried_without_aborting_independent_models(
        isolated_daily, monkeypatch, broken_record):
    root = isolated_daily
    day = "2026-11-16"
    for model in daily.ORDER:
        save_export(root, model, day)
    folder = root / "outputs/444" / day
    value = "null" if broken_record.endswith("null") else "[]"
    path = folder / ("status.json" if broken_record.startswith("status") else "native/evaluation_protocol.json")
    path.write_text(value)
    calls = []
    monkeypatch.setattr(daily, "_update_market", lambda: {"exit_code": 0})

    def run(model, mode):
        calls.append(model)
        return save_export(root, model, day)

    monkeypatch.setattr(daily, "_run_one", run)
    assert daily.main([], now=datetime.fromisoformat("2026-11-16T21:37:00+08:00")) == 0
    assert calls == ["v211", "444"]
    result = summary(root, day)
    assert result["steps"]["444"]["status"] == "success"
    assert result["steps"]["555"]["reason"] == "already_completed_for_this_session"
    assert not daily._RUN_CONTEXT
    fd = daily._acquire()
    assert fd is not None
    daily._release(fd)


@pytest.mark.parametrize("reference", ["absolute", "parent", "symlink", "symlink_loop"])
def test_completed_cache_cannot_reuse_an_export_outside_the_repository(
        isolated_daily, reference):
    root = isolated_daily
    day = "2026-11-16"
    status = save_export(root, "555", day)
    external = root.parent / f"{root.name}-outside.csv"
    external.write_text("signal_date,stock_id\n2026-11-16,2330.TW\n")
    if reference == "absolute":
        status["unified_csv"] = str(external)
    elif reference == "parent":
        status["unified_csv"] = f"../{external.name}"
    else:
        link = root / "linked-export.csv"
        link.symlink_to(link if reference == "symlink_loop" else external)
        status["unified_csv"] = link.name
    (root / "outputs/555" / day / "status.json").write_text(json.dumps(status))
    assert daily._completed_model("555", day, day) is None


def test_missing_signal_after_long_closure_is_skipped_and_label_waits_for_six_sessions():
    index = sessions("2022-01-26", "2022-02-14")
    panel = pd.DataFrame(100., index=index, columns=v211.TICKERS)

    def signal(day, entry):
        return pd.DataFrame([{"Signal_Date": day, "Planned_Execution_Date": entry,
                              "Ticker": ticker, "Target_Weight": 0.1, "Risk_Off": True, "Close": 100.}
                             for ticker in v211.TICKERS])

    result = v211.build_live_book(panel, pd.concat([
        signal("2022-01-26", "2022-02-07"), signal("2022-02-07", "2022-02-08"),
    ], ignore_index=True))
    assert set(result["fills"].Execution_Date) == {"2022-02-07"}
    daily_rows = result["daily"].set_index("Date")
    assert not bool(daily_rows.loc[pd.Timestamp("2022-02-08"), "Trade"])
    assert bool(daily_rows.loc[pd.Timestamp("2022-02-14"), "Skipped"])
    assert not v211.label_maturity("2022-01-26", index[:-1])["mature"]
    assert v211.label_maturity("2022-01-26", index)["label_end"] == "2022-02-14"
    assert v211.projected_label_end("2022-01-26") == "2022-02-14"


def test_force_reruns_exports_but_existing_v211_signal_keeps_bytes_and_publication_time(
        isolated_daily, monkeypatch):
    root = isolated_daily
    day = "2026-11-16"
    now = datetime.fromisoformat("2026-11-16T21:37:00+08:00")
    live = root / "experiments/results/v2/v211/daily"
    live.mkdir(parents=True)
    imported = live / "2026-11-05.csv"
    imported.write_bytes(b"original imported signal\r\n")
    existing = live / f"{day}.csv"
    created = "2026-11-16T17:40:03+08:00"
    pd.DataFrame([{"Signal_Date": day, "Ticker": ticker, "Created_At": created,
                   "Planned_Execution_Date": "2026-11-17", "Target_Weight": .2}
                  for ticker in TICKERS]).to_csv(existing, index=False)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (imported, existing)}
    for model in daily.ORDER:
        save_export(root, model, day)
    monkeypatch.setattr(v211, "ROOT", root)
    monkeypatch.setattr(v211, "LIVE", live)
    monkeypatch.setattr(v211, "predict_rows", lambda *_: pytest.fail("existing signal was recomputed"))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "run_day", lambda: (now, pd.Timestamp(day), pd.Timestamp(day)))
    monkeypatch.setattr(runner, "_frozen_card", lambda: (runner.EXPECTED_WEIGHTS, {}))
    monkeypatch.setattr(runner, "EXPECTED_SIGNAL", runner.sha256(imported))
    monkeypatch.setattr(daily, "_update_market", lambda: {"exit_code": 0})
    calls = []

    def tracker_process(command, **kwargs):
        assert command[-2:] == ["-m", "experiments.v2.v211_daily"]
        decision = v211.publish_today(now, (pd.DataFrame(index=[pd.Timestamp(day)]),), {})
        assert decision["reason"] == "already_written" and decision["created_at"] == created
        return SimpleNamespace(returncode=0, stdout=json.dumps(decision), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", tracker_process)

    def run(model, mode):
        calls.append(model)
        if model == "v211":
            code = runner.run_v211(mode)
            return {"model": model, "status": "success", "exit_code": code}
        return save_export(root, model, day)

    monkeypatch.setattr(daily, "_run_one", run)
    assert daily.main(["--force"], now=now) == 0
    assert calls == daily.ORDER
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    assert set(live.glob("*.csv")) == {imported, existing}
