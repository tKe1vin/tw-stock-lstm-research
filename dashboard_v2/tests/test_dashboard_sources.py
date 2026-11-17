from pathlib import Path

from dashboard_v2.data_access.load import (
    ROOT,
    historical_official,
    news_observations,
    paired_metrics,
    sentiment_class,
    stable_text,
    v211_signals,
    v211_status,
    v2_comparison,
)
from dashboard_v2.data_access.prospective import (
    EXPECTED_WEIGHT,
    compare_source_conflicts,
    day_review,
    next_day_strategy,
    ridge_status,
    source_status,
    trading_journal,
    v211_matured_local,
)

WEIGHT = ROOT / "experiments" / "results" / "v2" / "v211" / "weights" / "weights.npz"


def test_historical_tokens_stay_with_their_csv():
    data = historical_official()
    assert data["rows"]["444"]["ROI_Pct"] == "21.376684850112927"
    assert data["rows"]["555"]["ROI_Pct"] == "11.953686872710325"
    assert data["rows"]["^TWII benchmark"]["Strategy_ID"] == "^TWII benchmark"
    assert "不是可交易 ETF" in data["twii_note"]


def test_paired_protocol_is_separate():
    data = paired_metrics()
    by_arm = {row["Arm"]: row for row in data["rows"]}
    assert data["protocol"] == "paired-444-core-allocation-v1"
    assert by_arm["static_equal444"]["ROI_Pct"] == "18.32240681915924"
    assert by_arm["dynamic_news_risk555"]["ROI_Pct"] == "-2.9274705369013776"
    assert by_arm["risk_matched_equal_reference"]["ROI_Pct"] == "7.124321880311912"
    assert "theoretical_price_index" in by_arm["market_price_reference"]["Return_Basis"]
    assert data["causal_validated"] is False


def test_v2_ranking_keeps_incomparable_books_out():
    data = v2_comparison()
    names = [row["Name"] for row in data["ranked"]]
    assert "rolling_selected_threshold_weekly" in names
    assert "equal_weekly" in names
    assert "rolling_252_threshold_weekly" not in names
    assert "buy_hold" not in names
    selected = next(row for row in data["ranked"] if row["Name"] == "rolling_selected_threshold_weekly")
    assert selected["ROI_Pct"] == 25.40747279769098
    assert selected["Gap_Vs_Equal_ROI_Pct"] == -1.6396983864375585
    assert data["weights_saved"] is False


def test_v211_reads_local_prospective_snapshot():
    status = v211_status()
    assert status["model_sha256"] == EXPECTED_WEIGHT
    assert status["weight_ok"] is True
    assert status["registry_ok"] is True
    assert status["confirmation"] == "PENDING"
    assert status["matured_signal_days"] >= 2
    assert status["latest_signal_date"] >= "2026-11-16"
    signals = v211_signals()
    dates = sorted({row["Signal_Date"] for row in signals})
    assert "2026-11-05" in dates
    assert "2026-11-16" in dates
    assert all(row["_source_label"] == "LOCAL_PROSPECTIVE" for row in signals)
    assert all(row["_registry_match"] is True for row in signals)


def test_matured_and_pending_labels_are_separated():
    matured = v211_matured_local()
    assert set(matured["signal_days"]) >= {"2026-11-05", "2026-11-06"}
    assert "2026-11-09" not in matured["signal_days"]
    review = day_review()
    assert review["label_state"] in {"HAS_MATURED_LABELS", "PENDING_LABEL"}
    assert all(row["Signal_Date"] != "2026-11-09" for row in review["matured_rows"])


def test_source_conflict_is_visible():
    conflicts = compare_source_conflicts("2026-11-10")
    assert conflicts
    assert conflicts[0]["status"] == "SOURCE_CONFLICT"
    assert conflicts[0]["local_sha256"] != conflicts[0]["github_sha256"]
    assert len(conflicts[0]["local_sha256"]) == 64
    assert len(conflicts[0]["github_sha256"]) == 64
    assert conflicts[0]["same_price_sha"] is True
    assert conflicts[0]["same_model_sha"] is True
    assert conflicts[0].get("demo_flag") == "SYNTHETIC_DEMO"
    fields = {row["field"] for row in conflicts[0]["field_diffs"]}
    assert "Sentiment_Sha256" in fields
    assert "Created_At" in fields
    status = source_status()
    assert status["source_conflict"] is True
    assert status["not_official_performance"] is True


def test_ridge_not_merged_with_v211_and_scores_from_files():
    ridge = ridge_status()
    assert ridge["not_merged_with_v211"] is True
    assert ridge["status"] == "PENDING"
    assert ridge["scorable_days"] >= 5
    assert ridge["rows"] == ridge["scorable_days"] * 5
    assert ridge["first_day_ok"] is True
    assert ridge["mae"] is not None
    assert ridge["horizon"] == "1 trading day"
    assert ridge["demo_flag"] == "SYNTHETIC_DEMO"
    assert ridge["official_claim_blocked"] is True


def test_demo_fixtures_marked_and_not_official():
    status = source_status()
    assert status["data_kind"] == "SYNTHETIC_DEMO"
    assert status["demo_flag"] == "SYNTHETIC_DEMO"
    assert status["not_official_performance"] is True
    meta = Path(__file__).resolve().parents[1] / "fixtures" / "display_snapshots" / "DEMO_META.json"
    assert meta.exists()
    assert "SYNTHETIC_DEMO" in meta.read_text(encoding="utf-8")


def test_journal_and_next_day_use_real_local_ledger():
    rows = trading_journal("全部", "全部")
    assert rows
    assert any(row["state"] == "SIMULATED_EXECUTION" for row in rows)
    assert any(row["signal_date"] == "2026-11-12" and row["execution_date"] == "2026-11-13" for row in rows)
    plan = next_day_strategy()
    assert plan["state"] == "LATEST_PUBLISHED_SIGNAL"
    assert plan["signal_date"] == max(row["signal_date"] for row in rows)
    assert plan["source_label"] == "LOCAL_PROSPECTIVE"


def test_news_does_not_invent_titles_or_neutral_zeros():
    from dashboard.news import load_news_archive
    source = {row["cache_row_hash"]: row for row in load_news_archive(ROOT)["articles"].to_dict("records")}
    rows = news_observations()["rows"]
    assert rows
    assert all(row["title"] == source[row["cache_row_hash"]]["title"] for row in rows)
    assert all(row["title_status"] == ("DISPLAY_TITLE_LATER_RETRIEVAL" if row["title"] else "TITLE_NOT_STORED") for row in rows)
    assert sentiment_class("0") == "ZERO_UNVERIFIED"
    assert sentiment_class("0.0") == "ZERO_UNVERIFIED"
    assert sentiment_class("") == "UNVERIFIED"


def test_partial_file_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "partial.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    calls = {"n": 0}

    def flip(self):
        calls["n"] += 1
        if calls["n"] == 1:
            return b"a\n1\n"
        return b"a\n2\n"

    monkeypatch.setattr(Path, "read_bytes", flip)
    try:
        stable_text(path)
    except RuntimeError as exc:
        assert "不完整" in str(exc)
    else:
        raise AssertionError("半寫入檔應被拒絕")


def test_dashboard_read_does_not_touch_weights():
    before = WEIGHT.stat().st_mtime_ns
    v211_status()
    v2_comparison()
    historical_official()
    ridge_status()
    assert WEIGHT.stat().st_mtime_ns == before
