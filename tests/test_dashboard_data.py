"""Evidence-bound presentation curves and operational-state semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from dashboard import data
from scripts.export_dashboard_data import build_assets


@pytest.fixture
def retained_root(tmp_path):
    for spec in data.STUDIES.values():
        for name in data.REPORT_FILES:
            source = data.ROOT / spec["report_dir"] / name
            target = tmp_path / spec["report_dir"] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copytree(data.ROOT / data.ASSETS, tmp_path / data.ASSETS)
    return tmp_path


def update_curve_hash(root, version):
    path = root / data.ASSETS / "manifest.json"
    manifest = json.loads(path.read_text())
    curve = root / data.ASSETS / data.STUDIES[version]["curves_file"]
    manifest["studies"][version]["curves_sha256"] = data.sha256(curve)
    path.write_text(json.dumps(manifest))


def test_real_retained_studies_keep_separate_periods_and_exact_reported_values():
    monthly, weekly = data.load_study("v3.3"), data.load_study("v3.2")
    assert (monthly["period_start"], monthly["period_end"]) == ("2026-04-15", "2026-11-09")
    assert (weekly["period_start"], weekly["period_end"]) == ("2026-08-04", "2026-11-03")
    indexed = monthly["metrics"].set_index("Arm")
    assert indexed.loc["ridge_news", "ROI_Pct"] == pytest.approx(13.004870208538533)
    assert indexed.loc["B0_benchmark", "ROI_Pct"] == pytest.approx(15.904759519657773)
    assert indexed.loc["ridge_news", "Gap_Vs_Benchmark_Pp"] == pytest.approx(-2.8998893111192405)
    assert monthly["selection"]["selected"] == "B0_benchmark"
    assert monthly["limitations"]["fallback_diagnostic_pair"] is True
    assert set(monthly["validation"].Candidate) == {"ridge_price", "ridge_news", "hgb_price", "hgb_news"}


def test_wealth_base_keeps_entry_costs_and_reconciles_roi_and_drawdown():
    for version in data.STUDIES:
        study = data.load_study(version)
        for arm, part in study["curves"].groupby("Arm"):
            row = study["metrics"].set_index("Arm").loc[arm]
            assert part.Wealth_Index.iloc[0] < 100
            assert part.Wealth_Index.iloc[-1] - 100 == pytest.approx(row.ROI_Pct)
            assert part.Drawdown_Pct.min() == pytest.approx(row.MDD_Pct)


def test_load_is_read_only_and_has_no_private_provider_cache_dependency(retained_root):
    before = {str(p.relative_to(retained_root)): p.read_bytes() for p in retained_root.rglob("*") if p.is_file()}
    data.load_study("v3.3", retained_root)
    data.load_study("v3.2", retained_root)
    after = {str(p.relative_to(retained_root)): p.read_bytes() for p in retained_root.rglob("*") if p.is_file()}
    assert before == after
    assert not (retained_root / "inputs").exists()


def test_changed_curve_hash_or_source_report_fails_closed(retained_root):
    curve = retained_root / data.ASSETS / "curves_v33.csv"
    curve.write_bytes(curve.read_bytes() + b"\n")
    with pytest.raises(data.DataIntegrityError, match="雜湊"):
        data.load_study("v3.3", retained_root)
    source = retained_root / data.STUDIES["v3.2"]["report_dir"] / "holdout_metrics.csv"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(data.DataIntegrityError, match="研究來源已變動"):
        data.load_study("v3.2", retained_root)


@pytest.mark.parametrize("mutation", ["normalize_first", "missing_day", "nonfinite", "raw_price_column"])
def test_semantic_curve_validation_remains_after_manifest_rehash(retained_root, mutation):
    path = retained_root / data.ASSETS / "curves_v33.csv"
    frame = pd.read_csv(path)
    if mutation == "normalize_first":
        selected = frame.Arm.eq("ridge_news")
        frame.loc[selected, "Wealth_Index"] *= 100 / frame.loc[selected, "Wealth_Index"].iloc[0]
    elif mutation == "missing_day":
        frame = frame.drop(frame[frame.Arm.eq("ridge_news")].index[5])
    elif mutation == "nonfinite":
        frame.loc[0, "Wealth_Index"] = np.inf
    else:
        frame["Close"] = 100.
    frame.to_csv(path, index=False)
    update_curve_hash(retained_root, "v3.3")
    with pytest.raises(data.DataIntegrityError):
        data.load_study("v3.3", retained_root)


def test_unknown_version_cannot_select_arbitrary_files():
    with pytest.raises(data.DataIntegrityError, match="未知"):
        data.load_study("../../private")


def test_missing_daily_status_does_not_invent_an_operational_success(tmp_path):
    value = data.load_daily_status(tmp_path)
    assert value["available"] is False
    assert value["models"] == []


def test_daily_status_keeps_primary_research_and_deferred_tracker_distinct(tmp_path):
    target = tmp_path / "reports/latest/four_systems.json"
    target.parent.mkdir(parents=True)
    record = {"as_of_date": "2026-11-16", "run_date": "2026-11-17", "status": "partial",
              "research_status": "complete", "delayed_run": True,
              "models": {name: {"current": name != "v211", "status": "maintenance_only" if name == "v211" else "success",
                                  "maintenance_only": name == "v211"} for name in ("444", "555", "555_v2", "v211")},
              "news": {"article_text_included": False}}
    target.write_text(json.dumps(record))
    value = data.load_daily_status(tmp_path)
    assert value["status"] == "partial" and value["research_status"] == "complete"
    assert next(item for item in value["models"] if item["model"] == "v211")["current"] is False
    record["models"]["v211"]["current"] = True
    target.write_text(json.dumps(record))
    with pytest.raises(data.DataIntegrityError, match="僅維護"):
        data.load_daily_status(tmp_path)


def fixture_runs(tmp_path):
    root = tmp_path / "checkout"
    runs = {}
    for version, spec in data.STUDIES.items():
        report = root / spec["report_dir"]
        report.mkdir(parents=True)
        run = tmp_path / ("private_" + version)
        (run / "inputs").mkdir(parents=True)
        (run / "inputs/bars.csv").write_text("restricted-provider-raw-marker")
        metrics, files = [], {}
        for arm in data.ARM_NAMES:
            folder = run / ("holdout_" + arm)
            folder.mkdir()
            path = folder / "nav.csv"
            pd.DataFrame({"Date": ["2026-01-05", "2026-01-06"], "NAV": [980., 1100.],
                          "Close": [99., 101.], "Volume": [10000, 20000]}).to_csv(path, index=False)
            files[str(path.relative_to(run))] = data.sha256(path)
            metrics.append({"Arm": arm, "ROI_Pct": 10., "MDD_Pct": -2., "Total_Transaction_Cost": 20.,
                            "Initial_Capital": 1000., "Sessions": 2, "Start": "2026-01-05", "End": "2026-01-06"})
        pd.DataFrame(metrics).to_csv(report / "holdout_metrics.csv", index=False)
        files["holdout_metrics.csv"] = data.sha256(report / "holdout_metrics.csv")
        (run / "reproduction_manifest.json").write_text(json.dumps({"files_sha256": files}))
        for name in data.REPORT_FILES:
            if not (report / name).exists():
                (report / name).write_text("{}")
        (report / "input_provenance.json").write_text(json.dumps({"private_run_manifest_sha256": data.sha256(run / "reproduction_manifest.json")}))
        runs[version] = run
    return root, runs


def test_explicit_export_copies_derived_wealth_only_and_refuses_overwrite(tmp_path):
    root, runs = fixture_runs(tmp_path)
    result = build_assets(runs, root=root)
    folder = root / data.ASSETS
    assert not result["raw_provider_data_included"]
    assert set(p.name for p in folder.iterdir()) == {"curves_v33.csv", "curves_v32.csv", "manifest.json"}
    assert all(b"restricted-provider-raw-marker" not in p.read_bytes() for p in folder.iterdir())
    assert list(pd.read_csv(folder / "curves_v33.csv").columns) == ["Date", "Arm", "Wealth_Index", "Drawdown_Pct"]
    with pytest.raises(FileExistsError):
        build_assets(runs, root=root)


def test_export_rejects_mutated_private_nav_before_creating_any_asset(tmp_path):
    root, runs = fixture_runs(tmp_path)
    path = runs["v3.3"] / "holdout_B0_benchmark/nav.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(data.DataIntegrityError, match="NAV被更改"):
        build_assets(runs, root=root)
    assert not (root / data.ASSETS).exists()
