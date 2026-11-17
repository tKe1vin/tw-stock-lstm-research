"""Finite old/new allocator × price/news replay; default inspection is read-only."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("ridge_price", "ridge_news")
INPUTS = ("registered_config.json", "splits.json", "validation_predictions.csv",
          "holdout_ridge_price_predictions.csv", "holdout_ridge_news_predictions.csv",
          "inputs/bars.csv", "inputs/bars.metadata.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _book(engine, bars, prediction, dates, config, *, stress=False, benchmark=False):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(bars.Date).unique()))
    positions = calendar.get_indexer(dates)
    if (positions < 0).any() or not len(positions):
        raise ValueError("Retained split is not on the full calendar")
    last = positions[(positions - positions[0]) % 20 == 0][-1]
    if last + 21 >= len(calendar):
        raise ValueError("Incomplete mature horizon")
    options = {**config, "allocation_mode": "benchmark" if benchmark else "model"}
    if stress:
        options = {**options, "execution": {**config["execution"], "slippage_bps_each_side": 15.}}
    # Deliberately remove Actual, entry and label columns before allocation.
    supplied = prediction[["Date", "Ticker", "Predicted"]] if prediction is not None else None
    return engine(bars, supplied, dates, options, end_date=calendar[last + 21])


def _empty_destination(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Use a new empty experiment directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_experiment(retained_run, output_dir, publish_dir):
    from research.v3.data import load_dataset
    from research.v3_3.portfolio import run_backtest as legacy
    from research.v3_4.portfolio import run_backtest as incremental

    retained, output, publish = map(lambda p: Path(p).resolve(), (retained_run, output_dir, publish_dir))
    if output.is_relative_to(ROOT) or retained.is_relative_to(ROOT):
        raise ValueError("Private provider inputs and detailed runs must stay outside Git")
    if not publish.is_relative_to(ROOT / "reports/research_v3_4"):
        raise ValueError("Publish only derived metrics under reports/research_v3_4")
    old_manifest = json.loads((retained / "reproduction_manifest.json").read_text())
    published_proof = json.loads((ROOT / "reports/research_v3_3/2026-11-17/input_provenance.json").read_text())
    if digest(retained / "reproduction_manifest.json") != published_proof["private_run_manifest_sha256"]:
        raise ValueError("Retained source does not match the already published provenance")
    protected = {name: digest(retained / name) for name in INPUTS}
    for name, value in protected.items():
        if value != old_manifest["files_sha256"].get(name):
            raise ValueError("Retained input changed: " + name)
    source_paths = ["research/v3_3/portfolio.py", "research/v3_4/portfolio.py", "research/v3_4/allocator.py",
                    "research/v3_4/run.py", "research/v3_4/design.md"]
    source_hashes = {name: digest(ROOT / name) for name in source_paths}
    registration = {"experiment": "V3.4 single incremental-cost allocation replay",
                    "registered_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                    "source_run_manifest_sha256": digest(retained / "reproduction_manifest.json"),
                    "input_sha256": protected, "source_sha256": source_hashes,
                    "allocators": ["legacy", "incremental"], "forecasts": list(ARMS),
                    "validation_folds": [1, 2, 3], "holdout_used_for_selection": False,
                    "historical_alpha_proven": False, "no_retraining": True,
                    "decision_cost_multiplier": 1.5, "slippage_bps": [5, 15],
                    "target": "i+1 close to i+21 close excess versus0050",
                    "private_provider_data_published": False, "selected": "B0_benchmark",
                    "statement": "Historical exploratory fixed allocator comparison; no adoption or threshold search"}
    output, publish = _empty_destination(output), _empty_destination(publish)
    write_json(output / "registration.json", registration)
    write_json(publish / "registration.json", registration)
    config = json.loads((retained / "registered_config.json").read_text())
    splits = json.loads((retained / "splits.json").read_text())
    bars = load_dataset(retained / "inputs/bars.csv", metadata_path=retained / "inputs/bars.metadata.json",
                        expected_tickers=["0050.TW", *config["satellite_tickers"]])
    validation = pd.read_csv(retained / "validation_predictions.csv")
    sets = [("validation", i, split["validation"]) for i, split in enumerate(splits["folds"], 1)]
    sets.append(("holdout", 0, splits["test"]))
    records = []
    for split, fold, dates in sets:
        base = _book(legacy, bars, None, dates, config, benchmark=True)
        bm = base["metrics"]
        records.append({"Split": split, "Fold": fold, "Allocator": "benchmark", "Arm": "B0_benchmark", **bm,
                        "Stress_Relative_Wealth_Excess_Pct": 0.})
        for arm in ARMS:
            prediction = validation[validation.Candidate.eq(arm) & validation.Fold.eq(fold)] if split == "validation" else pd.read_csv(retained / f"holdout_{arm}_predictions.csv")
            for allocator, engine in (("legacy", legacy), ("incremental", incremental)):
                print(json.dumps({"split": split, "fold": fold, "arm": arm, "allocator": allocator}), flush=True)
                book = _book(engine, bars, prediction, dates, config)
                stress = _book(engine, bars, prediction, dates, config, stress=True)
                if allocator == "legacy":
                    relative = f"fold{fold}_{arm}/metrics.json" if split == "validation" else f"holdout_{arm}/metrics.json"
                    if digest(retained / relative) != old_manifest["files_sha256"].get(relative):
                        raise ValueError("Saved baseline metrics changed")
                    original = json.loads((retained / relative).read_text())
                    for key in ("ROI_Pct", "MDD_Pct", "Total_Transaction_Cost"):
                        if abs(float(book["metrics"][key]) - float(original[key])) > 1e-8:
                            raise ValueError("Legacy replay failed before accepting new results: " + key)
                name = f"{split}{fold}_{allocator}_{arm}"
                private = output / name
                private.mkdir()
                for table in ("nav", "fills", "weights"):
                    book[table].to_csv(private / (table + ".csv"), index=False)
                write_json(private / "metrics.json", book["metrics"])
                records.append({"Split": split, "Fold": fold, "Allocator": allocator, "Arm": arm,
                                **book["metrics"], "Stress_Relative_Wealth_Excess_Pct": stress["metrics"]["Relative_Wealth_Excess_Pct"]})
    frame = pd.DataFrame(records)
    for _, part in frame.groupby(["Split", "Fold", "Allocator"]):
        if set(ARMS).issubset(set(part.Arm)):
            paired = part.set_index("Arm")
            frame.loc[part.index, "News_Minus_Price_Pp"] = paired.loc["ridge_news", "ROI_Pct"] - paired.loc["ridge_price", "ROI_Pct"]
    gates = []
    for allocator in ("legacy", "incremental"):
        sub = frame[(frame.Split == "validation") & (frame.Allocator == allocator)]
        news, price = sub[sub.Arm == "ridge_news"], sub[sub.Arm == "ridge_price"]
        compound = lambda values: float((np.prod(1 + values / 100) - 1) * 100)
        positive = int((news.Relative_Wealth_Excess_Pct > 0).sum())
        x, p, stress = compound(news.Relative_Wealth_Excess_Pct), compound(price.Relative_Wealth_Excess_Pct), compound(news.Stress_Relative_Wealth_Excess_Pct)
        passed = positive >= 2 and x > 0 and x > p and stress > 0 and news.Tracking_Error_Annualized.max() <= .06
        gates.append({"Allocator": allocator, "Positive_Folds": positive, "Compound_News_Relative_Pct": x,
                      "Compound_Price_Relative_Pct": p, "Stress_Compound_News_Relative_Pct": stress,
                      "Maximum_Tracking_Error": float(news.Tracking_Error_Annualized.max()), "Historical_Rule_Check_Passed": bool(passed)})
    if any(digest(retained / name) != value for name, value in protected.items()) or any(digest(ROOT / name) != value for name, value in source_hashes.items()):
        raise ValueError("Inputs or source changed during experiment")
    holdout = frame[frame.Split.eq("holdout")]
    news = holdout[holdout.Arm.eq("ridge_news")].set_index("Allocator")
    current = news.loc["incremental"]
    summary = f'''# V3.4固定成本配置實驗

已看過歷史上的探索性配對，不重訓、不用留出選擇策略、不替換每日模型。

執行期間：{current.Start}～{current.End}。同一批V3.3價格／新聞預測，費率、20日排程、80%核心與20%衛星限制不變。

新配置新聞淨報酬 **{current.ROI_Pct:.4f}%**，同期0050 **{current.Benchmark_ROI_Pct:.4f}%**，差額 **{current.Net_Excess_ROI_Pp:+.4f}個百分點**。相對舊配置新聞改變 **{current.ROI_Pct - news.loc['legacy', 'ROI_Pct']:+.4f}個百分點**；相同新配置下新聞對價格控制為 **{current.News_Minus_Price_Pp:+.4f}個百分點**。

配置差異包含交易、持倉與現金路徑，不能全部歸因新聞，也不是固定成交單的純費用加回。舊配置三折與留出ROI／回撤／成本全部重播一致，才保留這份新結果。

| 配置器 | 新聞正超額折數 | 新聞驗證累積相對財富（%） | 同配置價格（%） | 新聞壓力情境（%） | 原門檻檢查 |
| --- | ---: | ---: | ---: | ---: | --- |
'''
    for gate in gates:
        summary += f"| {gate['Allocator']} | {gate['Positive_Folds']}/3 | {gate['Compound_News_Relative_Pct']:.4f} | {gate['Compound_Price_Relative_Pct']:.4f} | {gate['Stress_Compound_News_Relative_Pct']:.4f} | {'通過歷史規則，仍非前瞻採用' if gate['Historical_Rule_Check_Passed'] else '未通過'} |\n"
    summary += "\n維持B0／0050與原凍結模型。本實驗不新增歷史新聞可用性證明；原五股選定證據及理想化次日收盤成交的限制保留。20日重疊標籤不是獨立月度樣本。完整Ridge私人前瞻與此實驗是不同資料及任務，不混合。\n"
    frame.to_csv(output / "comparison_metrics.csv", index=False)
    frame.to_csv(publish / "comparison_metrics.csv", index=False)
    (publish / "summary.md").write_text(summary, encoding="utf-8")
    write_json(publish / "rule_checks.json", gates)
    manifest = {"schema": "v3_4_cost_replay_v1", "historical_alpha_proven": False, "no_retraining": True,
                "holdout_used_for_selection": False, "raw_provider_data_included": False,
                "legacy_replay_verified": True, "source_run_manifest_sha256": registration["source_run_manifest_sha256"],
                "files_sha256": {name: digest(publish / name) for name in ("summary.md", "comparison_metrics.csv", "registration.json", "rule_checks.json")}}
    write_json(publish / "manifest.json", manifest)
    write_json(output / "complete.json", manifest)
    return {"complete": True, "no_retraining": True, "selected": "B0_benchmark", "new_news_roi": float(current.ROI_Pct),
            "benchmark_roi": float(current.Benchmark_ROI_Pct), "gap_pp": float(current.Net_Excess_ROI_Pp)}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--retained-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--publish-dir", type=Path, default=ROOT / "reports/research_v3_4/2026-11-17")
    args = parser.parse_args(argv)
    if args.run:
        if args.retained_run is None or args.output_dir is None:
            parser.error("--run requires a retained private source and a new private output")
        result = run_experiment(args.retained_run, args.output_dir, args.publish_dir)
    else:
        result = {"read_only": True, "design": "fixed incremental-cost versus legacy; price/news paired", "trained": False}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
