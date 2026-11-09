"""Publish the shared-core primary research alongside unchanged legacy exports."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.common import ROOT, TZ, atomic_json, sha256
from scripts.paired_research import paired_daily_recommendation, run_paired_research, write_research_outputs
from scripts.paired_analysis import analyze_paired_research, write_analysis_outputs
from scripts.reproducibility import preserve_inputs, restore_inputs


def build_paired_report(baseline_dir, prices, sentiment, destination, *, root=ROOT, now=None):
    baseline_dir, prices, sentiment, destination = (Path(p).resolve() for p in (baseline_dir, prices, sentiment, destination))
    root = Path(root).resolve()
    protocol_path = baseline_dir / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sources = {"predictions": baseline_dir / "test_predictions.csv", "prices": prices,
               "sentiment": sentiment, "baseline_protocol": protocol_path,
               "core_recommendation": baseline_dir / "next_day_recommendation.csv"}
    result = run_paired_research(sources["predictions"], prices, sentiment, protocol)
    recommendation = paired_daily_recommendation(sources["core_recommendation"], prices, sentiment, protocol)
    now = now or datetime.now(TZ)
    day = str(pd.Timestamp(recommendation.Signal_Date.iloc[0]).date())
    recommendation["Created_At"] = now.astimezone(TZ).isoformat()
    recommendation["Run_Mode"] = "daily" if str(now.astimezone(TZ).date()) == day else "historical_replay"
    recommendation["Signal_Date"] = pd.to_datetime(recommendation.Signal_Date).dt.strftime("%Y-%m-%d")
    recommendation["Target_Date"] = pd.to_datetime(recommendation.Target_Date).dt.strftime("%Y-%m-%d")
    write_research_outputs(result, destination)
    analysis = analyze_paired_research(result)
    write_analysis_outputs(analysis, destination / "analysis")
    recommendation.to_csv(destination / "daily_recommendation.csv", index=False, encoding="utf-8-sig")
    recommendation[recommendation.Arm == "dynamic_news_risk555"].to_csv(
        destination / "555_next_day_recommendation.csv", index=False, encoding="utf-8-sig")
    atomic_json(destination / "daily_recommendation.provenance.json", recommendation.attrs["provenance"])
    manifest = preserve_inputs(sources, destination / "reproduction_manifest.json", root=Path(root))
    table = result["metrics"].set_index("Arm")
    lines = ["# 444／555 共同核心配對研究", "",
             f"資料截止：{day}。本頁為主研究；原六特徵 555 的 Notebook 輸出另保留為延伸／歷史診斷。", "",
             "全部配置組使用同一份 444 預測、次一交易日收盤、手續費 0.1425%、賣出稅 0.3%及共同期末處理。",
             "歷史新聞未有完整取得證據，且期間已被查看；結果皆為探索性，不宣稱因果已驗證或前瞻績效。", "",
             "| 組別 | 淨 ROI | Sharpe | MDD | 成本 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["metrics"].itertuples(index=False):
        lines.append(f"| {row.Arm} | {row.ROI_Pct:.2f}% | {row.Sharpe:.3f} | {row.MDD_Pct:.2f}% | {row.Total_Transaction_Cost:.2f} |")
    news_effect = table.loc["dynamic_news_risk555", "ROI_Pct"] - table.loc["dynamic_no_news_risk", "ROI_Pct"]
    lines += ["", f"同風控下新聞組與無新聞組的 ROI 差：{news_effect:+.2f} 個百分點。這是配對策略差異，尚無統計顯著或歷史新聞因果認證。",
              "", "444 固定持股的 ROI 不代表 LSTM 擇時獲利。模型的一日預測在次日收盤進場前已完成，其配置屬延遲訊號策略；首段漲跌未計入新部位收益。",
              "^TWII 為無股利的理論價格參考；該列不扣交易成本，不視為可交易淨報酬。另列同交易機會及同政策總曝險等權組。",
              "", "[逐日 NAV](nav.csv) · [成本與成交](fills.csv) · [權重](weights.csv) · [555 主研究當日配置](555_next_day_recommendation.csv)",
              "", "[研究協議](protocol.json) · [永久輸入快照與程式雜湊](reproduction_manifest.json)",
              "", "[固定配對效果與區塊重抽樣區間](analysis/contrasts.csv) · [按持有部位風險狀態分析](analysis/regime_effects.csv)",
              "", "重播：`python -m scripts.paired_delivery --replay 本目錄/reproduction_manifest.json --output 新目錄`。重播讀取保存預測，不重新訓練。", ""]
    (destination / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    manifest["outputs_sha256"] = {name: sha256(destination / name) for name in ("metrics.csv", "nav.csv", "weights.csv", "fills.csv")}
    atomic_json(destination / "reproduction_manifest.json", manifest)
    return {"status": "complete", "protocol": result["protocol"]["protocol_version"],
            "as_of_date": day, "research_status": "exploratory", "core_model": "444",
            "source_prediction_sha256": manifest["inputs"]["predictions"]["sha256"],
            "report": (destination / "comparison.md").relative_to(root).as_posix(),
            "recommendation": (destination / "555_next_day_recommendation.csv").relative_to(root).as_posix(),
            "news_increment_roi_pp": float(news_effect)}


def publish_comparison_contract(paired_dir, baseline_dir, comparison_dir, *, root=ROOT):
    """Keep comparison CSVs and journals, using the matched ledger for both ROIs."""
    paired_dir, baseline_dir, comparison_dir, root = map(Path, (paired_dir, baseline_dir, comparison_dir, root))
    table = pd.read_csv(paired_dir / "metrics.csv").set_index("Arm")
    protocol = json.loads((paired_dir / "protocol.json").read_text(encoding="utf-8"))
    day = protocol["shared_target_dates"][-1]
    comparisons = []
    for uid, arm in (("444", "static_equal444"), ("555", "dynamic_news_risk555"), ("^TWII benchmark", "market_price_reference")):
        metric = table.loc[arm]
        comparisons.append({"Strategy_ID": uid, "ROI_Pct": metric.ROI_Pct, "Sharpe": metric.Sharpe,
                            "MDD_Pct": metric.MDD_Pct, "Test_Days": metric.Observations, "As_Of_Date": day,
                            "Direction_Accuracy_Pct": np.nan, "Trade_Win_Rate_Pct": np.nan,
                            "Protocol": protocol["protocol_version"], "Return_Basis": metric.Return_Basis})
    comparison_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(comparison_dir / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    core_metrics = pd.read_csv(baseline_dir / "model_metrics.csv")
    core_metrics = core_metrics[core_metrics.Ticker.isin(protocol["tickers"])].copy()
    per_stock = core_metrics.merge(core_metrics, on="Ticker", suffixes=("_444", "_555"), validate="one_to_one")
    per_stock["Shared_Core_Model"] = "444"
    per_stock.to_csv(comparison_dir / "per_stock_comparison.csv", index=False, encoding="utf-8-sig")
    atomic_json(comparison_dir / "comparison_readme.json", {
        "comparison_status": "paired_444_core_next_close", "as_of_date": day,
        "protocol": protocol["protocol_version"], "core_model": "444", "result_status": "exploratory",
        "identical_model_metrics_expected": True, "trade_win_rate": "undefined",
        "original_444_native_roi": "legacy descriptive fixed-share ledger; main ROI recomputed through the paired next-close ledger",
        "market": "theoretical price reference only; not an investable net benchmark",
    })
    import os

    link = os.path.relpath(paired_dir / "comparison.md", comparison_dir).replace(os.sep, "/")
    (comparison_dir / "comparison.md").write_text(
        f"# 444／555 共同核心比較\n\n兩組使用同一 444 核心、次日收盤與相同成本。\n\n[查看完整消融、NAV、成本及限制]({link})\n",
        encoding="utf-8")
    rec = pd.read_csv(paired_dir / "daily_recommendation.csv")
    daily = root / "reports/daily" / day
    daily.mkdir(parents=True, exist_ok=True)
    for uid, arm in (("444", "static_equal444"), ("555", "dynamic_news_risk555")):
        row = table.loc[arm]
        subset = rec[rec.Arm == arm]
        text = (f"## {day} — {uid} 配對主研究\n\n"
                f"共同核心：444；協議：{protocol['protocol_version']}；歷史結果為探索性。\n\n"
                f"次日收盤帳本 ROI {row.ROI_Pct:+.2f}%；Sharpe {row.Sharpe:.3f}；MDD {row.MDD_Pct:.2f}%。\n\n"
                "一日模型方向預測不等於成交後報酬，也不等於逐筆交易勝率。\n\n"
                "| 股票 | 核心預測次日報酬 | 目標權重 |\n| --- | ---: | ---: |\n")
        for item in subset.itertuples():
            text += f"| {item.Ticker} | {item.Predicted_Return_Pct:+.3f}% | {item.Target_Weight:.2%} |\n"
        text += "\n配置僅作研究，尚未下單或成交；固定持股組的20%是初始參考，非每日再平衡。\n"
        (daily / f"{uid}.md").write_text(text, encoding="utf-8")
        journal = root / "reports/journals" / f"{uid}.md"
        journal.parent.mkdir(parents=True, exist_ok=True)
        old = journal.read_text(encoding="utf-8") if journal.exists() else f"# {uid} 研究日誌\n"
        marker = f"## {day} — {uid} 配對主研究"
        if marker in old:
            before, rest = old.split(marker, 1)
            next_section = rest.find("\n## ")
            old = before + (rest[next_section:] if next_section >= 0 else "")
        title, _, body = old.partition("\n")
        journal.write_text(title + "\n\n" + text + "\n" + body, encoding="utf-8")
    import shutil

    for name in ("comparison.md", "strategy_comparison.csv", "per_stock_comparison.csv", "comparison_readme.json"):
        shutil.copy2(comparison_dir / name, daily / name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--sentiment", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.replay:
        import hashlib

        recorded = json.loads(args.replay.read_text(encoding="utf-8"))
        for name in ("scripts/paired_research.py", "scripts/common.py"):
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != recorded["source_sha256"].get(name):
                raise ValueError(f"Replay source differs from the recorded research revision: {name}")
        paths = restore_inputs(args.replay, args.output / "replay_inputs")
        baseline = json.loads(paths["baseline_protocol"].read_text(encoding="utf-8"))
        result = run_paired_research(paths["predictions"], paths["prices"], paths["sentiment"], baseline)
        write_research_outputs(result, args.output)
        write_analysis_outputs(analyze_paired_research(result), args.output / "analysis")
        for name, expected in recorded.get("outputs_sha256", {}).items():
            if sha256(args.output / name) != expected:
                raise ValueError(f"Replay output differs from the recorded result: {name}")
        print(json.dumps({"status": "replayed", "output": str(args.output)}))
    else:
        if any(v is None for v in (args.baseline_dir, args.prices, args.sentiment)):
            parser.error("provide --baseline-dir, --prices and --sentiment, or --replay")
        print(json.dumps(build_paired_report(args.baseline_dir, args.prices, args.sentiment, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
