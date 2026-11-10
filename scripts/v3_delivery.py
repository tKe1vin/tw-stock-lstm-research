"""Publish derived V3 research evidence without redistributing provider inputs.

FinMind source snapshots remain in the authorized local research cache. Public
replay requires separately acquired, byte-identical normalized inputs; a clean
clone alone does not contain those source data.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import research_v3 as v3


def export_public(run_dir, destination):
    run, output = Path(run_dir).resolve(), Path(destination).resolve()
    v3.verify(run)
    if output.exists():
        raise FileExistsError("Public research exports are immutable")
    for name in ("data", "models", "outputs", "experiments/results", "research/v211"):
        if output.is_relative_to(v3.ROOT / name):
            raise ValueError("Cannot export into production state")
    output.mkdir(parents=True)
    allowed = ["registered_config.json", "selection.json", "validation_metrics.csv",
               "holdout_metrics.csv", "splits.json", "limitations.json", "data_quality.json"]
    allowed += [path.name for path in sorted(run.glob("*.json.gz"))]
    for name in allowed:
        shutil.copyfile(run / name, output / name)
    retained = json.loads((run / "reproduction_manifest.json").read_text())
    metadata = json.loads((run / "inputs/bars.metadata.json").read_text())
    provenance = {
        "provider": "FinMind", "provider_terms": "https://finmind.github.io/Disclaimer/",
        "raw_data_redistributed": False, "normalized_bars_redistributed": False,
        "clean_clone_contains_all_inputs": False,
        "replay_requires_authorized_external_inputs": True,
        "input_sha256": {name: retained["files_sha256"]["inputs/" + name]
                         for name in ("bars.csv", "bars.metadata.json", "sentiment.csv")},
        "private_run_manifest_sha256": v3.digest(run / "reproduction_manifest.json"),
        "runtime_versions": retained["runtime_versions"], "python": retained["python"],
        "provider_fetched_at": metadata["fetched_at"],
        "close_basis": metadata["close_basis"],
        "raw_manifest_sha256": metadata["raw_manifest_sha256"],
        "reviewed_suspensions": metadata["suspensions"],
        "reviewed_action_overrides": metadata["action_date_overrides"],
        "point_in_time_certified": metadata["point_in_time_certified"],
        "official_exhaustive_action_coverage_certified": metadata["official_exhaustive_action_coverage_certified"],
        "limitations": ["Provider revisions can prevent later exact source reconstruction.",
                        "Hashes show integrity; they do not prove historical information availability."]}
    v3.write_json(output / "input_provenance.json", provenance)
    holdout = pd.read_csv(run / "holdout_metrics.csv")
    validation = pd.read_csv(run / "validation_metrics.csv")
    selection = json.loads((run / "selection.json").read_text())
    selected = selection["selected"]
    baseline = holdout[holdout.Arm.eq("B0_benchmark")].iloc[0]
    rows = ["# V3：0050 相對新聞模型探索結果", "", "日期：2026-11-10（臺北）。",
            "", "這是既有歷史上的探索性比較，不是前瞻績效，也沒有保證超越 0050。",
            f"留出期共同執行區間：**{baseline['Start']}～{baseline['End']}**。",
            "", f"三折驗證採用結果：**{selected}**。選擇在任何留出期模型訓練／評分前寫入。",
            "若結果為 `B0_benchmark`，下列新聞模型只作失敗診斷，不因留出表現改選。", "",
            "| 留出期組別 | 淨 ROI | 相對 0050 財富增長差 | 年化追蹤誤差 |", "| --- | ---: | ---: | ---: |"]
    for row in holdout.to_dict("records"):
        rows.append(f"| {row['Arm']} | {row['ROI_Pct']:.4f}% | {row['Relative_Wealth_Excess_Pct']:.4f}% | {row['Tracking_Error_Annualized']*100:.4f}% |")
    rows += ["", "相對財富差定義為 `(策略期末淨值 / 0050期末淨值 - 1) × 100`；不是兩個 ROI 的相減。",
             "入場與期末成本皆計入；0050 基準持續持有，僅依共同五日排程投入已可使用的現金，不強制賣出重配。與策略同期間、同資本、同資產費率。",
             "", "## 規格與證據邊界", "",
             "- 四候選：價格／新聞 Ridge、價格／新聞 LSTM。三折各 60 個訊號日，train 上限 756、至少 252；實際日期與樣本數見 splits.json、模型檔。",
             "- 目標是下一交易日收盤至其後五日的個股含息報酬，減同期間 0050 含息報酬，單位為百分點。",
             "- 0050 核心目標至少 80%、股票衛星最多 20%，長倉、不借款；新聞未支付預計完整換倉成本時回到 0050。實際權重可隨價格與費用漂移。",
             "- 訊號日固定整數股委託、次日收盤理想化成交；手續費 0.1425%、最低20元、股票賣稅0.3%、ETF賣稅0.1%，每邊5bps滑價及15bps壓力情境。非券商成交紀錄。",
             "- 公司行動調整股數；有付款日時股利先列應收、付款後才能花用，任何未知日期或期末未平權益另行揭露。",
             "- 所有歷史新聞的實際當時可用性尚未完整證明；現有五股池也沒有研究前預選的證據，不宣稱消除了存活者或選股偏差。",
             "- 最後60個訊號日仍是探索性留出；真正驗證需完成來源與執行查核、封存後才累積新觀測。沒有部署或替換既有每日模型。",
             "", "## 重現", "",
             "本公開結果不含 FinMind 原始回覆或完整行情，避免違反供應商禁止再散布的條款。模型、設定、切分與輸入 SHA-256 已保留；單靠 clone 無法取得全部固定來源。",
             "先依供應商條款取得本機快照，再檢查 input_provenance.json 的行情與情緒雜湊；不同版本會拒絕精確重播。",
             "", "```bash", "python -m scripts.v3_delivery --replay reports/research_v3/2026-11-10 --data /your/cache/bars.csv --metadata /your/cache/bars.meta.json --sentiment /your/cache/sentiment.csv", "```", "",
             "各候選三折完整結果見 [validation_metrics.csv](validation_metrics.csv)，採用／拒絕理由見 [selection.json](selection.json)。"]
    (output / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {"source_sha256": retained["source_sha256"],
                "exporter_sha256": v3.digest(Path(__file__)),
                "files_sha256": {str(path.relative_to(output)): v3.digest(path)
                                 for path in sorted(output.rglob("*")) if path.is_file()}}
    v3.write_json(output / "reproduction_manifest.json", manifest)
    return {"exported": str(output), "provider_inputs_redistributed": False, "selected": selected}


def replay(public_run, data_path, metadata_path, sentiment_path):
    """Rebuild the retained final forecasts/books without fitting or selecting."""
    from research.v3.data import load_dataset
    from research.v3.models import prepare_features, predict_saved
    output = Path(public_run)
    v3.verify(output)
    manifest = json.loads((output / "reproduction_manifest.json").read_text())
    if manifest.get("exporter_sha256") != v3.digest(Path(__file__)):
        raise ValueError("V3 public replay source differs from the retained exporter")
    provenance = json.loads((output / "input_provenance.json").read_text())
    for name, path in (("bars.csv", data_path), ("sentiment.csv", sentiment_path)):
        if v3.digest(path) != provenance["input_sha256"][name]:
            raise ValueError(f"External V3 replay input changed: {name}")
    # A later download time may change the metadata bytes. Its input hash and
    # conventions are independently checked by load_dataset against bars.csv.
    config = v3.load_config(output / "registered_config.json")
    bars = load_dataset(data_path, metadata_path=metadata_path,
                        expected_tickers=["0050.TW"] + config["satellite_tickers"])
    features, _, _ = prepare_features(bars, pd.read_csv(sentiment_path), benchmark="0050.TW",
                                     tickers=config["satellite_tickers"], time_step=config["time_step"])
    split = json.loads((output / "splits.json").read_text())
    dates = pd.DatetimeIndex(split["test"])
    saved = pd.read_csv(output / "holdout_metrics.csv").set_index("Arm")
    baseline = v3._book(bars, pd.DataFrame(), dates, config, mode="benchmark")
    deltas = {}
    for path in sorted(output.glob("final_*.json.gz")):
        artifact = json.loads(gzip.decompress(path.read_bytes()))
        artifact["arrays"] = {name: np.asarray(value) for name, value in artifact["arrays"].items()}
        candidate = artifact["metadata"]["candidate"]
        predictions = predict_saved(features, dates, artifact)
        book = v3._book(bars, predictions, dates, config)
        actual = {**book["metrics"], **v3.relative_metrics(book["nav"], baseline["nav"])}
        errors = {name: abs(float(actual[name]) - float(saved.loc[candidate, name]))
                  for name in ("ROI_Pct", "Relative_Wealth_Excess_Pct", "Tracking_Error_Annualized")}
        if max(errors.values()) > 1e-6:
            raise ValueError(f"Saved V3 book reconstruction changed: {candidate}: {errors}")
        deltas[candidate] = errors
    if not deltas:
        raise ValueError("No retained final candidate artifacts")
    return {"reproduced": True, "fit_used": False, "selection_used": False,
            "maximum_deltas": deltas, "historical_alpha_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--sentiment", type=Path)
    args = parser.parse_args()
    if args.export and args.output_dir and not args.replay:
        result = export_public(args.export, args.output_dir)
    elif args.replay and all((args.data, args.metadata, args.sentiment)) and not args.export:
        result = replay(args.replay, args.data, args.metadata, args.sentiment)
    else:
        parser.error("Use --export RUN --output-dir NEW or --replay RUN --data --metadata --sentiment")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
