"""V2.12 research: read-only checks, explicit new training, and retained replay."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

from scripts.common import ROOT, TZ

REPORT = "reports/research_v2_common/2026-11-17"
PAIRED = "reports/daily/2026-11-17/paired"
MODEL_PACK = "research/v2_common/models/2026-11-17/fold_models.zip"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def inside(root, relative):
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Research source or artifact escapes its retained root")
    return path


@contextmanager
def retained_inputs(root):
    """Restore an exact old snapshot only to a temporary, non-repository folder."""
    root = Path(root)
    manifest = json.loads((root / PAIRED / "reproduction_manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="v2-common-retained-") as temporary:
        paths = {}
        for name in ("prices", "sentiment"):
            record = manifest["inputs"][name]
            original = inside(root, record["original_path"])
            if original.is_file() and digest(original) == record["sha256"]:
                paths[name] = original
                continue
            raw = gzip.decompress(inside(root, record["archive"]).read_bytes())
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Retained input archive differs: {name}")
            path = Path(temporary) / f"{name}.csv"
            path.write_bytes(raw)
            paths[name] = path
        for name, expected in manifest["outputs_sha256"].items():
            if digest(inside(root / PAIRED, name)) != expected:
                raise ValueError(f"Canonical 444/555 output changed: {name}")
        yield paths, manifest


def verify_report(root=ROOT):
    root = Path(root)
    directory = root / REPORT
    path = directory / "manifest.json"
    if not path.is_file():
        return {"available": False, "read_only": True, "status": "historical_report_not_created"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "v2-common-public-v1":
        raise ValueError("Unrecognized V2 common research manifest")
    for name, expected in manifest["files"].items():
        if digest(inside(directory, name)) != expected:
            raise ValueError(f"Published research artifact differs: {name}")
    pack = manifest["models"]
    if digest(inside(root, pack["archive_path"])) != pack["sha256"]:
        raise ValueError("Retained fold model pack differs")
    return {"available": True, "read_only": True, "verified_files": len(manifest["files"]),
            "model_count": pack["model_count"], "summary": json.loads((directory / "summary.json").read_text(encoding="utf-8"))}


def pack_models(run, destination):
    """Retain every inference artifact; do not add raw provider responses."""
    from scripts.v2_common_training import verify_run

    run, destination = Path(run), Path(destination)
    manifest = verify_run(run)
    allowed = set(manifest["files_sha256"]) | {"reproduction_manifest.json"}
    if any(Path(name).suffix.lower() not in {".json", ".npz", ".csv"} for name in allowed):
        raise ValueError("Model pack includes an unexpected artifact type")
    if any("prices" in Path(name).name.lower() or "sentiment.csv" == Path(name).name for name in allowed):
        raise ValueError("Raw provider inputs cannot enter the model pack")
    if destination.exists():
        raise FileExistsError("Use a new model pack path; retained evidence is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(allowed):
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, inside(run, name).read_bytes())


@contextmanager
def unpack_models(root=ROOT):
    root = Path(root)
    verified = verify_report(root)
    if not verified["available"]:
        raise FileNotFoundError("No published historical model pack")
    with tempfile.TemporaryDirectory(prefix="v2-common-models-") as temporary:
        with zipfile.ZipFile(root / MODEL_PACK) as archive:
            for entry in archive.infolist():
                destination = inside(temporary, entry.filename)
                if entry.is_dir():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(entry))
        yield Path(temporary)


def news_coverage(data):
    raw = pd.read_csv(data["source_paths"]["sentiment"])
    raw["Date"] = pd.to_datetime(raw.Date)
    rows = []
    for fold in data["folds"]:
        dates = data["calendar"][fold["test"]]
        part = raw[raw.Date.isin(dates)].copy()
        status = part.get("Source_Status", pd.Series(index=part.index, dtype=object)).fillna("").astype(str)
        timestamp = part.get("Latest_News_Timestamp", pd.Series(index=part.index, dtype=object))
        rows.append({"Fold": fold["Fold"], "Signal_Days": len(dates), "Stock_Rows": len(part),
                     "Nonzero_Scores": int(part.Daily_Sentiment.ne(0).sum()),
                     "Unknown_Source_Status_Rows": int(status.str.strip().eq("").sum()),
                     "Observed_Article_Rows": int(status.eq("observed_article").sum()),
                     "Verified_No_Article_Rows": int(status.eq("verified_no_matching_article").sum()),
                     "Publication_Timestamp_Rows": int(timestamp.notna().sum()),
                     "Historical_Availability_Certified": False})
    return pd.DataFrame(rows)


def publish(root, run):
    from scripts.v2_common_books import evaluate_common_books
    from scripts.v2_common_training import load_config, prepare_data, replay_predictions, verify_run

    root, run = Path(root), Path(run)
    directory, pack = root / REPORT, root / MODEL_PACK
    if directory.exists() or pack.exists():
        raise FileExistsError("Published research/model artifacts cannot be overwritten")
    training_manifest = verify_run(run)
    config = load_config(run / "registered_config.json")
    with retained_inputs(root) as (paths, canonical):
        for key, path in paths.items():
            if digest(path) != training_manifest["external_sources"][key]["sha256"]:
                raise ValueError("Training source differs from the canonical 444/555 snapshot")
        data = prepare_data(root, config, paths["prices"], paths["sentiment"], root / PAIRED / "protocol.json")
        replay = replay_predictions(run, root, paths["prices"], paths["sentiment"], root / PAIRED / "protocol.json")
        predictions = pd.read_csv(run / "predictions.csv")
        paired_protocol = json.loads((root / PAIRED / "protocol.json").read_text(encoding="utf-8"))
        result = evaluate_common_books(data["panel"], data["sentiment"], predictions, paired_protocol,
                                      pd.read_csv(root / PAIRED / "nav.csv"), pd.read_csv(root / PAIRED / "metrics.csv"))
        result["news_coverage"] = news_coverage(data)
    pack_models(run, pack)
    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(directory / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    predictions.to_csv(directory / "predictions.csv", index=False)
    test = predictions[predictions.Split.eq("test") & predictions.Model.eq("news_lstm")]
    summary = {"version": "V2.12", "study_status": "historical_exploratory", "period_start": "2026-04-10",
               "period_end": "2026-11-17", "as_of": config["as_of"], "valuation_sessions": 150,
               "test_signal_days": int(test.Signal_Date.nunique()),
               "matured_signal_days": int(test.loc[test.Label_State.eq("MATURED"), "Signal_Date"].nunique()),
               "pending_signal_days": int(test.loc[test.Label_State.eq("PENDING"), "Signal_Date"].nunique()),
               "forecasts_per_model": len(test), "matured_rows_per_model": int(test.Label_State.eq("MATURED").sum()),
               "pending_rows_per_model": int(test.Label_State.eq("PENDING").sum()), "fold_count": 3,
               "initial_capital": 100000, "benchmark_arm": "static_equal444", "model_arm": "v2_news_lstm",
               "matched_arm": "v2_matched_equal", "fresh_model_fits": 45,
               "reproducible_model_pack": True, "historical_news_point_in_time_proven": False,
               "independent_holdout": False, "test_used_for_selection": False, "production_replaced": False,
               "model_replay": replay, "created_at": datetime.now(TZ).isoformat()}
    write_json(directory / "summary.json", summary)
    write_json(directory / "protocol.json", result["protocol"])
    write_json(directory / "model_replay.json", replay)
    rows = result["metrics"].set_index("Arm")
    lines = ["# V2.12 與 444／555 同期間歷史回測", "", "資料截止：2026-11-17；本次歷史重訓日期：2026-11-18。",
             "", "150 個共同估值日；三折新訓練，45 份新權重與 Scaler。每個候選 145 個成熟訊號日、725 筆成熟股票觀測；5 日、25 筆 pending 不填零。",
             "", "共同資本100,000元、分數股、次日收盤、手續費0.1425%、股票賣稅0.3%、期末清算；未含最低費用、滑價、股利及公司行動。",
             "444固定股數、555每日配置、V2每五日配置；三者任務及頻率差異保留，新聞增量只在相同V2任務內比較。",
             "共同期末會強制平倉，最後一期可能未持滿五日；帳本清算不等於五日預測標籤成熟。",
             "", "| 組別 | 淨ROI | MDD | 成本（元） |", "| --- | ---: | ---: | ---: |"]
    for row in result["metrics"].itertuples():
        lines.append(f"| {row.Display_Name} | {row.ROI_Pct:+.4f}% | {row.MDD_Pct:.4f}% | {row.Total_Transaction_Cost:.2f} |")
    news, control = rows.loc["v2_news_lstm"], rows.loc["v2_no_news_lstm"]
    lines += ["", f"新聞模型對同架構無新聞模型的淨ROI差：{news.ROI_Pct - control.ROI_Pct:+.4f}個百分點。",
              "配對日報酬的不確定區間見 paired_effects.csv；歷史新聞時點未完整認證，這不是獨立前瞻或因果效果。",
              "", "測試期間已被觀察，全部結果保持探索性，不以本輪報酬選新贏家或替換每日V2.11。",
              "新聞750列中680列缺來源狀態，零分不等於查詢完成。^TWII非可交易ETF；本研究不宣稱超越0050。",
              "", f"已保存模型推論重播最大差異：{replay['maximum_prediction_delta']:.12g}。重新訓練不保證逐位元相同；保存權重推論已核對。",
              "", "[完整設定與限制](../../../docs/v2_common_backtest.md) · [來源與產物雜湊](manifest.json) · [配對效果](paired_effects.csv)", ""]
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    sources = {f"{PAIRED}/{name}": digest(root / PAIRED / name) for name in
               ("protocol.json", "reproduction_manifest.json", "nav.csv", "metrics.csv")}
    sources.update({record["archive"]: digest(root / record["archive"]) for record in canonical["inputs"].values()})
    manifest = {"schema": "v2-common-public-v1", "period_start": summary["period_start"], "period_end": summary["period_end"],
                "initial_capital": 100000, "sources": sources,
                "analysis_code_sha256": {name: digest(root / name) for name in
                    ("scripts/research_v2_common.py", "scripts/v2_common_books.py")},
                "models": {"archive_path": MODEL_PACK, "sha256": digest(pack), "model_count": 45, "scaler_count": 45,
                           "bytes": pack.stat().st_size, "format": "lossless NPZ weights plus scaler/architecture/runtime and reproduction manifest"},
                "files": {path.name: digest(path) for path in directory.iterdir() if path.is_file()},
                "raw_inputs_redistributed": False, "historical_availability_certified": False}
    write_json(directory / "manifest.json", manifest)
    verify_report(root)
    return summary


def replay_research(root, run, paths):
    """Recompute retained model predictions and every derived financial table."""
    from scripts.v2_common_books import evaluate_common_books
    from scripts.v2_common_training import load_config, prepare_data, replay_predictions

    root, run = Path(root), Path(run)
    directory = root / REPORT
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("analysis_code_sha256", {}).items():
        if digest(inside(root, name)) != expected:
            raise ValueError(f"Analysis source differs from the retained research revision: {name}")
    inference = replay_predictions(run, root, paths["prices"], paths["sentiment"], root / PAIRED / "protocol.json")
    config = load_config(run / "registered_config.json")
    data = prepare_data(root, config, paths["prices"], paths["sentiment"], root / PAIRED / "protocol.json")
    protocol = json.loads((root / PAIRED / "protocol.json").read_text(encoding="utf-8"))
    result = evaluate_common_books(data["panel"], data["sentiment"], pd.read_csv(run / "predictions.csv"), protocol,
                                  pd.read_csv(root / PAIRED / "nav.csv"), pd.read_csv(root / PAIRED / "metrics.csv"))
    result["news_coverage"] = news_coverage(data)
    verified = []
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            raw = frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != manifest["files"].get(name + ".csv"):
                raise ValueError(f"Recomputed financial table differs: {name}")
            verified.append(name + ".csv")
    return {**inference, "financial_tables_recomputed": len(verified), "financial_csv_bytes_identical": True,
            "verified_tables": verified, "model_pack_used": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "train", "publish", "replay"), default="check")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "check":
        result = verify_report(ROOT)
    elif args.mode in ("train", "publish"):
        if args.run_dir is None:
            parser.error("Use an explicit new --run-dir for training or an existing retained run for publishing")
        if args.mode == "train":
            from scripts.v2_common_training import load_config, prepare_data, train
            with retained_inputs(ROOT) as (paths, _):
                config = load_config()
                data = prepare_data(ROOT, config, paths["prices"], paths["sentiment"], ROOT / PAIRED / "protocol.json")
                train(data, config, args.run_dir, ROOT / "research/v2_common/config.json")
        result = publish(ROOT, args.run_dir)
    else:
        with retained_inputs(ROOT) as (paths, _), unpack_models(ROOT) as run:
            result = replay_research(ROOT, run, paths)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
