"""Explicitly export only derived portfolio wealth for the read-only dashboard.

Private provider caches are not copied. The original private run manifest must
match the already published manifest hash before any source NAV is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.data import (ARM_NAMES, ASSETS, ROOT, SCHEMA, STUDIES, REPORT_FILES,
                            DataIntegrityError, load_study, sha256, validate_curves, validate_metrics)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_assets(run_dirs, *, root=ROOT, output_dir=None):
    """Create a new immutable presentation snapshot after checking both studies."""
    root = Path(root).resolve()
    output = Path(output_dir or root / ASSETS).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("展示快照不可覆寫，請選擇新的空目錄")
    manifest = {"schema": SCHEMA, "created_on": "2026-11-17", "raw_provider_data_included": False,
                "curve_definition": "Wealth_Index = 100 × saved portfolio NAV / common initial capital; entry costs retained. Drawdown includes initial wealth100.",
                "provider_source_note": "Only derived portfolio wealth is redistributed, no raw/normalized provider price panel, orders, quotes or article text.",
                "cross_version_comparison": "Periods differ. Never combine curves or interpret cross-version ROI ranking as improvement.",
                "studies": {}}
    payloads = {}
    if set(run_dirs) != set(STUDIES):
        raise DataIntegrityError("必須明確提供兩個已保留研究版本")
    for version, spec in STUDIES.items():
        run, report = Path(run_dirs[version]).resolve(), root / spec["report_dir"]
        published = _load(report / "input_provenance.json")
        run_manifest = run / "reproduction_manifest.json"
        if sha256(run_manifest) != published["private_run_manifest_sha256"]:
            raise DataIntegrityError("本機研究來源manifest與公開來源hash不符")
        retained = _load(run_manifest)
        metrics_path = report / "holdout_metrics.csv"
        if sha256(metrics_path) != retained["files_sha256"]["holdout_metrics.csv"]:
            raise DataIntegrityError("本機研究績效與公開績效不一致")
        metrics = validate_metrics(pd.read_csv(metrics_path))
        parts, hashes = [], {}
        for arm in ARM_NAMES:
            relative = f"holdout_{arm}/nav.csv"
            path = run / relative
            if sha256(path) != retained["files_sha256"].get(relative):
                raise DataIntegrityError("已保存的投資組合NAV被更改")
            nav = pd.read_csv(path)
            if not {"Date", "NAV"}.issubset(nav) or not np.isfinite(nav.NAV.to_numpy(float)).all() or (nav.NAV <= 0).any():
                raise DataIntegrityError("NAV缺欄位、缺值或非正")
            capital = float(metrics.loc[metrics.Arm.eq(arm), "Initial_Capital"].iloc[0])
            wealth = nav.NAV.to_numpy(float) / capital * 100
            peak = np.maximum.accumulate(np.r_[100., wealth])[1:]
            parts.append(pd.DataFrame({"Date": nav.Date, "Arm": arm, "Wealth_Index": wealth,
                                       "Drawdown_Pct": (wealth / peak - 1) * 100}))
            hashes[relative] = sha256(path)
        curves = validate_curves(pd.concat(parts, ignore_index=True), metrics)
        encoded = curves.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g").encode()
        payloads[spec["curves_file"]] = encoded
        manifest["studies"][version] = {
            "curves_file": spec["curves_file"], "curves_sha256": hashlib.sha256(encoded).hexdigest(),
            "report_sha256": {spec["report_dir"] + "/" + name: sha256(report / name) for name in REPORT_FILES},
            "private_run_manifest_sha256": sha256(run_manifest), "retained_derived_nav_sha256": hashes,
            "initial_capital": float(metrics.Initial_Capital.iloc[0]),
            "period_start": str(metrics.Start.iloc[0])[:10], "period_end": str(metrics.End.iloc[0])[:10],
            "arms": list(ARM_NAMES), "rows": len(curves), "derived_only": True}
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    # Finish verification before the first write; no report or model is a destination.
    for entry in manifest["studies"].values():
        if any(sha256(root / path) != wanted for path, wanted in entry["report_sha256"].items()):
            raise DataIntegrityError("公開來源在匯出期間變動")
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return {"created": True, "studies": list(STUDIES), "raw_provider_data_included": False,
            "files": list(payloads)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--v33-run", type=Path)
    parser.add_argument("--v32-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.write:
        if not args.v33_run or not args.v32_run:
            parser.error("--write requires both authorized retained run directories")
        value = build_assets({"v3.3": args.v33_run, "v3.2": args.v32_run}, output_dir=args.output_dir)
    else:
        value = {"read_only": True, "studies": {version: len(load_study(version)["curves"]) for version in STUDIES}}
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
