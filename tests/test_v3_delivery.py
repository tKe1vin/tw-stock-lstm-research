"""Public evidence excludes restricted provider data and fails on input drift."""
import json

import pandas as pd
import pytest

from scripts import research_v3 as v3
from scripts import v3_delivery as delivery


def fake_run(tmp_path):
    run = tmp_path / "local"
    (run / "inputs/raw").mkdir(parents=True)
    (run / "inputs/raw/provider.json").write_text("restricted-provider-response")
    metadata = {"fetched_at": "2026-11-10T00:00:00+00:00", "close_basis": "truly_raw_unadjusted",
                "raw_manifest_sha256": "a" * 64, "suspensions": [], "action_date_overrides": [],
                "point_in_time_certified": False, "official_exhaustive_action_coverage_certified": False}
    v3.write_json(run / "inputs/bars.metadata.json", metadata)
    (run / "inputs/bars.csv").write_text("restricted-normalized-bars")
    (run / "inputs/sentiment.csv").write_text("external-sentiment")
    for name in ("registered_config.json", "splits.json", "limitations.json", "data_quality.json"):
        v3.write_json(run / name, {})
    v3.write_json(run / "selection.json", {"selected": "B0_benchmark"})
    (run / "validation_metrics.csv").write_text("Candidate,Fold\nridge_news,1\n")
    pd.DataFrame([{"Arm": "B0_benchmark", "ROI_Pct": 1., "Relative_Wealth_Excess_Pct": 0.,
                   "Tracking_Error_Annualized": 0., "Start": "2026-08-04", "End": "2026-11-03"}]).to_csv(run / "holdout_metrics.csv", index=False)
    manifest = {"files_sha256": {"inputs/" + name: v3.digest(run / "inputs" / name)
                                  for name in ("bars.csv", "bars.metadata.json", "sentiment.csv")},
                "runtime_versions": {}, "python": "3.12", "source_sha256": {}}
    v3.write_json(run / "reproduction_manifest.json", manifest)
    return run


def test_public_export_retains_evidence_without_provider_response_or_bars(tmp_path, monkeypatch):
    run = fake_run(tmp_path)
    monkeypatch.setattr(v3, "verify", lambda _: {"verified": True})
    output = tmp_path / "public"
    result = delivery.export_public(run, output)
    assert not result["provider_inputs_redistributed"]
    assert not (output / "inputs").exists()
    assert not list(output.rglob("bars.csv"))
    assert all(b"restricted-provider-response" not in path.read_bytes() and
               b"restricted-normalized-bars" not in path.read_bytes()
               for path in output.rglob("*") if path.is_file())
    provenance = json.loads((output / "input_provenance.json").read_text())
    assert provenance["input_sha256"]["bars.csv"] == v3.digest(run / "inputs/bars.csv")
    assert not provenance["clean_clone_contains_all_inputs"]
    with pytest.raises(FileExistsError):
        delivery.export_public(run, output)


def test_replay_rejects_changed_external_input_before_features_or_training(tmp_path, monkeypatch):
    run = fake_run(tmp_path)
    monkeypatch.setattr(v3, "verify", lambda _: {"verified": True})
    output = tmp_path / "public"
    delivery.export_public(run, output)
    changed = tmp_path / "changed.csv"
    changed.write_text("different-bars")
    with pytest.raises(ValueError, match="External V3 replay input changed"):
        delivery.replay(output, changed, run / "inputs/bars.metadata.json", run / "inputs/sentiment.csv")


def test_replay_rejects_changed_exporter_source(tmp_path, monkeypatch):
    run = fake_run(tmp_path)
    monkeypatch.setattr(v3, "verify", lambda _: {"verified": True})
    output = tmp_path / "public"
    delivery.export_public(run, output)
    manifest = json.loads((output / "reproduction_manifest.json").read_text())
    manifest["exporter_sha256"] = "0" * 64
    v3.write_json(output / "reproduction_manifest.json", manifest)
    with pytest.raises(ValueError, match="public replay source differs"):
        delivery.replay(output, None, None, None)
