"""Registry consistency and sample counts cannot certify missing publication proof."""
from copy import deepcopy
from pathlib import Path

from dashboard_v2.data_access import prospective as data


def test_live_v211_cannot_certify_old_availability_even_with_enough_reported_samples(monkeypatch):
    root = data.ROOT / "experiments/results/v2/v211"
    monkeypatch.setenv("TW_V211_LOCAL", str(root))
    original = data.json_file

    def more_samples(path):
        payload = original(path)
        if Path(path) == root / "status.json":
            payload = deepcopy(payload)
            payload["scoring"]["matured_signal_days"] = 60
        return payload

    monkeypatch.setattr(data, "json_file", more_samples)
    result = data.v211_status_local()
    assert result["registry_ok"] is True and result["weight_ok"] is True
    assert result["matured_signal_days"] == 60
    assert result["historical_availability_certified"] is False
    assert result["performance"] == "PENDING" and result["official_claim_blocked"] is True


def test_ridge_reader_reports_unverified_complete_chain_instead_of_certifying_a_counter():
    result = data.ridge_status()
    assert result["forward_evidence_verified"] is False
    assert result["official_claim_blocked"] is True
    assert "full_forward_evidence_not_verified" in result["integrity_reasons"]
