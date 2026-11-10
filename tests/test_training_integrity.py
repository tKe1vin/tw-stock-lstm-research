"""Training audit counterexamples, including the original second-fit Adam bug."""
import numpy as np
import pytest

from scripts import training_audit
from experiments.v2 import v28


def test_original_notebook_scalers_ignore_future_extreme_features():
    result = training_audit.scaler_future_shock_probe()
    assert result["444"]["scaler_unchanged"]
    assert result["legacy_555"]["training_inputs_unchanged"]
    assert result["historical_news_availability_certified"] is False


def test_reconstructed_sixteen_fold_labels_and_test_membership_are_isolated():
    report = training_audit.fold_manifest(include_indices=True)
    assert report["fold_count"] == 16
    assert report["duplicate_test_signals"] == 0
    seen = set()
    for fold in report["folds"]:
        train, validation, test = (fold[name] for name in ("train", "val", "test"))
        assert train["last_position"] + 6 < validation["first_position"]
        assert validation["last_position"] + 6 < test["first_position"]
        assert not seen.intersection(test["positions"])
        seen.update(test["positions"])
    final = report["folds"][-1]["rolling_756"]
    assert final["signals"] == 756
    assert final["input_rows_if_consecutive"] == 765


def test_two_from_scratch_fits_and_carried_fit_have_live_adam_and_update_weights():
    pytest.importorskip("tensorflow")
    report = training_audit.optimizer_probe()
    assert len(report["fits"]) == 3
    assert report["historical_results_retrained"] is False
    for fit in report["fits"]:
        assert np.isclose(fit["learning_rate"], 0.001)
        assert fit["weight_l2_change"] > 0
        assert fit["fresh_optimizer"] is True
        assert fit["optimizer_iterations"] == 4
        assert np.isclose(fit["best_validation_loss"], fit["restored_validation_loss"], rtol=1e-5)


@pytest.mark.parametrize("origin", ["retrained_walk_forward_v28", None, "mixed"])
def test_corrected_training_rejects_legacy_or_mixed_cached_fits_before_tensorflow(tmp_path, monkeypatch, origin):
    import pandas as pd

    prediction = tmp_path / "fixed_predictions.csv"
    costs = tmp_path / "fixed_cost.csv"
    frame = pd.DataFrame({"Model": ["lstm1_32", "lstm1_50"]})
    if origin is not None:
        frame["Origin"] = [v28.ORIGIN, "retrained_walk_forward_v28"] if origin == "mixed" else origin
    frame.to_csv(prediction, index=False)
    pd.DataFrame({"Model": ["lstm1_32"], "Status": ["fit"]}).to_csv(costs, index=False)
    before = {path: path.read_bytes() for path in (prediction, costs)}
    monkeypatch.setattr(v28, "V28", tmp_path)
    monkeypatch.setattr(v28, "Fitter", lambda: pytest.fail("A legacy cache must be rejected before loading TF"))
    with pytest.raises(RuntimeError, match="Legacy or unverified"):
        v28.train_fixed_models(None, None, None, None, None, None)
    assert {path: path.read_bytes() for path in before} == before


def test_corrected_training_rejects_orphan_costs_instead_of_appending_new_predictions(tmp_path):
    import pandas as pd

    costs = tmp_path / "schedule_cost.csv"
    pd.DataFrame({"Status": ["fit"]}).to_csv(costs, index=False)
    with pytest.raises(RuntimeError, match="Incomplete"):
        v28._assert_corrected_training_cache(tmp_path / "schedule_predictions.csv", costs)
