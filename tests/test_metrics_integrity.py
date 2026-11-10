"""Neutral directions and complete paired cross-sections cannot inflate skill."""
import numpy as np
import pandas as pd
import pytest

from experiments.v2.v28_rules import classification_scores
from experiments.horizon_experiment import choose_model, flat_skill, rank_metrics
from scripts import research_extension as extension


def candidate_rows():
    rows = []
    for model in ("price_lstm", "sentiment_risk_lstm"):
        for date in ("2025-01-02", "2025-01-03"):
            for i, ticker in enumerate(extension.TICKERS):
                rows.append({"Model": model, "Split": "validation", "Signal_Date": date,
                             "Ticker": ticker, "Predicted": float(i), "Actual_5d_Pct": float(i),
                             "Fold": 1, "Execution_Date": date, "Label_End": "2025-01-13"})
    return pd.DataFrame(rows)


def test_neutral_prediction_is_incorrect_for_both_actual_nonzero_classes():
    result = classification_scores([0, 0], [1, -1])
    assert result["Direction_Accuracy"] == 0
    assert result["Balanced_Accuracy"] == 0
    assert result["True_Positive_Rate"] == 0
    assert result["True_Negative_Rate"] == 0
    assert result["Prediction_Zero_Rate"] == 1


def test_actual_zero_matches_only_neutral_prediction_and_is_outside_balanced_classes():
    result = classification_scores([0, 1, -1, 1, -1], [0, 0, 0, 2, -2])
    assert result["Direction_Accuracy"] == pytest.approx(3/5)
    assert result["Balanced_Accuracy"] == 1
    assert result["Actual_Zero_Rate"] == pytest.approx(3/5)
    assert result["Prediction_Zero_Rate"] == pytest.approx(1/5)
    assert result["Positive_Rate"] == pytest.approx(2/5)
    assert result["MAE"] == pytest.approx(4/5)
    assert result["RMSE"] == pytest.approx(np.sqrt(4/5))


def test_single_class_does_not_invent_balanced_accuracy():
    result = classification_scores([0, 1], [0, 1])
    assert result["Direction_Accuracy"] == 1
    assert np.isnan(result["Balanced_Accuracy"])
    assert np.isnan(result["True_Negative_Rate"])


@pytest.mark.parametrize("predicted,actual", [([np.nan], [1]), ([1], [np.inf]),
                                              ([1, 2], [1]), ([[1]], [[1]])])
def test_malformed_classification_inputs_fail_instead_of_broadcasting_or_dropping(predicted, actual):
    with pytest.raises(ValueError):
        classification_scores(predicted, actual)


@pytest.mark.parametrize("column,value", [("Predicted", np.nan), ("Actual_5d_Pct", np.nan),
                                         ("Predicted", np.inf), ("Actual_5d_Pct", -np.inf)])
def test_spearman_cannot_silently_drop_a_nonfinite_stock(column, value):
    frame = candidate_rows()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="must be finite"):
        extension.choose_candidate(frame, extension.load_config())


@pytest.mark.parametrize("change", ["missing_stock", "duplicate_stock", "wrong_member"])
def test_incomplete_cross_section_is_rejected(change):
    frame = candidate_rows()
    if change == "missing_stock":
        frame = frame.drop(index=0)
    elif change == "duplicate_stock":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:
        frame.loc[0, "Ticker"] = "unknown.TW"
    with pytest.raises(ValueError):
        extension.choose_candidate(frame, extension.load_config())


@pytest.mark.parametrize("column,value", [("Signal_Date", "2025-01-06"),
                                         ("Actual_5d_Pct", 99.0), ("Fold", 2),
                                         ("Execution_Date", "2025-01-07"), ("Label_End", "2025-02-01")])
def test_candidate_selection_requires_same_dates_labels_and_target_intervals(column, value):
    frame = candidate_rows()
    if column == "Signal_Date":
        frame.loc[frame.Model.eq("sentiment_risk_lstm") & frame.Signal_Date.eq("2025-01-02"), column] = value
    else:
        frame.loc[frame.Model.eq("sentiment_risk_lstm"), column] = value
    with pytest.raises(ValueError, match="differ"):
        extension.choose_candidate(frame, extension.load_config())


def test_test_nan_values_are_ignored_and_do_not_change_validation_selection():
    validation = candidate_rows()
    before = extension.choose_candidate(validation, extension.load_config())
    test = validation.assign(Split="test", Predicted=np.nan, Actual_5d_Pct=np.nan,
                             Signal_Date=None, Ticker=None, Model="unregistered_test_model")
    after = extension.choose_candidate(pd.concat([test, validation]), extension.load_config())
    assert before == after


def test_constant_cross_sections_contribute_zero_and_ties_use_average_ranks():
    frame = candidate_rows()
    one = frame[frame.Model.eq("price_lstm")].copy()
    one["Predicted"] = 5.0
    assert extension.validation_score(one) == 0
    one["Predicted"] = [0, 0, 1, 1, 2] * 2
    expected = one[one.Signal_Date.eq("2025-01-02")].Predicted.corr(
        one[one.Signal_Date.eq("2025-01-02")].Actual_5d_Pct, method="spearman")
    assert extension.validation_score(one) == pytest.approx(expected)


def test_candidate_row_order_does_not_change_score_or_pairing():
    frame = candidate_rows()
    expected = extension.choose_candidate(frame, extension.load_config())
    assert extension.choose_candidate(frame.sample(frac=1, random_state=7), extension.load_config()) == expected


@pytest.mark.parametrize("change", ["nan", "infinite", "shape", "wrong_stock_count", "empty"])
def test_daily_rank_metrics_reject_incomplete_or_malformed_arrays(change):
    predicted = np.arange(10, dtype=float).reshape(2, 5)
    actual = predicted.copy()
    if change == "nan":
        predicted[0, 0] = np.nan
    elif change == "infinite":
        actual[1, 1] = np.inf
    elif change == "shape":
        actual = actual[:1]
    elif change == "wrong_stock_count":
        predicted, actual = predicted[:, :4], actual[:, :4]
    else:
        predicted, actual = predicted[:0], actual[:0]
    with pytest.raises(ValueError):
        rank_metrics(predicted, actual)


def test_constant_daily_rank_is_zero_and_each_date_gets_equal_weight():
    predicted = np.array([[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]])
    actual = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    result = rank_metrics(predicted, actual)
    assert result["Cross_Section_Spearman"] == pytest.approx(0.5)
    assert result["Days"] == 2


@pytest.mark.parametrize("predicted,actual", [([np.nan], [1]), ([1], [np.inf]),
                                              ([1, 2], [1]), ([], []),
                                              ([[1, 2]], [1, 2])])
def test_pooled_skill_never_drops_nonfinite_rows_or_flattens_mismatched_shapes(predicted, actual):
    with pytest.raises(ValueError):
        flat_skill(predicted, actual)


def test_pooled_and_classification_direction_use_identical_exact_zero_definition():
    predicted, actual = [0, 1, -1, 1, -1], [0, 0, 0, 2, -2]
    assert flat_skill(predicted, actual)["Direction_Accuracy"] == classification_scores(predicted, actual)["Direction_Accuracy"]
    assert flat_skill(predicted, actual)["N"] == 5


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_horizon_selection_rejects_nonfinite_model_scores(value):
    with pytest.raises(ValueError, match="must be finite"):
        choose_model({"A": value, "B": 0.1, "C": 0.2, "D": 0.3})


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_candidate_margin_is_not_a_silent_nan_or_negative_setting(value):
    config = extension.load_config()
    config["minimum_validation_improvement"] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        extension.choose_candidate(candidate_rows(), config)
