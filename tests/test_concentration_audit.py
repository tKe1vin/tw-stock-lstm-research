"""Net-return concentration is an exact arithmetic diagnostic, not selection."""
import numpy as np
import pandas as pd
import pytest

from scripts import concentration_audit as concentration


def path(returns, initial=100.):
    return pd.DataFrame({"Date": pd.bdate_range("2026-01-01", periods=len(returns)),
                         "NAV": initial * np.cumprod(1 + np.asarray(returns))})


def test_entry_cost_and_exact_realized_wealth_are_preserved():
    nav = path([-.01, .10, -.05, .20])
    dates, factors = concentration.net_return_factors(nav, 100.)
    np.testing.assert_allclose(factors, [.99, 1.10, .95, 1.20])
    result = concentration.path_diagnostic(nav, 100.)
    assert result["wealth_return_pct"] == pytest.approx((.99 * 1.10 * .95 * 1.20 - 1) * 100)
    assert result["sessions"] == 4
    assert result["last_partial_block_sessions"] == 4


def test_single_outlier_reveals_best_day_and_block_concentration():
    returns = np.zeros(45)
    returns[22] = .5
    result = concentration.path_diagnostic(path(returns), 100.)
    assert result["wealth_return_pct"] == pytest.approx(50.)
    assert result["wealth_return_without_best_five_pct"] == pytest.approx(0.)
    assert result["best_five_share_of_positive_log_returns"] == pytest.approx(1.)
    assert result["largest_positive_block_share_of_positive_block_log_returns"] == pytest.approx(1.)
    assert result["leave_one_block_out_return_range_pct"] == pytest.approx([0., 50.])
    assert [block["sessions"] for block in result["blocks"]] == [20, 20, 5]


def test_relative_wealth_uses_factor_ratios_not_subtracted_rois():
    strategy, control = path([.10, .20]), path([.05, -.10])
    result = concentration.paired_diagnostic(strategy, control, 100.)
    assert result["wealth_return_pct"] == pytest.approx(((1.1 * 1.2) / (1.05 * .9) - 1) * 100)
    assert result["wealth_return_without_worst_five_pct"] == pytest.approx(0.)


def test_relative_dates_must_match_without_dropping_missing_observations():
    strategy, control = path([.1, -.1, .2]), path([.1, .1, .1])
    with pytest.raises(ValueError, match="identical complete dates"):
        concentration.paired_diagnostic(strategy, control.iloc[:-1], 100.)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0., -1.])
def test_nonfinite_or_nonpositive_nav_is_rejected(bad):
    nav = path([0., .01, .02])
    nav.loc[1, "NAV"] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        concentration.path_diagnostic(nav, 100.)


def test_duplicate_or_unsorted_dates_are_rejected():
    nav = path([0., .01, .02])
    with pytest.raises(ValueError, match="unique increasing"):
        concentration.path_diagnostic(nav.iloc[::-1], 100.)
    nav.loc[1, "Date"] = nav.loc[0, "Date"]
    with pytest.raises(ValueError, match="unique increasing"):
        concentration.path_diagnostic(nav, 100.)


def test_all_negative_path_has_no_false_positive_concentration_share():
    result = concentration.path_diagnostic(path([-.01] * 22), 100.)
    assert result["best_five_share_of_positive_log_returns"] is None
    assert result["largest_positive_block_share_of_positive_block_log_returns"] is None


def test_summarize_rejects_mismatched_or_invalid_factor_vectors():
    dates = pd.bdate_range("2026-01-01", periods=2)
    for factors in ([1.], [1., np.nan], [1., 0.], [[1., 1.]]):
        with pytest.raises(ValueError, match="complete finite positive dated vector"):
            concentration.summarize_factors(dates, factors)
