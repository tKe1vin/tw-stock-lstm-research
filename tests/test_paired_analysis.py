"""Known paired effects, reproducibility and holding-period risk attribution."""
import numpy as np
import pandas as pd
import pytest

from scripts.paired_analysis import (
    BootstrapConfig, CONTRASTS, analyze_paired_outputs, circular_block_indices,
)


def books(days=30):
    dates = pd.bdate_range("2026-01-02", periods=days)
    base = {
        "dynamic_no_news": .002,
        "dynamic_news_no_risk": .003,
        "dynamic_no_news_risk": .001,
        "dynamic_news_risk555": .0025,
    }
    nav, weights = [], []
    for arm, daily_return in base.items():
        value = 990.0
        for i, date in enumerate(dates):
            if i:
                value *= 1 + daily_return
            nav.append({"Arm": arm, "Date": date, "NAV": value,
                        "Transaction_Cost": 10.0 if i == 0 else 0.0,
                        "Risk_Off": bool(i % 3 == 1),
                        "Signal_Date": date - pd.offsets.BDay(1)})
            weights.append({"Arm": arm, "Target_Date": date, "Ticker": "stock",
                            "Risk_Off": bool(i % 3 == 1)})
    return pd.DataFrame(nav), pd.DataFrame(weights), {"rules": {"initial_capital": 1000.0}}


def test_known_daily_effects_and_entry_cost_separation():
    nav, weights, protocol = books()
    result = analyze_paired_outputs(nav, weights, protocol)
    effects = result["contrasts"].set_index("Contrast")
    assert effects.loc["news_without_risk", "Mean_Daily_Difference_Pp"] == pytest.approx(.1)
    assert effects.loc["news_with_risk", "Mean_Daily_Difference_Pp"] == pytest.approx(.15)
    assert effects.loc["risk_without_news", "Mean_Daily_Difference_Pp"] == pytest.approx(-.1)
    assert effects.loc["risk_with_news", "Mean_Daily_Difference_Pp"] == pytest.approx(-.05)
    assert effects.N_Days.eq(29).all()
    assert result["entry_costs"].Entry_Return_Pct.to_numpy() == pytest.approx([-1] * 4)
    assert not result["entry_costs"].Included_In_Bootstrap.any()
    np.testing.assert_allclose(effects.CI_Lower_Pp, effects.Mean_Daily_Difference_Pp, atol=1e-10)
    np.testing.assert_allclose(effects.CI_Upper_Pp, effects.Mean_Daily_Difference_Pp, atol=1e-10)
    assert result["notes"]["independent_holdout"] is False


def test_regime_uses_prior_execution_policy_and_ignores_final_new_risk_state():
    nav, weights, protocol = books()
    result = analyze_paired_outputs(nav, weights, protocol)
    daily = result["daily_returns"]
    arm = "dynamic_no_news"
    source = nav[nav.Arm == arm].sort_values("Date")
    reported = daily[daily.Arm == arm].sort_values("Date")
    assert reported.Holding_Policy_Risk_Off.tolist() == source.Risk_Off.iloc[:-1].tolist()
    assert reported.Holding_Policy_Execution_Date.tolist() == source.Date.iloc[:-1].tolist()
    assert reported.Regime.iloc[0] == "risk_on"
    assert source.Risk_Off.iloc[1]  # using today's flag would incorrectly label this first return risk-off
    last = nav.Date.max()
    changed_nav, changed_weights = nav.copy(), weights.copy()
    changed_nav.loc[changed_nav.Date == last, "Risk_Off"] = ~changed_nav.loc[changed_nav.Date == last, "Risk_Off"]
    changed_weights.loc[changed_weights.Target_Date == last, "Risk_Off"] = ~changed_weights.loc[changed_weights.Target_Date == last, "Risk_Off"]
    changed = analyze_paired_outputs(changed_nav, changed_weights, protocol)
    pd.testing.assert_frame_equal(result["regime_effects"], changed["regime_effects"])


def test_circular_blocks_are_reproducible_joint_time_blocks():
    cfg = BootstrapConfig()
    first, second = circular_block_indices(37, cfg), circular_block_indices(37, cfg)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1000, 37)
    # Every complete 10-day sampled block preserves adjacent dates modulo sample length.
    for start in (0, 10, 20):
        np.testing.assert_array_equal(np.diff(first[:, start:start + 10], axis=1) % 37, np.ones((1000, 9)))
    with pytest.raises(ValueError, match="10- or 20-session"):
        BootstrapConfig(block_length=7)


def test_net_return_decomposes_into_holding_and_today_trading_cost():
    nav, weights, protocol = books()
    day = nav.Date.max()
    nav.loc[nav.Date == day, "NAV"] -= 5
    nav.loc[nav.Date == day, "Transaction_Cost"] = 5
    result = analyze_paired_outputs(nav, weights, protocol)
    daily = result["daily_returns"]
    np.testing.assert_allclose(daily.Net_Return_Pct, daily.Holding_Return_Pct + daily.Trading_Cost_Return_Pct, atol=1e-10)
    assert daily.loc[daily.Date == day, "Trading_Cost_Return_Pct"].lt(0).all()


def test_different_dates_and_policy_mismatch_are_rejected():
    nav, weights, protocol = books()
    with pytest.raises(ValueError, match="dates differ"):
        analyze_paired_outputs(nav.iloc[1:], weights, protocol)
    changed = weights.copy()
    changed.loc[0, "Risk_Off"] = not changed.loc[0, "Risk_Off"]
    with pytest.raises(ValueError, match="differs from allocation"):
        analyze_paired_outputs(nav, changed, protocol)


def test_absent_regime_is_reported_without_fabricated_interval():
    nav, weights, protocol = books()
    nav["Risk_Off"], weights["Risk_Off"] = False, False
    result = analyze_paired_outputs(nav, weights, protocol)
    absent = result["regime_effects"].query("Regime == 'risk_off'")
    assert len(absent) == len(CONTRASTS)
    assert absent.N_Days.eq(0).all() and absent.Usable_Draws.eq(0).all()
    assert absent.CI_Lower_Pp.isna().all() and absent.CI_Upper_Pp.isna().all()
