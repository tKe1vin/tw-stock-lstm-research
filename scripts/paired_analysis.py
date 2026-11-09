"""Fixed paired contrasts and descriptive block-bootstrap uncertainty.

This analysis reuses observed history. Its confidence intervals describe sampling
uncertainty under a fixed circular-block scheme; they do not establish causality,
repair news provenance, or create an independent holdout. Regimes use the prior
execution day's policy risk state for the holdings that enter today's return.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRASTS = (
    ("news_without_risk", "dynamic_news_no_risk", "dynamic_no_news"),
    ("news_with_risk", "dynamic_news_risk555", "dynamic_no_news_risk"),
    ("risk_without_news", "dynamic_no_news_risk", "dynamic_no_news"),
    ("risk_with_news", "dynamic_news_risk555", "dynamic_news_no_risk"),
)


@dataclass(frozen=True)
class BootstrapConfig:
    block_length: int = 10
    draws: int = 1000
    seed: int = 42
    confidence: float = 0.95

    def __post_init__(self):
        if self.block_length not in (10, 20):
            raise ValueError("Use a fixed, declared 10- or 20-session block, never select the best on test")
        if self.draws < 100 or self.draws > 100000:
            raise ValueError("Bootstrap draws must be between 100 and 100000")
        if not 0 < self.confidence < 1:
            raise ValueError("Invalid confidence level")


def _frame(value):
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _bools(values):
    mapping = {True: True, False: False, 1: True, 0: False, "true": True, "false": False, "1": True, "0": False}
    parsed = values.map(lambda value: mapping.get(value.strip().lower() if isinstance(value, str) else value, np.nan))
    if parsed.isna().any():
        raise ValueError("Risk_Off must contain unambiguous booleans")
    return parsed.astype(bool)


def circular_block_indices(length, config=None):
    """Resample aligned paired dates jointly, retaining short-run serial dependence."""
    cfg = config or BootstrapConfig()
    if length < 2:
        raise ValueError("At least two post-entry return dates are required")
    blocks = int(np.ceil(length / cfg.block_length))
    rng = np.random.default_rng(cfg.seed)
    starts = rng.integers(0, length, size=(cfg.draws, blocks))
    indices = (starts[:, :, None] + np.arange(cfg.block_length)) % length
    return indices.reshape(cfg.draws, -1)[:, :length]


def _interval(values, indices, cfg, mask=None):
    sampled = values[indices]
    if mask is None:
        means = sampled.mean(axis=1)
    else:
        sampled_mask = mask[indices]
        count = sampled_mask.sum(axis=1)
        valid = count > 0
        means = np.divide((sampled * sampled_mask).sum(axis=1)[valid], count[valid])
    if not len(means):
        return np.nan, np.nan, 0
    tail = (1 - cfg.confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return float(lower), float(upper), len(means)


def analyze_paired_outputs(nav, weights, protocol=None, config=None):
    """Return contrasts/regime_effects, daily decomposition, entry costs and notes.

    Return differences are percentage points of *daily arithmetic net returns*,
    not differences in terminal ROI. The first entry day has no prior holding
    and is excluded from the bootstrap; its cost is separately disclosed.
    """
    cfg = config or BootstrapConfig()
    protocol = dict(protocol or {})
    book, policy = _frame(nav), _frame(weights)
    required = {"Arm", "Date", "NAV", "Transaction_Cost", "Risk_Off"}
    if not required.issubset(book.columns):
        raise ValueError("NAV needs Arm, Date, NAV, Transaction_Cost and Risk_Off")
    if not {"Arm", "Target_Date", "Risk_Off"}.issubset(policy.columns):
        raise ValueError("Weights need Arm, Target_Date and Risk_Off")
    book = book[book.Arm != "market_price_reference"].copy()
    book["Date"] = pd.to_datetime(book.Date).dt.tz_localize(None).dt.normalize()
    policy["Target_Date"] = pd.to_datetime(policy.Target_Date).dt.tz_localize(None).dt.normalize()
    book["Risk_Off"], policy["Risk_Off"] = _bools(book.Risk_Off), _bools(policy.Risk_Off)
    if book.empty or book.duplicated(["Arm", "Date"]).any():
        raise ValueError("Empty or duplicate NAV date/arm")
    if not np.isfinite(book[["NAV", "Transaction_Cost"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite NAV or costs")
    if book.NAV.le(0).any() or book.Transaction_Cost.lt(0).any():
        raise ValueError("NAV must be positive and transaction costs nonnegative")
    needed_arms = {arm for _, arm_a, arm_b in CONTRASTS for arm in (arm_a, arm_b)}
    if not needed_arms.issubset(set(book.Arm)):
        raise ValueError("All four factorial allocation arms are required")
    dates = pd.DatetimeIndex(sorted(book.Date.unique()))
    if len(dates) < 3:
        raise ValueError("At least three valuation dates are needed after separating entry")
    for arm, part in book.groupby("Arm"):
        if not pd.DatetimeIndex(part.Date.sort_values()).equals(dates):
            raise ValueError(f"NAV dates differ across arms: {arm}")
        matching = policy[policy.Arm == arm]
        if set(matching.Target_Date) != set(dates):
            raise ValueError(f"Policy dates differ from NAV: {arm}")
        risk_per_date = matching.groupby("Target_Date").Risk_Off
        if risk_per_date.nunique().ne(1).any():
            raise ValueError("Per-stock policy risk flags disagree")
        actual = part.set_index("Date").Risk_Off.reindex(dates)
        if not np.array_equal(actual.to_numpy(), risk_per_date.first().reindex(dates).to_numpy()):
            raise ValueError("NAV risk state differs from allocation policy")
    if book.groupby("Date").Risk_Off.nunique().ne(1).any():
        raise ValueError("Factorial arms must share common market-risk states")
    daily_rows, entries = [], []
    known_capital = protocol.get("rules", {}).get("initial_capital")
    for arm, part in book.groupby("Arm", sort=False):
        part = part.sort_values("Date").reset_index(drop=True)
        inferred = float(part.NAV.iloc[0] + part.Transaction_Cost.iloc[0])
        initial = float(known_capital) if known_capital is not None else inferred
        if not np.isfinite(initial) or initial <= 0 or not np.isclose(initial, inferred, atol=1e-5, rtol=1e-10):
            raise ValueError("Entry NAV and entry cost do not reconcile to common initial cash")
        entries.append({"Arm": arm, "Entry_Date": part.Date.iloc[0], "Initial_Capital": initial,
                        "Entry_NAV": float(part.NAV.iloc[0]), "Entry_Transaction_Cost": float(part.Transaction_Cost.iloc[0]),
                        "Entry_Return_Pct": float((part.NAV.iloc[0] / initial - 1) * 100),
                        "Included_In_Bootstrap": False})
        for i in range(1, len(part)):
            row, previous = part.iloc[i], part.iloc[i - 1]
            net = (row.NAV / previous.NAV - 1) * 100
            holding = ((row.NAV + row.Transaction_Cost) / previous.NAV - 1) * 100
            trading_cost = -row.Transaction_Cost / previous.NAV * 100
            daily_rows.append({"Arm": arm, "Date": row.Date, "Net_Return_Pct": float(net),
                               "Holding_Return_Pct": float(holding), "Trading_Cost_Return_Pct": float(trading_cost),
                               "Holding_Policy_Execution_Date": previous.Date,
                               "Holding_Policy_Signal_Date": previous.get("Signal_Date", pd.NaT),
                               "Holding_Policy_Risk_Off": bool(previous.Risk_Off),
                               "Regime": "risk_off" if previous.Risk_Off else "risk_on"})
    entry_costs = pd.DataFrame(entries)
    if not np.allclose(entry_costs.Initial_Capital, entry_costs.Initial_Capital.iloc[0], rtol=1e-10, atol=1e-5):
        raise ValueError("Arms do not share initial capital")
    daily = pd.DataFrame(daily_rows)
    net = daily.pivot(index="Date", columns="Arm", values="Net_Return_Pct").sort_index()
    holding = daily.pivot(index="Date", columns="Arm", values="Holding_Return_Pct").reindex(net.index)
    cost = daily.pivot(index="Date", columns="Arm", values="Trading_Cost_Return_Pct").reindex(net.index)
    group = daily[daily.Arm == next(iter(sorted(needed_arms)))].set_index("Date").Regime.reindex(net.index)
    indices = circular_block_indices(len(net), cfg)
    contrasts, regimes = [], []
    for name, arm_a, arm_b in CONTRASTS:
        delta = (net[arm_a] - net[arm_b]).to_numpy(dtype=float)
        lower, upper, draws = _interval(delta, indices, cfg)
        common = {"Contrast": name, "Arm_A": arm_a, "Arm_B": arm_b,
                  "Interpretation": "A minus B; positive means higher arithmetic daily net return",
                  "Result_Status": "exploratory_observed_history", "Block_Length": cfg.block_length,
                  "Requested_Draws": cfg.draws, "Seed": cfg.seed, "Confidence_Level": cfg.confidence}
        contrasts.append({**common, "N_Days": len(delta), "Mean_Daily_Difference_Pp": float(delta.mean()),
                          "CI_Lower_Pp": lower, "CI_Upper_Pp": upper, "Usable_Draws": draws,
                          "Mean_Daily_A_Pct": float(net[arm_a].mean()), "Mean_Daily_B_Pct": float(net[arm_b].mean())})
        for regime in ("risk_on", "risk_off"):
            mask = group.eq(regime).to_numpy()
            n = int(mask.sum())
            lower, upper, draws = _interval(delta, indices, cfg, mask)
            # Resample the complete paired date series, then apply sampled regime
            # labels. Do not concatenate disjoint risk-off days into fictitious blocks.
            regimes.append({**common, "Regime": regime, "N_Days": n,
                            "Mean_Daily_Difference_Pp": float(delta[mask].mean()) if n else np.nan,
                            "CI_Lower_Pp": lower, "CI_Upper_Pp": upper, "Usable_Draws": draws,
                            "Mean_Daily_A_Pct": float(net.loc[mask, arm_a].mean()) if n else np.nan,
                            "Mean_Daily_B_Pct": float(net.loc[mask, arm_b].mean()) if n else np.nan,
                            "Mean_Holding_Difference_Pp": float((holding.loc[mask, arm_a] - holding.loc[mask, arm_b]).mean()) if n else np.nan,
                            "Mean_Trading_Cost_Difference_Pp": float((cost.loc[mask, arm_a] - cost.loc[mask, arm_b]).mean()) if n else np.nan})
    notes = {
        "analysis_version": "paired-effects-block-bootstrap-v1", "status": "exploratory_observed_history",
        "causal_validated": False, "independent_holdout": False, "bootstrap": asdict(cfg),
        "bootstrap_method": "Circular blocks of fixed length; identical resampled paired dates for every contrast. Regime labels travel with those dates",
        "block_selection": "Default 10 sessions, declared in code; no selection among blocks, seeds or regimes based on test results",
        "contrast_unit": "Percentage points of mean arithmetic daily net return, not terminal ROI or annualized alpha",
        "regime_timing": "Today's return uses the prior execution day's common policy Risk_Off state; never today's newly available state",
        "regime_interpretation": "Regimes describe the policy state of entering holdings. Static references need not themselves trade on that state",
        "cost_timing": "Net returns include costs charged at today's close. Holding-return and trading-cost differences are also shown separately",
        "entry_policy": "First valuation/entry day excluded from contrasts and bootstrap, because there are no prior holdings. Its entry cost and return are separately reported",
        "terminal_policy": protocol.get("terminal_policy", "Read source ledger protocol for liquidation assumptions"),
        "first_analyzed_return_date": str(net.index[0].date()), "last_analyzed_return_date": str(net.index[-1].date()),
        "return_dates": len(net), "minimum_interval_days": cfg.block_length,
        "limitations": ["Observed and overlapping research history is reused; these intervals are not an independent holdout or proof of causal benefit",
                        "News provenance and model training integrity must be established separately",
                        "Fixed descriptive contrasts and two risk regimes are reported together; no multiple-testing-adjusted superiority claim is made",
                        "Short or rare regimes may have unstable intervals; N_Days and usable bootstrap draws are reported"],
    }
    return {"contrasts": pd.DataFrame(contrasts), "regime_effects": pd.DataFrame(regimes),
            "daily_returns": daily, "entry_costs": entry_costs, "notes": notes}


def analyze_paired_research(result, config=None):
    return analyze_paired_outputs(result["nav"], result["weights"], result.get("protocol"), config)


def write_analysis_outputs(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("contrasts", "regime_effects", "daily_returns", "entry_costs"):
        result[name].to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "notes.json").write_text(json.dumps(result["notes"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8")) if args.protocol else {}
    result = analyze_paired_outputs(args.nav, args.weights, protocol)
    write_analysis_outputs(result, args.output)
    print(result["contrasts"].to_string(index=False))
    print(result["regime_effects"].to_string(index=False))


if __name__ == "__main__":
    main()
