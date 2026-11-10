"""Read-only concentration diagnostics of retained net NAV paths.

The fixed diagnostics remove realized daily return factors, without refitting,
reselecting, or simulating replacement trades. They describe sensitivity of a
retained path; they are not confidence intervals or executable alternatives.
Default mode writes only to stdout. No raw provider prices are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLOCK_SESSIONS = 20
TAIL_DAYS = 5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def net_return_factors(nav, initial_capital):
    """Include initial capital -> first reported NAV, preserving entry costs."""
    if not {"Date", "NAV"}.issubset(nav.columns):
        raise ValueError("NAV data requires Date and NAV")
    dates = pd.DatetimeIndex(pd.to_datetime(nav.Date, errors="raise"))
    values = pd.to_numeric(nav.NAV, errors="raise").to_numpy(float)
    if (len(dates) == 0 or dates.hasnans or dates.has_duplicates or dates.tz is not None
            or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
        raise ValueError("NAV dates must be unique increasing local session dates")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be finite and positive")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Every NAV observation must be finite and positive")
    factors = values / np.r_[float(initial_capital), values[:-1]]
    return dates, factors


def summarize_factors(dates, factors):
    """Fixed best/worst five sessions and chronological 20-session deletions."""
    dates = pd.DatetimeIndex(dates)
    factors = np.asarray(factors, dtype=float)
    if (factors.ndim != 1 or len(factors) != len(dates) or len(factors) == 0
            or not np.isfinite(factors).all() or np.any(factors <= 0)):
        raise ValueError("Factors must be a complete finite positive dated vector")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Factor dates must be complete, unique and increasing")
    logs = np.log(factors)
    total = float(logs.sum())
    if not np.isfinite(total) or abs(total) > 500:
        raise ValueError("Return path exceeds the diagnostic numeric range")
    # Stable order makes tied returns deterministic by their chronological date.
    order = np.argsort(logs, kind="stable")
    count = min(TAIL_DAYS, len(logs))
    best, worst = order[-count:][::-1], order[:count]
    blocks = []
    for start in range(0, len(logs), BLOCK_SESSIONS):
        stop = min(start + BLOCK_SESSIONS, len(logs))
        contribution = float(logs[start:stop].sum())
        blocks.append({"first_date": str(dates[start].date()), "last_date": str(dates[stop - 1].date()),
                       "sessions": stop - start, "log_wealth_contribution": contribution,
                       "block_return_pct": float(np.expm1(contribution) * 100),
                       "wealth_return_without_block_pct": float(np.expm1(total - contribution) * 100)})
    positive_blocks = np.maximum([part["log_wealth_contribution"] for part in blocks], 0)
    positive_sum = float(positive_blocks.sum())
    positive_days = float(np.maximum(logs, 0).sum())
    def tails(indices):
        return [{"date": str(dates[i].date()), "return_pct": float((factors[i] - 1) * 100),
                 "log_wealth_contribution": float(logs[i])} for i in indices]
    leave_out = [part["wealth_return_without_block_pct"] for part in blocks]
    return {"sessions": len(dates), "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
            "wealth_return_pct": float(np.expm1(total) * 100),
            "top_five_best_sessions": tails(best), "top_five_worst_sessions": tails(worst),
            "wealth_return_without_best_five_pct": float(np.expm1(total - logs[best].sum()) * 100),
            "wealth_return_without_worst_five_pct": float(np.expm1(total - logs[worst].sum()) * 100),
            "best_five_share_of_positive_log_returns": float(np.maximum(logs[best], 0).sum() / positive_days) if positive_days > 0 else None,
            "fixed_block_sessions": BLOCK_SESSIONS, "blocks": blocks,
            "leave_one_block_out_return_range_pct": [float(min(leave_out)), float(max(leave_out))],
            "largest_positive_block_share_of_positive_block_log_returns": float(positive_blocks.max() / positive_sum) if positive_sum > 0 else None,
            "complete_blocks": int(len(logs) // BLOCK_SESSIONS),
            "last_partial_block_sessions": int(len(logs) % BLOCK_SESSIONS)}


def path_diagnostic(nav, initial_capital):
    dates, factors = net_return_factors(nav, initial_capital)
    return summarize_factors(dates, factors)


def paired_diagnostic(strategy, control, initial_capital):
    dates, strategy_factors = net_return_factors(strategy, initial_capital)
    other_dates, control_factors = net_return_factors(control, initial_capital)
    if not dates.equals(other_dates):
        raise ValueError("Relative concentration requires identical complete dates")
    result = summarize_factors(dates, strategy_factors / control_factors)
    result["return_interpretation"] = "relative wealth: strategy wealth / control wealth - 1"
    return result


def _load(path, nav_column="NAV"):
    frame = pd.read_csv(path)
    return frame[["Date", nav_column]].rename(columns={nav_column: "NAV"})


def audit(v3_run_dir=None):
    sources, studies = {}, {}
    paired_path = ROOT / "reports/research_revision/2026-11-06/paired/nav.csv"
    paired = pd.read_csv(paired_path)
    sources[paired_path.relative_to(ROOT).as_posix()] = digest(paired_path)
    arms = {name: part[["Date", "NAV"]].reset_index(drop=True)
            for name, part in paired.groupby("Arm", sort=True)}
    studies["shared_core_paired"] = {
        "initial_capital": 100_000,
        "paths": {name: path_diagnostic(frame, 100_000) for name, frame in arms.items()},
        "relative_pairs": {"news_risk_vs_same_risk_no_news": paired_diagnostic(
            arms["dynamic_news_risk555"], arms["dynamic_no_news_risk"], 100_000),
            "news_no_risk_vs_no_news_no_risk": paired_diagnostic(
                arms["dynamic_news_no_risk"], arms["dynamic_no_news"], 100_000)}}
    extension = ROOT / "reports/research_extension/2026-11-06"
    arms = {}
    for name in ("price_lstm", "sentiment_risk_lstm", "matched_risk_equal", "full_exposure_buy_hold"):
        path = extension / ("nav_" + name + ".csv")
        arms[name] = _load(path, "NAV_Reported")
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    studies["retained_lstm_extension"] = {
        "initial_capital": 100_000,
        "paths": {name: path_diagnostic(frame, 100_000) for name, frame in arms.items()},
        "relative_pairs": {"news_vs_price": paired_diagnostic(arms["sentiment_risk_lstm"], arms["price_lstm"], 100_000),
                           "news_vs_same_risk_equal": paired_diagnostic(arms["sentiment_risk_lstm"], arms["matched_risk_equal"], 100_000)}}
    if v3_run_dir is not None:
        v3 = Path(v3_run_dir)
        arms = {}
        for name in ("B0_benchmark", "B1_equal", "ridge_price", "ridge_news"):
            path = v3 / ("holdout_" + name) / "nav.csv"
            arms[name] = _load(path)
            # The public record identifies an authorized local derived NAV by
            # alias+hash, never by a user's absolute machine path.
            sources["local_v3_run/holdout_" + name + "/nav.csv"] = digest(path)
        studies["v3_exploratory_holdout"] = {
            "initial_capital": 1_000_000, "selected": "B0_benchmark", "news_adopted": False,
            "paths": {name: path_diagnostic(frame, 1_000_000) for name, frame in arms.items()},
            "relative_pairs": {"news_vs_0050": paired_diagnostic(arms["ridge_news"], arms["B0_benchmark"], 1_000_000),
                               "news_vs_price": paired_diagnostic(arms["ridge_news"], arms["ridge_price"], 1_000_000)}}
    return {"schema": "retained-nav-concentration-v1", "reviewed_on": "2026-11-10",
            "scope": "available retained net NAV paths; not missing original v28 exports",
            "code_sha256": {"scripts/concentration_audit.py": digest(Path(__file__))},
            "method_fixed": {"tail_days": TAIL_DAYS, "chronological_block_sessions": BLOCK_SESSIONS,
                             "last_short_block_included": True, "entry_cost_included": True},
            "input_sha256": sources, "studies": studies,
            "limitations": ["Descriptive realized-return deletion, not refitted or re-traded strategies.",
                            "No model or parameter is selected using these diagnostics.",
                            "No confidence interval, independent alpha or causal attribution is claimed.",
                            "Missing original v28 prediction/portfolio exports remain unassessed.",
                            "New V3 provider raw prices are neither read by this diagnostic nor redistributed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit(args.v3_run_dir)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        # New audit evidence only; never replace a published historical report.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
