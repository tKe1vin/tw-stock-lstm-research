"""Render the retained paired ledger and fixed descriptive contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def plot_report(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    nav = pd.read_csv(folder / "nav.csv", parse_dates=["Date"])
    contrasts = pd.read_csv(folder / "analysis/contrasts.csv")
    labels = {"static_equal444": "444 fixed shares",
              "dynamic_news_risk555": "555: shared core + news + risk",
              "risk_matched_equal_reference": "Same-exposure equal allocation",
              "market_price_reference": "TWII price index (gross reference)"}
    colors = ["#2366a8", "#c54d42", "#2d8d76", "#8893a3"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.45, 1]})
        for (arm, label), color in zip(labels.items(), colors):
            rows = nav[nav.Arm == arm]
            axes[0].plot(rows.Date, rows.NAV / 1000, color=color, label=label,
                         linestyle="--" if arm == "market_price_reference" else "-", linewidth=1.8)
        axes[0].set(title="Shared-core portfolio replay", ylabel="NAV (NT$ thousands)")
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
        axes[0].grid(alpha=.15)
        effects = contrasts.iloc[::-1].reset_index(drop=True)
        pretty = {"news_without_risk": "News effect: no risk filter",
                  "news_with_risk": "News effect: with risk filter",
                  "risk_without_news": "Risk filter: no news",
                  "risk_with_news": "Risk filter: with news"}
        for i, row in effects.iterrows():
            axes[1].plot([row.CI_Lower_Pp, row.CI_Upper_Pp], [i, i], color="#60758d", linewidth=3)
            axes[1].plot(row.Mean_Daily_Difference_Pp, i, "o", color="#173c62")
        axes[1].axvline(0, color="#aaa", linewidth=1)
        axes[1].set_yticks(range(len(effects)), [pretty[name] for name in effects.Contrast], fontsize=8)
        axes[1].set(title="Fixed paired contrasts", xlabel="Mean daily net return difference (pp)")
        axes[1].grid(axis="x", alpha=.15)
        fig.suptitle("Exploratory research • 2026-03-30 to 2026-11-06", fontsize=14, fontweight="bold")
        fig.text(.015, .015, "Next-close fills; stock fees/tax included. TWII is a gross price reference.\n"
                 "Descriptive 95% circular-block intervals: 10 sessions, 1,000 draws, seed 42. Historical news timing is unverified.", fontsize=8, color="#555")
        fig.tight_layout(rect=[0, .095, 1, .94])
        fig.savefig(folder / "performance.png", dpi=180, facecolor="white")
        fig.savefig(folder / "performance.svg", facecolor="white")
        plt.close(fig)
    report = folder / "comparison.md"
    if report.is_file() and "(performance.png)" not in report.read_text(encoding="utf-8"):
        with report.open("a", encoding="utf-8") as handle:
            handle.write("\n![配對帳本淨值與固定效果區間](performance.png)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    plot_report(parser.parse_args().folder)


if __name__ == "__main__":
    main()
