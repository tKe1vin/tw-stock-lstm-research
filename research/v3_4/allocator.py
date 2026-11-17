"""Estimate turnover using signal-time holdings and quotes, never future fills."""
from itertools import combinations
import math


def planned_shares(weights, nav, prices):
    return {asset: int(math.floor(nav * weight / prices[asset])) if weight > 0 else 0
            for asset, weight in weights.items()}


def estimate_trade_cost(current, target, prices, settings):
    """Same fee/tax/slippage conventions as the ledger, in currency units.

    This is a decision-time estimate. Future close, volume and affordability
    still determine real fills in the unchanged accounting engine.
    """
    total = 0.0
    for asset, desired in target.items():
        delta = desired - float(current.get(asset, 0))
        quantity = int(math.floor(abs(delta) + 1e-9))
        if not quantity:
            continue
        sell = delta < 0
        raw = float(prices[asset])
        fill = raw * (1 - settings.slip if sell else 1 + settings.slip)
        gross = quantity * fill
        fee = max(settings.minimum_fee, gross * settings.fee)
        tax = gross * (settings.etf_tax if asset == settings.benchmark else settings.stock_tax) if sell else 0
        total += fee + tax + quantity * abs(fill - raw)
    return total


def incremental_weights(info, previous, nav, settings, prices, current):
    assets = [settings.benchmark, *settings.stocks]
    base = {asset: float(asset == settings.benchmark) for asset in assets}
    if any(asset not in prices or not math.isfinite(float(prices[asset])) or prices[asset] <= 0 for asset in assets):
        return base, 0., "invalid_signal_quote_fallback_to_0050"
    target_base = planned_shares(base, nav, prices)
    base_cost = estimate_trade_cost(current, target_base, prices, settings)
    positive = sorted(asset for asset in settings.stocks if float(info.loc[asset, "Predicted"]) > 0)
    best, best_score, best_threshold = base, 0., 0.
    for size in range(1, min(settings.top_n, len(positive)) + 1):
        for chosen in combinations(positive, size):
            weights = dict.fromkeys(assets, 0.)
            for asset in chosen:
                wanted = min(settings.stock_cap, settings.satellite_cap / settings.top_n)
                old = min(float(previous.get(asset, 0)), settings.stock_cap)
                weights[asset] = old if old > 0 and abs(wanted - old) < settings.band else wanted
            weights[settings.benchmark] = 1 - sum(weights.values())
            target = planned_shares(weights, nav, prices)
            entry_cost = estimate_trade_cost(current, target, prices, settings)
            exit_reserve = estimate_trade_cost(target, target_base, prices, settings)
            # Compare against moving the same current holdings back to0050;
            # reserve the satellite exit at today's known quotes conservatively.
            incremental_cost = max(0., entry_cost - base_cost + exit_reserve)
            expected_gain = sum(nav * weights[asset] * float(info.loc[asset, "Predicted"]) / 100 for asset in chosen)
            score = expected_gain - settings.edge_multiplier * incremental_cost
            if score > best_score + 1e-9:
                best, best_score = weights, score
                satellite = sum(weights[asset] for asset in settings.stocks)
                best_threshold = settings.edge_multiplier * incremental_cost / (nav * satellite) * 100 if satellite else 0.
    if best[settings.benchmark] < settings.core_floor - 1e-10:
        raise AssertionError("Incremental allocator violates core floor")
    return best, best_threshold, "incremental_turnover_gate_passed" if best_score > 0 else "no_incremental_edge_fallback_to_0050"
