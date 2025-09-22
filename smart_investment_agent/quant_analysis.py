"""Quantitative routines for the smart investment agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .data_models import (
    HedgeFundPosition,
    HedgeSentiment,
    InvestmentSignal,
    MarketSnapshot,
    PortfolioRecommendation,
    TechnicalIndicators,
)


def aggregate_hedge_sentiment(
    positions: Iterable[HedgeFundPosition],
    min_weight: float = 0.0,
) -> Dict[str, HedgeSentiment]:
    """Aggregate positions into a hedge sentiment score per ticker."""

    bucket: Dict[str, List[HedgeFundPosition]] = {}
    for pos in positions:
        if pos.portfolio_weight < min_weight:
            continue
        bucket.setdefault(pos.ticker, []).append(pos)

    sentiments: Dict[str, HedgeSentiment] = {}
    for ticker, rows in bucket.items():
        net_change = sum(r.market_value * r.qoq_change_pct for r in rows)
        avg_position = np.mean([r.market_value for r in rows]) if rows else 0.0
        score = 0.0
        if avg_position:
            increases = [max(r.qoq_change_pct, 0) for r in rows]
            decreases = [abs(min(r.qoq_change_pct, 0)) for r in rows]
            score = (np.mean(increases) * avg_position) - (np.mean(decreases) * avg_position)
            score /= max(avg_position, 1.0)
        sentiments[ticker] = HedgeSentiment(
            ticker=ticker,
            score=float(score),
            total_net_change=float(net_change),
            contributors=rows,
        )
    return sentiments


def compute_technicals(price_series: pd.Series) -> TechnicalIndicators:
    """Calculate SMA, RSI, and MACD indicators for a price series."""

    closes = price_series.dropna()
    sma_50 = closes.rolling(window=50).mean().iloc[-1]
    sma_200 = closes.rolling(window=200).mean().iloc[-1]

    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return TechnicalIndicators(
        sma_50=float(sma_50),
        sma_200=float(sma_200),
        rsi_14=float(rsi),
        macd=float(macd.iloc[-1]),
        macd_signal=float(signal.iloc[-1]),
    )


@dataclass
class ScenarioResult:
    expected_return: float
    volatility: float


def monte_carlo_projection(
    price_series: pd.Series,
    horizon_days: int = 252,
    simulations: int = 5_000,
) -> ScenarioResult:
    """Project price evolution using a geometric Brownian motion approximation."""

    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    rng = np.random.default_rng(7)
    simulated_paths = rng.normal(mu, sigma, size=(simulations, horizon_days))
    simulated_returns = simulated_paths.sum(axis=1)

    expected_return = float(simulated_returns.mean())
    volatility = float(simulated_returns.std())
    return ScenarioResult(expected_return=expected_return, volatility=volatility)


def heuristic_signal(
    ticker: str,
    sentiment: HedgeSentiment,
    technicals: TechnicalIndicators,
    snapshot: MarketSnapshot,
    projection: ScenarioResult,
) -> InvestmentSignal:
    """Generate a human-readable recommendation based on heuristics."""

    reasons = []
    action = "HOLD"

    if sentiment.score > 0.5:
        reasons.append("Hedge funds accumulating positions")
    elif sentiment.score < -0.5:
        reasons.append("Smart money reducing exposure")

    if technicals.sma_50 > technicals.sma_200:
        reasons.append("Positive trend (50DMA above 200DMA)")
    else:
        reasons.append("Long-term trend deterioration")

    if technicals.rsi_14 < 40:
        reasons.append("RSI indicates oversold levels")
    elif technicals.rsi_14 > 70:
        reasons.append("RSI indicates overbought conditions")

    valuation_gap = None
    if snapshot.pe_ratio is not None and snapshot.sector_pe is not None:
        valuation_gap = snapshot.sector_pe - snapshot.pe_ratio
        if valuation_gap > 3:
            reasons.append("Valuation below sector peers")
        elif valuation_gap < -3:
            reasons.append("Premium valuation vs sector")

    expected_price_move = np.exp(projection.expected_return) - 1
    if expected_price_move > 0.1:
        reasons.append("Monte Carlo suggests double-digit upside")
    elif expected_price_move < -0.1:
        reasons.append("Monte Carlo points to notable downside")

    # Determine action
    bullish = sentiment.score > 0.5 and technicals.rsi_14 < 60 and (valuation_gap or 0) >= 0
    bearish = sentiment.score < -0.5 or technicals.rsi_14 > 70 or (valuation_gap or 0) < -5

    if bullish:
        action = "BUY"
    elif bearish:
        action = "SELL"

    timing_hint = None
    if action == "BUY" and technicals.rsi_14 > 50:
        timing_hint = "Consider scaling in after a pullback toward RSI 45"
    elif action == "SELL" and technicals.rsi_14 < 50:
        timing_hint = "Watch for relief rallies before trimming"

    rationale = "; ".join(reasons) or "Insufficient data"
    return InvestmentSignal(
        ticker=ticker,
        action=action,
        rationale=rationale,
        timing_hint=timing_hint,
    )


def optimize_portfolio(
    tickers: List[str],
    expected_returns: Dict[str, float],
    covariance: pd.DataFrame,
    target_sector_weights: Dict[str, float] | None = None,
    max_allocation: float = 0.3,
) -> PortfolioRecommendation:
    """Optimize a toy portfolio using quadratic programming or random search."""

    try:
        import pulp  # type: ignore
    except ModuleNotFoundError:
        return _random_portfolio(tickers, expected_returns, covariance, max_allocation)

    problem = pulp.LpProblem("PortfolioOptimization", pulp.LpMaximize)
    weights = {t: pulp.LpVariable(f"w_{t}", lowBound=0, upBound=max_allocation) for t in tickers}
    risk_free = 0.04

    # Objective: approximate Sharpe ratio numerator minus penalty on variance
    mu = [expected_returns.get(t, 0.08) - risk_free for t in tickers]
    cov_matrix = covariance.loc[tickers, tickers].values
    variance = pulp.lpSum(
        cov_matrix[i, j] * weights[tickers[i]] * weights[tickers[j]]
        for i in range(len(tickers))
        for j in range(len(tickers))
    )
    objective = pulp.lpSum(mu[i] * weights[ticker] for i, ticker in enumerate(tickers)) - 5 * variance
    problem += objective

    # Constraints
    problem += pulp.lpSum(weights.values()) == 1.0

    if target_sector_weights:
        for sector, target in target_sector_weights.items():
            members = [t for t in tickers if t.startswith(sector)]
            if not members:
                continue
            problem += pulp.lpSum(weights[m] for m in members) >= target * 0.5
            problem += pulp.lpSum(weights[m] for m in members) <= target * 1.5

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    allocation = {t: float(weights[t].value() or 0) for t in tickers}
    expected_ret = float(sum(allocation[t] * expected_returns.get(t, 0.08) for t in tickers))
    vol = float(np.sqrt(np.dot(np.array(list(allocation.values())), np.dot(covariance.loc[tickers, tickers], np.array(list(allocation.values()))))))
    sharpe = (expected_ret - risk_free) / vol if vol else 0.0
    return PortfolioRecommendation(allocation, expected_ret, vol, sharpe)


def _random_portfolio(
    tickers: List[str],
    expected_returns: Dict[str, float],
    covariance: pd.DataFrame,
    max_allocation: float,
    simulations: int = 5_000,
) -> PortfolioRecommendation:
    rng = np.random.default_rng(99)
    best_weights: Dict[str, float] | None = None
    best_sharpe = -np.inf
    risk_free = 0.04
    cov = covariance.loc[tickers, tickers].values

    for _ in range(simulations):
        weights = rng.random(len(tickers))
        weights /= weights.sum()
        weights = np.minimum(weights, max_allocation)
        weights /= weights.sum()

        w_vec = np.array(weights)
        mu = np.array([expected_returns.get(t, 0.08) for t in tickers])
        expected_ret = float(np.dot(w_vec, mu))
        variance = float(np.dot(w_vec, np.dot(cov, w_vec)))
        vol = np.sqrt(max(variance, 1e-8))
        sharpe = (expected_ret - risk_free) / vol

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = {t: float(w) for t, w in zip(tickers, w_vec)}
            best_ret = expected_ret
            best_vol = vol

    if best_weights is None:
        best_weights = {t: 1 / len(tickers) for t in tickers}
        best_ret = float(np.mean(list(expected_returns.values())))
        best_vol = float(np.sqrt(np.mean(np.diag(cov))))
        best_sharpe = (best_ret - risk_free) / best_vol if best_vol else 0.0

    return PortfolioRecommendation(best_weights, best_ret, best_vol, best_sharpe)

