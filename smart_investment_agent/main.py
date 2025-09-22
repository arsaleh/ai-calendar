"""Entry point for running the smart investment analysis agent."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data_collection import (
    load_hedge_fund_positions,
    load_market_snapshots,
    load_price_history,
)
from .data_models import (
    AnalysisReport,
    HedgeSentiment,
    InvestmentSignal,
    MarketSnapshot,
    TechnicalIndicators,
)
from .quant_analysis import (
    ScenarioResult,
    aggregate_hedge_sentiment,
    compute_technicals,
    heuristic_signal,
    monte_carlo_projection,
    optimize_portfolio,
)
from .reporting import assemble_report, build_data_table, compose_summary

DEFAULT_TICKERS = ["NVDA", "TSLA", "MSFT", "AAPL"]


def run_analysis(
    query: str,
    tickers: Optional[Iterable[str]] = None,
    *,
    hedge_csv: Optional[Path] = None,
    market_csv: Optional[Path] = None,
    price_dir: Optional[Path] = None,
) -> AnalysisReport:
    """Execute the end-to-end pipeline described in the system prompt."""

    tickers = [t.upper() for t in (tickers or DEFAULT_TICKERS)]
    if not tickers:
        raise ValueError("At least one ticker is required for analysis.")

    positions = load_hedge_fund_positions(tickers, hedge_csv)
    hedge_sentiment = aggregate_hedge_sentiment(positions)

    snapshots = load_market_snapshots(tickers, market_csv)

    technicals_map = {}
    projections = {}
    log_return_map: Dict[str, pd.Series] = {}

    signals = []
    for ticker in tickers:
        price_path = price_dir / f"{ticker}.csv" if price_dir else None
        price_series = load_price_history(ticker, price_path)
        technicals = compute_technicals(price_series)
        projection = monte_carlo_projection(price_series)

        technicals_map[ticker] = technicals
        projections[ticker] = projection
        log_return_map[ticker] = np.log(price_series / price_series.shift(1)).dropna()

        sentiment = hedge_sentiment.get(ticker)
        if sentiment is None:
            sentiment = hedge_sentiment[ticker] = HedgeSentiment(
                ticker=ticker,
                score=0.0,
                total_net_change=0.0,
                contributors=[],
            )

        snapshot = snapshots.get(ticker)
        if snapshot is None:
            snapshot = load_market_snapshots([ticker])[ticker]
            snapshots[ticker] = snapshot

        signal = heuristic_signal(
            ticker=ticker,
            sentiment=sentiment,
            technicals=technicals,
            snapshot=snapshot,
            projection=projection,
        )
        signals.append(signal)

    # Prepare portfolio inputs
    aligned = pd.DataFrame(log_return_map).dropna()
    if aligned.empty:
        aligned = pd.DataFrame({t: pd.Series([0.0]) for t in tickers})
    covariance = aligned.cov()
    expected_returns = {
        ticker: float(np.exp(projections[ticker].expected_return) - 1)
        for ticker in tickers
    }

    portfolio = optimize_portfolio(tickers, expected_returns, covariance)

    hedge_scores = {
        ticker: hedge_sentiment.get(ticker, HedgeSentiment(ticker, 0.0, 0.0, [])).score
        for ticker in tickers
    }
    prices = {ticker: snapshots[ticker].price for ticker in tickers}
    targets = {ticker: snapshots[ticker].analyst_target or snapshots[ticker].price for ticker in tickers}

    data_table = build_data_table(signals, hedge_scores, prices, targets)
    summary = compose_summary(query, signals, portfolio)

    insights = _generate_insights(signals, snapshots, technicals_map, projections)
    risks = _default_risks(signals)

    code_snippet = _code_snippet_example(tickers)

    return assemble_report(
        current_date=date.today(),
        query=query,
        summary=summary,
        data_table=data_table,
        insights=insights,
        portfolio=portfolio,
        signals=signals,
        risks=risks,
        code_snippet=code_snippet,
    )


def _generate_insights(
    signals: List[InvestmentSignal],
    snapshots: Dict[str, MarketSnapshot],
    technicals_map: Dict[str, TechnicalIndicators],
    projections: Dict[str, ScenarioResult],
) -> List[str]:
    insights: List[str] = []
    for signal in signals:
        ticker = signal.ticker
        snapshot = snapshots[ticker]
        technicals = technicals_map[ticker]
        projection = projections[ticker]
        if snapshot.pe_ratio and snapshot.sector_pe:
            gap = snapshot.sector_pe - snapshot.pe_ratio
            if gap > 5:
                insights.append(
                    f"{ticker}: trades at a {gap:.1f}x discount to sector P/E, supporting value thesis."
                )
            elif gap < -5:
                insights.append(
                    f"{ticker}: carries a {abs(gap):.1f}x premium to sector P/E; monitor for de-rating risk."
                )
        if technicals.rsi_14 < 35:
            insights.append(f"{ticker}: RSI {technicals.rsi_14:.1f} signals potential mean reversion entry.")
        if technicals.sma_50 > technicals.sma_200:
            insights.append(f"{ticker}: 50/200-day crossover remains constructive for momentum investors.")
        exp_move = np.exp(projection.expected_return) - 1
        if exp_move > 0.15:
            insights.append(
                f"{ticker}: Monte Carlo base case implies {exp_move:.1%} upside over next year under GBM assumption."
            )
    if not insights:
        insights.append("No standout quantitative anomalies detected; consider deeper fundamental diligence.")
    return insights[:8]


def _default_risks(signals: List[InvestmentSignal]) -> List[str]:
    risks = {
        "Model uses simulated data when live feeds are unavailable, which may diverge from actual market conditions.",
        "Technical indicators derived from historical prices may fail in regime shifts or low-liquidity scenarios.",
        "Hedge fund sentiment approximations cannot replace primary source verification of 13F filings.",
    }
    if any(signal.action == "BUY" for signal in signals):
        risks.add("Upside theses assume access to capital and steady macro conditions; recessions can impair outcomes.")
    if any(signal.action == "SELL" for signal in signals):
        risks.add("Short or underweight calls require strict risk management to avoid squeeze-driven losses.")
    return sorted(risks)


def _code_snippet_example(tickers: Iterable[str]) -> str:
    tickers_repr = ", ".join(f"'{t}'" for t in tickers)
    return f"""
from pathlib import Path
from smart_investment_agent import run_analysis

report = run_analysis(
    query="Evaluate smart money positioning",
    tickers=[{tickers_repr}],
    hedge_csv=Path("hedge_positions.csv"),
    market_csv=Path("market_snapshots.csv"),
    price_dir=Path("price_history"),
)
print(report.summary)
"""


def format_report(report: AnalysisReport) -> str:
    allocations_df = pd.DataFrame(
        {
            "Ticker": list(report.portfolio.allocations.keys()),
            "Allocation": [f"{w:.1%}" for w in report.portfolio.allocations.values()],
        }
    )
    allocations_md = allocations_df.to_markdown(index=False)

    insights_md = "\n".join(f"- {item}" for item in report.insights)
    risks_md = "\n".join(f"- {item}" for item in report.risks)

    return f"""
# Smart Investment Analysis
- Date: {report.current_date.isoformat()}
- Query: {report.query}

## Executive Summary
{report.summary}

## Data Table
{report.data_table_markdown}

## Insights
{insights_md}

## Portfolio Recommendation
Expected Return: {report.portfolio.expected_return:.2%}  \\
Expected Volatility: {report.portfolio.expected_volatility:.2%}  \\
Sharpe Ratio: {report.portfolio.sharpe_ratio:.2f}

{allocations_md}

## Risks & Disclaimer
{risks_md}

> {report.disclaimer}

## Reproducible Code Snippet
```python
{report.code_snippet}
```
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the smart investment analysis agent")
    parser.add_argument("query", help="User question or investment theme to analyze")
    parser.add_argument("--tickers", help="Comma-separated list of tickers to focus on")
    parser.add_argument("--hedge-csv", type=Path, help="CSV file with hedge fund positions")
    parser.add_argument("--market-csv", type=Path, help="CSV file with market metrics")
    parser.add_argument("--price-dir", type=Path, help="Directory containing per-ticker price history CSVs")
    parser.add_argument("--output", type=Path, help="Optional path to save the markdown report")

    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    report = run_analysis(
        query=args.query,
        tickers=tickers,
        hedge_csv=args.hedge_csv,
        market_csv=args.market_csv,
        price_dir=args.price_dir,
    )

    markdown = format_report(report)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":  # pragma: no cover
    main()

