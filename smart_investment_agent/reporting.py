"""Reporting helpers for the smart investment agent."""
from __future__ import annotations

from datetime import date
from textwrap import shorten
from typing import Dict, Iterable, List

import pandas as pd

from .data_models import AnalysisReport, InvestmentSignal, PortfolioRecommendation


def build_data_table(signals: Iterable[InvestmentSignal], hedge_scores: Dict[str, float], prices: Dict[str, float], targets: Dict[str, float]) -> str:
    rows = []
    for signal in signals:
        rationale = signal.rationale
        if signal.timing_hint:
            rationale = f"{rationale}. Timing: {signal.timing_hint}"
        rows.append(
            {
                "Ticker": signal.ticker,
                "Hedge Score": round(hedge_scores.get(signal.ticker, 0.0), 2),
                "Current Price": round(prices.get(signal.ticker, float("nan")), 2),
                "1Y Target": round(targets.get(signal.ticker, float("nan")), 2),
                "Signal": signal.action,
                "Rationale": shorten(rationale, width=120, placeholder="…"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return "| Ticker | Hedge Score | Current Price | 1Y Target | Signal | Rationale |\n|---|---|---|---|---|---|"
    return df.to_markdown(index=False)


def compose_summary(query: str, signals: List[InvestmentSignal], portfolio: PortfolioRecommendation) -> str:
    if not signals:
        return f"No actionable insights generated for '{query}'."

    buys = [s.ticker for s in signals if s.action == "BUY"]
    sells = [s.ticker for s in signals if s.action == "SELL"]
    holds = [s.ticker for s in signals if s.action == "HOLD"]

    highlight = "Top conviction signals: "
    parts = []
    if buys:
        parts.append(f"BUY {', '.join(buys)}")
    if sells:
        parts.append(f"SELL {', '.join(sells)}")
    if not parts and holds:
        parts.append(f"Maintain exposure to {', '.join(holds)}")
    highlight += "; ".join(parts) if parts else "No high-conviction trades."

    return (
        f"{highlight} Portfolio optimization suggests expected return of "
        f"{portfolio.expected_return:.2%} with Sharpe {portfolio.sharpe_ratio:.2f}."
    )


def assemble_report(
    *,
    current_date: date,
    query: str,
    summary: str,
    data_table: str,
    insights: List[str],
    portfolio: PortfolioRecommendation,
    signals: List[InvestmentSignal],
    risks: List[str],
    code_snippet: str,
) -> AnalysisReport:
    disclaimer = (
        "This analysis is for informational purposes only and does not constitute investment advice. "
        "Market data may include simulated values when live sources are unavailable."
    )

    return AnalysisReport(
        current_date=current_date,
        query=query,
        summary=summary,
        data_table_markdown=data_table,
        insights=insights,
        portfolio=portfolio,
        signals=signals,
        risks=risks,
        disclaimer=disclaimer,
        code_snippet=code_snippet.strip(),
    )

