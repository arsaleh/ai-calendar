"""Data models used by the smart investment agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class HedgeFundPosition:
    """Represents a single hedge fund's position in a security."""

    fund_name: str
    ticker: str
    shares: float
    market_value: float
    portfolio_weight: float
    qoq_change_pct: float


@dataclass
class MarketSnapshot:
    """Simplified market metrics for a security."""

    ticker: str
    price: float
    pe_ratio: Optional[float]
    sector_pe: Optional[float]
    eps_growth_1y: Optional[float]
    analyst_target: Optional[float]
    free_text_summary: str = ""


@dataclass
class TechnicalIndicators:
    """Key technical indicator outputs."""

    sma_50: float
    sma_200: float
    rsi_14: float
    macd: float
    macd_signal: float


@dataclass
class HedgeSentiment:
    """Aggregated hedge fund sentiment for a ticker."""

    ticker: str
    score: float
    total_net_change: float
    contributors: List[HedgeFundPosition] = field(default_factory=list)


@dataclass
class PortfolioRecommendation:
    """Recommended allocation for a hypothetical portfolio."""

    allocations: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


@dataclass
class InvestmentSignal:
    """Final recommendation for a security."""

    ticker: str
    action: str
    rationale: str
    timing_hint: Optional[str] = None


@dataclass
class AnalysisReport:
    """Container for final analysis artefacts."""

    current_date: date
    query: str
    summary: str
    data_table_markdown: str
    insights: List[str]
    portfolio: PortfolioRecommendation
    signals: List[InvestmentSignal]
    risks: List[str]
    disclaimer: str
    code_snippet: str

