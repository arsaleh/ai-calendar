"""Utility functions to source or simulate investment data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data_models import HedgeFundPosition, MarketSnapshot


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV file at {path} was not found.")
    return pd.read_csv(path)


def load_hedge_fund_positions(
    tickers: Iterable[str],
    csv_path: Optional[Path] = None,
) -> List[HedgeFundPosition]:
    """Load hedge fund positions for the given tickers.

    Parameters
    ----------
    tickers:
        The securities that should be present in the returned dataset.
    csv_path:
        Optional path to a CSV file with columns matching
        :class:`HedgeFundPosition`. When omitted, synthetic placeholder data
        is generated that roughly emulates 13F activity.
    """

    tickers = list({ticker.upper() for ticker in tickers})
    if not tickers:
        return []

    if csv_path is not None:
        df = _read_csv(csv_path)
    else:
        rng = np.random.default_rng(42)
        funds = [
            "Appaloosa",
            "Bridgewater",
            "Citadel",
            "Coatue",
            "Two Sigma",
        ]
        rows = []
        for ticker in tickers:
            for fund in funds:
                shares = rng.uniform(50_000, 500_000)
                price = rng.uniform(10, 400)
                market_value = shares * price
                weight = rng.uniform(0.5, 5.0)
                qoq = rng.uniform(-0.4, 0.6)
                rows.append(
                    {
                        "fund_name": fund,
                        "ticker": ticker,
                        "shares": shares,
                        "market_value": market_value,
                        "portfolio_weight": weight,
                        "qoq_change_pct": qoq,
                    }
                )
        df = pd.DataFrame(rows)

    positions = [
        HedgeFundPosition(
            fund_name=row["fund_name"],
            ticker=row["ticker"].upper(),
            shares=float(row["shares"]),
            market_value=float(row["market_value"]),
            portfolio_weight=float(row["portfolio_weight"]),
            qoq_change_pct=float(row["qoq_change_pct"]),
        )
        for _, row in df.iterrows()
        if row["ticker"].upper() in tickers
    ]
    return positions


def load_market_snapshots(
    tickers: Iterable[str],
    csv_path: Optional[Path] = None,
) -> Dict[str, MarketSnapshot]:
    """Load core valuation metrics for tickers.

    The data can be sourced from a CSV or simulated to keep the pipeline
    functional without internet access.
    """

    tickers = [ticker.upper() for ticker in tickers]
    if csv_path is not None:
        df = _read_csv(csv_path)
    else:
        rng = np.random.default_rng(123)
        df = pd.DataFrame(
            {
                "ticker": tickers,
                "price": rng.uniform(20, 350, size=len(tickers)),
                "pe_ratio": rng.uniform(8, 40, size=len(tickers)),
                "sector_pe": rng.uniform(10, 35, size=len(tickers)),
                "eps_growth_1y": rng.uniform(-0.1, 0.45, size=len(tickers)),
                "analyst_target": rng.uniform(25, 420, size=len(tickers)),
                "summary": [
                    "Simulated valuation snapshot pending live data" for _ in tickers
                ],
            }
        )

    snapshots: Dict[str, MarketSnapshot] = {}
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        if ticker not in tickers:
            continue
        snapshots[ticker] = MarketSnapshot(
            ticker=ticker,
            price=float(row["price"]),
            pe_ratio=_maybe_float(row.get("pe_ratio")),
            sector_pe=_maybe_float(row.get("sector_pe")),
            eps_growth_1y=_maybe_float(row.get("eps_growth_1y")),
            analyst_target=_maybe_float(row.get("analyst_target")),
            free_text_summary=str(row.get("summary", "")),
        )
    return snapshots


def load_price_history(
    ticker: str,
    csv_path: Optional[Path] = None,
    periods: int = 400,
) -> pd.Series:
    """Return a synthetic or CSV-backed historical price series."""

    ticker = ticker.upper()
    if csv_path is not None:
        df = _read_csv(csv_path)
        if "close" not in df.columns:
            raise ValueError("CSV must contain a 'close' column.")
        return pd.Series(df["close"], name=ticker)

    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    steps = rng.normal(loc=0.0008, scale=0.02, size=periods)
    price = 100 * np.exp(np.cumsum(steps))
    index = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="B")
    return pd.Series(price, index=index, name=ticker)


def _maybe_float(value: Optional[float]) -> Optional[float]:
    try:
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)
    except (TypeError, ValueError):
        return None

