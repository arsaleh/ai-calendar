import pathlib

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from smart_investment_agent.main import format_report, run_analysis


def test_run_analysis_with_sample_data():
    base = pathlib.Path(__file__).resolve().parent.parent / "smart_investment_agent" / "sample_data"
    report = run_analysis(
        query="Evaluate AI leaders",
        tickers=["NVDA", "MSFT", "TSLA"],
        hedge_csv=base / "hedge_positions.csv",
        market_csv=base / "market_snapshots.csv",
        price_dir=base / "price_history",
    )

    assert report.summary
    assert "|" in report.data_table_markdown
    assert report.portfolio.allocations

    markdown = format_report(report)
    assert "## Executive Summary" in markdown
    assert "Reproducible Code Snippet" in markdown
