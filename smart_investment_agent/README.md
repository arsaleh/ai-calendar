# Smart Investment Agent

This module implements a reference workflow for the "Generic Hedge Fund and Market Data Analysis" coding agent.
It mirrors the structure described in the prompt: data collection, quantitative crunching, LLM-style reasoning, and
rich report generation.  The package exposes a `run_analysis` function and a CLI that can operate on either live CSV
feeds or the bundled synthetic sample data found in `sample_data/`.

Key features:

- Pluggable data loaders that accept CSV exports from 13F filings, valuation snapshots, or price history feeds.
- Quantitative analytics including hedge fund sentiment scoring, technical indicators, Monte Carlo projections,
  and a portfolio optimization routine with a random-search fallback when PuLP is unavailable.
- Markdown report construction that aligns with the prompt's executive-summary/table/insights/portfolio/risks layout.
- Pytest coverage (skipped automatically when `numpy`/`pandas` are missing) demonstrating how to exercise the agent
  end-to-end against the bundled synthetic dataset.

To run the agent against the example data:

```bash
python -m smart_investment_agent.main "Scan AI leaders" \
  --tickers NVDA,TSLA,MSFT,AAPL \
  --hedge-csv smart_investment_agent/sample_data/hedge_positions.csv \
  --market-csv smart_investment_agent/sample_data/market_snapshots.csv \
  --price-dir smart_investment_agent/sample_data/price_history
```

Dependencies: `numpy`, `pandas`, and optionally `pulp` for linear programming.  Install them with
`pip install -r requirements.txt` (not provided) or via your preferred environment manager.
