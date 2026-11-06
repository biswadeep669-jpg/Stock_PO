# NIFTY 50 Portfolio Optimization: MVP vs HRP

A Streamlit dashboard comparing two portfolio-construction strategies using daily **raw Close prices** for ten selected NIFTY 50 stocks:

- **Minimum Variance Portfolio (MVP)**
- **Hierarchical Risk Parity (HRP)**

The app estimates portfolio weights from a training period, holds those weights fixed, and evaluates performance on a separate out-of-sample test period.

## 1. Project objective

The project investigates whether HRP or MVP delivers better out-of-sample performance for a selected basket of Indian large-cap equities.

The dashboard compares total return, annualized return (CAGR), annualized volatility, Sharpe ratio, maximum drawdown, cumulative growth, and stock-level portfolio allocations.

> This is an educational and quantitative research tool, not investment advice or a forecast.

## 2. Default analysis period

| Dataset | Default period | Purpose |
|---|---|---|
| Training / in-sample | 1 January 2025 – 30 April 2026 | Estimate portfolio weights |
| Testing / out-of-sample | 1 May 2026 – 31 October 2026 | Evaluate fixed-weight portfolios |

The end date is treated as inclusive by the app. Yahoo Finance uses an exclusive end date, so the program adds one day to the requested end date.

You can change the dates in the Streamlit sidebar. The test start date separates training observations from testing observations.

## 3. Default stock universe

The app is preconfigured with these Yahoo Finance symbols:

| Company | Yahoo Finance ticker |
|---|---|
| HDFC Bank | `HDFCBANK.NS` |
| ICICI Bank | `ICICIBANK.NS` |
| Reliance Industries | `RELIANCE.NS` |
| Bharti Airtel | `BHARTIARTL.NS` |
| Larsen & Toubro | `LT.NS` |
| State Bank of India | `SBIN.NS` |
| Axis Bank | `AXISBANK.NS` |
| Infosys | `INFY.NS` |
| Kotak Mahindra Bank | `KOTAKBANK.NS` |
| Mahindra & Mahindra | `M&M.NS` |

These are the application's default symbols, not a guarantee that they were the ten largest NIFTY 50 constituents on every historical date. Verify point-in-time index constituents and weights from official NSE/Nifty Indices records before presenting the results as a historically unbiased NIFTY 50 study.

The sidebar allows you to choose the ten stocks used in the analysis.

## 4. Data source and price field

Historical data is downloaded from Yahoo Finance through `yfinance`.

- Price field used: **`Close` only**.
- `Adj Close`, `Open`, `High`, `Low`, and `Volume` are not used.
- Daily simple returns:

  `r_t = (P_t / P_(t-1)) - 1`

The app does not forward-fill missing prices. It retains common complete trading dates across the selected stocks.

**Important:** Raw Close prices are not adjusted for dividends and may not reflect split-adjusted total shareholder returns. Therefore, this backtest measures price returns, not total returns. This limitation is intentional because the project specifies Close prices only.

## 5. Portfolio construction

### 5.1 Minimum Variance Portfolio (MVP)

MVP minimizes training-period portfolio variance:

`minimize w.T @ covariance_matrix @ w`

Subject to:

- The weights sum to 1 (100% invested).
- Each weight is between 0 and 1.
- No short selling or leverage is permitted.

The application uses SciPy's SLSQP optimizer.

### 5.2 Hierarchical Risk Parity (HRP)

HRP uses the correlation structure of the assets:

1. Calculate training-period covariance and correlation matrices.
2. Convert correlations into distances: `distance = sqrt((1 - correlation) / 2)`.
3. Apply hierarchical clustering using single linkage.
4. Quasi-diagonalize assets using the cluster ordering.
5. Apply recursive bisection, assigning less weight to the higher-variance cluster.

HRP does not require direct inversion of the covariance matrix and does not use expected-return estimates to determine weights.

## 6. Backtesting design

Weights are estimated using training-period daily returns only. They are then applied to the test-period stock returns:

`Portfolio return at t = sum(weight_i * stock return_i at t)`

Weights remain fixed throughout the test period. This is a fixed-weight, buy-and-hold-weight simulation; it does not periodically rebalance to the original weights.

The first test-period return is calculated using the last training-period Close and the first test-period Close, including the return across the training/test boundary.

## 7. Performance measures

The annualization factor defaults to 252 trading days and can be changed in the sidebar.

- **Total return:** compound growth over the test observations: `product(1 + portfolio_daily_return) - 1`.
- **Annualized return (CAGR):** `(1 + total_return) ** (annualization_factor / number_of_test_returns) - 1`.
- **Annualized volatility:** daily return standard deviation multiplied by the square root of the annualization factor.
- **Sharpe ratio:** annualized average daily excess return divided by daily return standard deviation. The annual risk-free rate is converted to an effective daily rate first. The default is 6.5%.
- **Maximum drawdown:** the minimum of `(portfolio_value / running_peak) - 1`.

## 8. Dashboard outputs

The app displays:

1. Data and backtest overview, including selected stocks and observation counts.
2. A preview of downloaded Close prices.
3. An out-of-sample performance comparison table.
4. Cumulative growth of ₹1 invested in each strategy.
5. Annualized volatility comparison chart.
6. Sharpe ratio comparison chart.
7. Drawdown comparison chart.
8. Separate MVP and HRP allocation pie charts.
9. Side-by-side stock allocation table.
10. CSV exports for allocations, performance metrics, and daily portfolio returns.

## 9. Installation and execution

### Requirements

- Python 3.9 or later recommended.
- Internet connection for Yahoo Finance downloads.
- The packages listed in `requirements.txt`.

### Step 1: Put the project files in one folder

```text
Portfolio_Optimization/
├── app.py
├── requirements.txt
└── README.md
```

### Step 2: Open a terminal in that folder

A virtual environment is recommended.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the app

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## 10. How to use the dashboard

1. Open the app in your browser.
2. Set the historical start date, test start date, and historical end date in the sidebar.
3. Enter the annual risk-free rate and annualization factor.
4. Select exactly ten stocks.
5. Allow Yahoo Finance data to download and both portfolios to be optimized.
6. Compare the out-of-sample metrics and charts.
7. Download the allocation, performance, and daily-return CSV files.

If a stock's data is unavailable, check its Yahoo Finance symbol and date range. The app requires a common complete price history for all selected stocks.

## 11. Important research limitations

Disclose these limitations in an academic report or presentation:

- **Universe / survivorship bias:** Selecting today's or a later date's NIFTY 50 constituents to study an earlier period can introduce survivorship and look-ahead bias. Use a point-in-time universe known before the test period for a rigorous historical study.
- **Raw Close prices:** Dividends and corporate actions are not included as they would be in a total-return series.
- **No trading costs:** Brokerage, taxes, exchange charges, and slippage are excluded.
- **No rebalancing:** Weights are fixed during the test period. Results may differ under monthly or quarterly rebalancing.
- **Parameter sensitivity:** MVP can be sensitive to estimated covariance matrices. HRP depends on correlation distances and clustering choices.
- **Historical performance:** A higher test-period Sharpe ratio or return does not establish future outperformance.
- **Data quality:** Yahoo Finance availability, symbol history, missing observations, and corporate-action handling can affect results.

For a stronger research design, consider a rolling or walk-forward backtest, a point-in-time NIFTY 50 universe, explicit transaction costs, and a benchmark such as the NIFTY 50 TRI. The benchmark would be for comparison only; this app's stock-level optimization continues to use raw Close prices as specified.

## 12. Suggested research question

**Does Hierarchical Risk Parity provide better out-of-sample risk-adjusted performance than the traditional Minimum Variance Portfolio when applied to a selected basket of NIFTY 50 equities?**

Useful discussion questions:

- Which strategy produces the higher test-period return?
- Which strategy has lower annualized volatility?
- Which strategy has the higher Sharpe ratio?
- Which strategy experiences the smaller maximum drawdown?
- Does HRP distribute weights more evenly across correlated groups of stocks?
- Does MVP concentrate capital in a small number of low-volatility or highly correlated stocks?
- Are performance differences economically meaningful after transaction costs?

## 13. Project files

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard and portfolio optimization logic |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation and execution instructions |

## Disclaimer

This project is intended for learning, research, and demonstration. It does not constitute investment advice, a recommendation to buy or sell securities, or a guarantee of future performance.
