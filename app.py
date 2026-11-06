# app.py
# Streamlit dashboard: Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)
# Data source: Yahoo Finance via yfinance; uses ONLY raw "Close" prices.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="MVP vs HRP Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Optimization: MVP vs HRP")
st.caption(
    "NIFTY 50 large-cap stocks | Close-price-only analysis | "
    "In-sample portfolio construction and out-of-sample testing"
)

# Top 10 NIFTY 50 constituents by index weight as of 30-Sep-2026.
# Source reference: NIFTY 50 factsheet / constituent weights.
# Yahoo Finance symbols use the .NS suffix.
DEFAULT_TICKERS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Infosys": "INFY.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Mahindra & Mahindra": "M&M.NS",
}

# ----------------------------- SIDEBAR -----------------------------
st.sidebar.header("⚙️ Analysis settings")

start_date = st.sidebar.date_input(
    "Historical data start date", value=pd.Timestamp("2025-01-01").date()
)
end_date_inclusive = st.sidebar.date_input(
    "Historical data end date", value=pd.Timestamp("2026-10-31").date()
)
test_start = st.sidebar.date_input(
    "Out-of-sample test start date", value=pd.Timestamp("2026-05-01").date()
)
risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate (%)", min_value=0.0, max_value=30.0,
    value=6.5, step=0.25
) / 100
trading_days = st.sidebar.number_input(
    "Annualization factor (trading days)", min_value=200, max_value=260,
    value=252, step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Select 10 NIFTY 50 stocks")
selected_names = st.sidebar.multiselect(
    "Stocks (Yahoo Finance symbols)",
    options=list(DEFAULT_TICKERS.keys()),
    default=list(DEFAULT_TICKERS.keys()),
)
if len(selected_names) != 10:
    st.sidebar.warning("For this comparison, select exactly 10 stocks.")

selected_tickers = [DEFAULT_TICKERS[name] for name in selected_names]

# ----------------------------- HELPERS -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_close_prices(tickers, start, end_exclusive):
    """Download raw Close prices only. Yahoo end date is exclusive."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    # yfinance may return either a MultiIndex or a single ticker's flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response does not contain a Close column.")
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Yahoo Finance response does not contain a Close column.")
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close.sort_index()
    close = close.loc[:, [c for c in tickers if c in close.columns]]
    # Do not forward-fill prices: that can create artificial zero-return observations.
    close = close.dropna(axis=1, how="all")
    return close

def clean_prices(prices):
    """Keep securities with complete observations across the chosen sample."""
    if prices.empty:
        return prices
    # Require all selected assets to have data on a date; avoid pairwise sample mismatch.
    prices = prices.dropna(axis=0, how="any")
    return prices

def daily_returns(close_prices):
    return close_prices.pct_change().dropna()

def portfolio_variance(weights, cov_matrix):
    return float(weights.T @ cov_matrix @ weights)

def optimize_mvp(train_returns):
    """Long-only, fully-invested minimum-variance portfolio."""
    cov = train_returns.cov().values
    n = train_returns.shape[1]
    initial = np.repeat(1 / n, n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        portfolio_variance,
        initial,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"MVP optimization failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return pd.Series(weights / weights.sum(), index=train_returns.columns, name="MVP")

def get_cluster_variance(cov, cluster_items):
    """Variance of an inverse-variance-weighted cluster portfolio."""
    cov_slice = cov.loc[cluster_items, cluster_items]
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp /= ivp.sum()
    return float(ivp @ cov_slice.values @ ivp)

def optimize_hrp(train_returns):
    """Hierarchical Risk Parity using correlation distance and recursive bisection."""
    cov = train_returns.cov()
    corr = train_returns.corr().clip(-1, 1)

    # López de Prado correlation distance.
    # Convert to an independent, writable NumPy array before modifying the diagonal.
    # Some pandas/NumPy combinations expose DataFrame.values as read-only, which
    # can otherwise raise: "underlying array is read-only" in np.fill_diagonal.
    corr_values = corr.to_numpy(dtype=float, copy=True)
    distance_values = np.sqrt(np.maximum(0.0, (1.0 - corr_values) / 2.0))
    np.fill_diagonal(distance_values, 0.0)

    if not np.isfinite(distance_values).all():
        raise ValueError(
            "The HRP correlation-distance matrix contains NaN or infinite values. "
            "Check for stocks with constant or insufficient return observations."
        )

    condensed = squareform(distance_values, checks=False)
    link = linkage(condensed, method="single")

    # Quasi-diagonalization: order leaves according to the dendrogram.
    from scipy.cluster.hierarchy import leaves_list
    sort_ix = corr.index[leaves_list(link)].tolist()

    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            var_left = get_cluster_variance(cov, left)
            var_right = get_cluster_variance(cov, right)
            alpha_left = 1 - var_left / (var_left + var_right)

            weights[left] *= alpha_left
            weights[right] *= (1 - alpha_left)

            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = weights.reindex(train_returns.columns).fillna(0)
    weights = weights / weights.sum()
    weights.name = "HRP"
    return weights

def performance_metrics(portfolio_returns, rf_annual, periods=252):
    """Metrics computed on the out-of-sample daily return series."""
    r = portfolio_returns.dropna()
    if len(r) < 2:
        raise ValueError("Not enough out-of-sample return observations.")
    wealth = (1 + r).cumprod()
    total_return = wealth.iloc[-1] - 1
    years = len(r) / periods
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = r.std(ddof=1) * np.sqrt(periods)
    daily_rf = (1 + rf_annual) ** (1 / periods) - 1
    excess = r - daily_rf
    sharpe = (excess.mean() / r.std(ddof=1)) * np.sqrt(periods) if r.std(ddof=1) != 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    return {
        "Total Return": total_return,
        "Annualized Return (CAGR)": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_dd,
    }, wealth, drawdown

# ----------------------------- VALIDATION & DATA -----------------------------
if len(selected_names) != 10:
    st.info("Choose exactly 10 stocks in the sidebar to run the analysis.")
    st.stop()

if start_date >= test_start or test_start > end_date_inclusive:
    st.error("Date settings are inconsistent. Ensure start < test start ≤ end date.")
    st.stop()

# yfinance's end parameter is exclusive, so add one day to include 31-Oct-2026.
end_exclusive = (pd.Timestamp(end_date_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

try:
    with st.spinner("Downloading daily Close prices from Yahoo Finance..."):
        close_all = download_close_prices(
            selected_tickers,
            pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            end_exclusive,
        )
except Exception as exc:
    st.error(f"Data download failed: {exc}")
    st.stop()

if close_all.empty:
    st.error("No price data was returned. Check your internet connection and ticker symbols.")
    st.stop()

missing = sorted(set(selected_tickers) - set(close_all.columns))
if missing:
    st.error(
        "Price data is missing for: " + ", ".join(missing) +
        ". Check the Yahoo Finance symbols or choose another stock."
    )
    st.stop()

close_all = clean_prices(close_all)
if close_all.empty:
    st.error("No common complete trading dates were found for all selected stocks.")
    st.stop()

train_close = close_all.loc[close_all.index < pd.Timestamp(test_start)]
test_close = close_all.loc[close_all.index >= pd.Timestamp(test_start)]

train_ret = daily_returns(train_close)
test_ret = daily_returns(test_close)

# The first test return is calculated from the first test-day close relative to
# the last available training-day close, so the boundary-day return is included.
if len(test_close) > 0 and len(train_close) > 0:
    boundary_prices = pd.concat([train_close.tail(1), test_close])
    test_ret = daily_returns(boundary_prices).loc[test_close.index]

if len(train_ret) < 60 or len(test_ret) < 20:
    st.error(
        f"Insufficient observations. Training returns: {len(train_ret)}; "
        f"testing returns: {len(test_ret)}. Check the dates and ticker data."
    )
    st.stop()

# ----------------------------- OPTIMIZATION -----------------------------
try:
    mvp_weights = optimize_mvp(train_ret)
    hrp_weights = optimize_hrp(train_ret)
except Exception as exc:
    st.error(f"Portfolio optimization failed: {exc}")
    st.stop()

weights_df = pd.concat([mvp_weights, hrp_weights], axis=1)
weights_df.index = selected_names

# Weights are fixed at the end of the training period; no rebalancing in test period.
test_returns = test_ret.copy()
test_returns.columns = selected_names
portfolio_returns = pd.DataFrame({
    "MVP": test_returns @ mvp_weights.values,
    "HRP": test_returns @ hrp_weights.values,
})

mvp_metrics, mvp_wealth, mvp_dd = performance_metrics(
    portfolio_returns["MVP"], risk_free_rate, int(trading_days)
)
hrp_metrics, hrp_wealth, hrp_dd = performance_metrics(
    portfolio_returns["HRP"], risk_free_rate, int(trading_days)
)

metrics_df = pd.DataFrame([mvp_metrics, hrp_metrics], index=["MVP", "HRP"])

# ----------------------------- SUMMARY -----------------------------
st.markdown("### 1. Data and backtest overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks selected", len(selected_names))
c2.metric("Training observations", len(train_ret))
c3.metric("Testing observations", len(test_ret))
c4.metric("Risk-free rate", f"{risk_free_rate:.2%}")

st.write(
    f"**Training window:** {train_ret.index.min().date()} to {train_ret.index.max().date()}  "
    f"| **Testing window:** {test_ret.index.min().date()} to {test_ret.index.max().date()}"
)
st.caption(
    "The portfolios are optimized only on the training sample. Their weights are then held "
    "constant throughout the test period (buy-and-hold weights; no periodic rebalancing)."
)

with st.expander("View downloaded Close prices"):
    st.dataframe(close_all.tail(20), use_container_width=True)

# ----------------------------- PERFORMANCE TABLE -----------------------------
st.markdown("### 2. Out-of-sample performance comparison")
display_metrics = metrics_df.copy()
for col in ["Total Return", "Annualized Return (CAGR)", "Annualized Volatility", "Maximum Drawdown"]:
    display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
display_metrics["Sharpe Ratio"] = display_metrics["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
st.dataframe(display_metrics, use_container_width=True)

best_sharpe = metrics_df["Sharpe Ratio"].idxmax()
best_return = metrics_df["Total Return"].idxmax()
lowest_risk = metrics_df["Annualized Volatility"].idxmin()
st.info(
    f"**Test-period leaders:** Sharpe ratio – {best_sharpe}; "
    f"total return – {best_return}; lowest annualized volatility – {lowest_risk}. "
    "These are historical test results, not forecasts."
)

# ----------------------------- CHARTS -----------------------------
st.markdown("### 3. Cumulative growth of ₹1")
growth_df = pd.DataFrame({"MVP": mvp_wealth, "HRP": hrp_wealth})
growth_fig = go.Figure()
for col in growth_df.columns:
    growth_fig.add_trace(go.Scatter(
        x=growth_df.index, y=growth_df[col], mode="lines", name=col
    ))
growth_fig.update_layout(
    xaxis_title="Date", yaxis_title="Portfolio value (₹ per ₹1 invested)",
    legend_title="Strategy", hovermode="x unified"
)
st.plotly_chart(growth_fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 4. Annualized risk")
    risk_fig = px.bar(
        x=metrics_df.index,
        y=metrics_df["Annualized Volatility"] * 100,
        labels={"x": "Strategy", "y": "Annualized volatility (%)"},
        text=(metrics_df["Annualized Volatility"] * 100).map(lambda x: f"{x:.2f}%"),
    )
    risk_fig.update_traces(textposition="outside")
    st.plotly_chart(risk_fig, use_container_width=True)

with col2:
    st.markdown("### 5. Sharpe ratio")
    sharpe_fig = px.bar(
        x=metrics_df.index,
        y=metrics_df["Sharpe Ratio"],
        labels={"x": "Strategy", "y": "Sharpe ratio"},
        text=metrics_df["Sharpe Ratio"].map(lambda x: f"{x:.3f}"),
    )
    sharpe_fig.update_traces(textposition="outside")
    st.plotly_chart(sharpe_fig, use_container_width=True)

st.markdown("### 6. Drawdown comparison")
dd_df = pd.DataFrame({"MVP": mvp_dd, "HRP": hrp_dd})
dd_fig = go.Figure()
for col in dd_df.columns:
    dd_fig.add_trace(go.Scatter(
        x=dd_df.index, y=dd_df[col] * 100, mode="lines", name=col
    ))
dd_fig.update_layout(
    xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode="x unified"
)
st.plotly_chart(dd_fig, use_container_width=True)

# ----------------------------- ALLOCATIONS -----------------------------
st.markdown("### 7. Portfolio allocations")
pie1, pie2 = st.columns(2)
with pie1:
    st.markdown("**Minimum Variance Portfolio (MVP)**")
    fig_mvp = px.pie(
        names=weights_df.index, values=weights_df["MVP"],
        hole=0.35, title="MVP allocation"
    )
    fig_mvp.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_mvp, use_container_width=True)

with pie2:
    st.markdown("**Hierarchical Risk Parity (HRP)**")
    fig_hrp = px.pie(
        names=weights_df.index, values=weights_df["HRP"],
        hole=0.35, title="HRP allocation"
    )
    fig_hrp.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_hrp, use_container_width=True)

st.markdown("### 8. Allocation comparison table")
allocation_table = weights_df.copy()
allocation_table["MVP"] = allocation_table["MVP"].map(lambda x: f"{x:.2%}")
allocation_table["HRP"] = allocation_table["HRP"].map(lambda x: f"{x:.2%}")
st.dataframe(allocation_table, use_container_width=True)

# ----------------------------- DOWNLOADS -----------------------------
st.markdown("### 9. Export results")
weights_csv = weights_df.to_csv().encode("utf-8")
metrics_csv = metrics_df.to_csv().encode("utf-8")
returns_csv = portfolio_returns.to_csv().encode("utf-8")
d1, d2, d3 = st.columns(3)
d1.download_button("Download allocations CSV", weights_csv, "portfolio_allocations.csv", "text/csv")
d2.download_button("Download performance CSV", metrics_csv, "portfolio_performance.csv", "text/csv")
d3.download_button("Download test returns CSV", returns_csv, "portfolio_test_returns.csv", "text/csv")

st.markdown("---")
st.markdown("### Methodology and interpretation notes")
st.markdown(
    """
    - **Price input:** raw Yahoo Finance `Close` only; `Adj Close`, Open, High, Low and Volume are not used.
    - **Daily return:** simple percentage return, `P_t / P_(t-1) - 1`.
    - **MVP:** long-only, fully invested (weights between 0 and 100%), minimizing training-sample portfolio variance.
    - **HRP:** correlation-distance clustering, quasi-diagonalization and recursive bisection using training-sample covariance.
    - **Annualized volatility:** daily standard deviation × square root of annualization factor.
    - **Sharpe ratio:** annualized mean daily excess return divided by daily standard deviation, using the entered annual risk-free rate.
    - **Test return:** daily portfolio return using fixed training-derived weights. Transaction costs, taxes, slippage and rebalancing are excluded.
    - **Close-price limitation:** raw Close prices do not adjust for dividends or stock splits. Results are price-return results, not total-return results.
    """
)

st.warning(
    "Research limitation: the default ten-stock universe is not reconstructed from "
    "point-in-time NIFTY 50 constituent weights. Using a universe selected after the "
    "start of the May–October 2026 test window can introduce look-ahead / survivorship "
    "bias. For a strictly unbiased historical backtest, select the top ten constituents "
    "using information available before the test period (for example, 30-Apr-2026), "
    "and keep that universe fixed."
)
