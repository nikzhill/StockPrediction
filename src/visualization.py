# visualization.py - Streamlit dashboard: robust live price + updated features + prediction
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import yfinance as yf

# Try to import ta, otherwise fallback
try:
    import ta

    HAS_TA = True
except Exception:
    HAS_TA = False

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="AI Stock Prediction", page_icon="📈", layout="wide")
st.title("📈 AI Stock Price Forecasting")
st.markdown("**MCA Project**")

# ------------------------------
# Sidebar - Controls
# ------------------------------
st.sidebar.header("Controls")
stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]
selected_stock = st.sidebar.selectbox("Select Stock", stocks)


# ------------------------------
# Helpers
# ------------------------------
@st.cache_data
def load_data(ticker: str):
    file_path = f"data/processed/processed_{ticker}_stock_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    return None


@st.cache_data
def load_model(ticker: str):
    candidates = [
        f"models/{ticker}_xgboost.pkl",
        f"models/{ticker}_xgb_model.joblib",
        f"models/{ticker}_xgboost.joblib",
        f"models/xgb_model.joblib",
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    return None


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_latest_price(ticker: str):
    try:
        t = yf.Ticker(ticker)
    except Exception:
        return None, None, None
    # Try fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            if isinstance(fi, dict):
                for key in (
                    "last_price",
                    "last_trade_price",
                    "lastclose",
                    "last_close",
                    "last",
                ):
                    val = fi.get(key)
                    if val is not None:
                        return float(val), "fast_info", None
            else:
                for k in ("last_price", "last_trade_price"):
                    val = getattr(fi, k, None)
                    if val is not None:
                        return float(val), "fast_info", None
    except Exception:
        pass
    # Try info
    try:
        info = t.info or {}
        price = info.get("regularMarketPrice") or info.get("previousClose")
        ts = info.get("regularMarketTime") or info.get("quoteTimeEpoch")
        ts_dt = None
        if price is not None:
            if ts:
                try:
                    ts_dt = pd.to_datetime(int(ts), unit="s")
                except Exception:
                    ts_dt = None
            return float(price), "info", ts_dt
    except Exception:
        pass
    # Try intraday history
    try:
        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist is None or hist.empty:
            hist = t.history(period="2d", interval="1m", prepost=True)
        if hist is not None and not hist.empty:
            last_idx = hist.index[-1]
            last_price = hist["Close"].iloc[-1]
            return float(last_price), "history_1m", pd.to_datetime(last_idx)
    except Exception:
        pass
    # yf.download fallback
    try:
        dl = yf.download(ticker, period="1d", interval="1m", progress=False)
        if dl is not None and not dl.empty:
            last_idx = dl.index[-1]
            last_price = dl["Close"].iloc[-1]
            return float(last_price), "download_1m", pd.to_datetime(last_idx)
    except Exception:
        pass
    return None, None, None


# ------------------------------
# Main dashboard logic
# ------------------------------
if selected_stock:
    raw_df = load_data(selected_stock)
    if raw_df is None:
        st.error(
            f"No processed CSV found for {selected_stock}. Run preprocessing/merge pipeline first."
        )
        st.stop()

    data = raw_df.copy()

    # ------------------------------
    # Column detection with fallbacks
    # ------------------------------
    def find_col(keywords, fallback=None):
        for col in data.columns:
            if any(k.lower() in col.lower() for k in keywords):
                return col
        return fallback

    close_col = find_col(["close"], fallback=find_col(["price"]))
    open_col = find_col(["open"], fallback="Open")
    high_col = find_col(["high"], fallback="High")
    low_col = find_col(["low"], fallback="Low")
    volume_col = find_col(["volume"], fallback="Volume")
    date_col = find_col(["date"], fallback="Date")

    # Ensure date column
    if date_col and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col])
            x_vals = data[date_col]
        except Exception:
            x_vals = data.index
    else:
        x_vals = data.index

    st.success(f"✅ Loaded {selected_stock}: {len(data)} records")

    # ------------------------------
    # Fetch latest price
    # ------------------------------
    refresh = st.sidebar.button("Refresh Live Price")
    price, src, ts = get_latest_price(selected_stock)
    if refresh:
        price, src, ts = get_latest_price(selected_stock)
    if price is not None and close_col in data.columns:
        data.loc[data.index[-1], close_col] = price

    # ------------------------------
    # Recalculate indicators
    # ------------------------------
    try:
        if close_col in data.columns:
            if HAS_TA:
                data["SMA_10"] = ta.trend.SMAIndicator(
                    close=data[close_col], window=10
                ).sma_indicator()
                data["SMA_50"] = ta.trend.SMAIndicator(
                    close=data[close_col], window=50
                ).sma_indicator()
                data["EMA_10"] = ta.trend.EMAIndicator(
                    close=data[close_col], window=10
                ).ema_indicator()
                data["EMA_50"] = ta.trend.EMAIndicator(
                    close=data[close_col], window=50
                ).ema_indicator()
                data["RSI"] = ta.momentum.RSIIndicator(
                    close=data[close_col], window=14
                ).rsi()
            else:
                data["SMA_10"] = data[close_col].rolling(10).mean()
                data["SMA_50"] = data[close_col].rolling(50).mean()
                data["EMA_10"] = data[close_col].ewm(span=10, adjust=False).mean()
                data["EMA_50"] = data[close_col].ewm(span=50, adjust=False).mean()
                data["RSI"] = compute_rsi(data[close_col], 14)
    except Exception as e:
        st.warning(f"Indicator recalculation warning: {e}")

    # ------------------------------
    # Normalize columns safely
    # ------------------------------
    for col in [open_col, high_col, low_col, close_col, volume_col]:
        if col in data.columns:
            try:
                mean = data[col].mean()
                std = data[col].std()
                data[f"{col}_Norm"] = (
                    0.0 if std == 0 or np.isnan(std) else (data[col] - mean) / std
                )
            except Exception:
                data[f"{col}_Norm"] = 0.0

    # ------------------------------
    # Metrics display
    # ------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    current_price = data[close_col].iloc[-1] if close_col in data.columns else np.nan
    prev_price = (
        data[close_col].iloc[-2]
        if close_col in data.columns and len(data) > 1
        else current_price
    )
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100 if prev_price not in (0, np.nan) else 0

    with col1:
        st.metric("Current Price (updated)", f"${current_price:.2f}")
    with col2:
        st.metric("Live Price (yfinance)", f"${price:.2f}" if price else "N/A")
    with col3:
        vol_val = (
            data[volume_col].iloc[-1]
            if volume_col in data.columns and not pd.isna(data[volume_col].iloc[-1])
            else 0
        )
        st.metric("Volume", f"{int(vol_val):,}")
    with col4:
        rsi_val = data["RSI"].iloc[-1] if "RSI" in data.columns else np.nan
        st.metric("RSI", f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "N/A")
    with col5:
        if "sentiment_compound" in data.columns:
            s = data["sentiment_compound"].iloc[-1]
            lbl = "Positive" if s > 0.05 else ("Negative" if s < -0.05 else "Neutral")
            st.metric("Sentiment", lbl, f"{s:.3f}")

    # ------------------------------
    # Price chart
    # ------------------------------
    st.subheader("📊 Stock Price Chart")
    fig = go.Figure()
    if all([col in data.columns for col in [open_col, high_col, low_col, close_col]]):
        fig.add_trace(
            go.Candlestick(
                x=x_vals,
                open=data[open_col],
                high=data[high_col],
                low=data[low_col],
                close=data[close_col],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        )
    for ma in ["SMA_10", "SMA_50", "EMA_10", "EMA_50"]:
        if ma in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=data[ma], mode="lines", name=ma, line=dict(width=1.5)
                )
            )
    if volume_col in data.columns:
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=data[volume_col],
                name="Volume",
                marker_color="blue",
                yaxis="y2",
                opacity=0.3,
            )
        )
    fig.update_layout(
        title=f"{selected_stock} Price Chart",
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all"),
                ]
            ),
            type="date",
            showgrid=True,
        ),
        yaxis=dict(title="Price", showgrid=True),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=True,
        ),
        hovermode="x unified",
        template="plotly_dark",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Sentiment chart
    # ------------------------------
    if "sentiment_compound" in data.columns:
        st.subheader("💬 Daily Sentiment (Compound Score)")
        fig_sent = go.Figure()
        fig_sent.add_trace(
            go.Bar(
                x=x_vals,
                y=data["sentiment_compound"],
                marker_color=data["sentiment_compound"].apply(
                    lambda v: "green" if v > 0.05 else ("red" if v < -0.05 else "gray")
                ),
            )
        )
        fig_sent.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Compound Score",
            showlegend=False,
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # ------------------------------
    # AI Prediction
    # ------------------------------
    st.subheader("🤖 AI Prediction")
    model = load_model(selected_stock)
    if model is None:
        st.warning(
            "No trained model found for this ticker. Place model file in models/ folder."
        )
    else:
        try:
            latest_row = data.iloc[-1:].copy()
            if hasattr(model, "feature_names_in_"):
                X_pred = latest_row.reindex(
                    columns=model.feature_names_in_, fill_value=0
                )
            else:
                feature_cols = [
                    c
                    for c in latest_row.columns
                    if any(k in c for k in ["SMA", "EMA", "RSI", "_Norm", "sentiment"])
                ]
                X_pred = latest_row[feature_cols].fillna(0)
            pred = model.predict(X_pred)[0]
            pred_change = pred - current_price
            pred_change_pct = (
                (pred_change / current_price * 100)
                if current_price not in (0, np.nan)
                else 0
            )
            arrow = "📈" if pred_change > 0 else ("📉" if pred_change < 0 else "➡️")
            st.success(
                f"**Tomorrow's Prediction:**\n\n{arrow} **${pred:.2f}**\n\nExpected Change: ${pred_change:.2f} ({pred_change_pct:.2f}%)\n\nCurrent: ${current_price:.2f}"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ------------------------------
    # Raw data preview
    # ------------------------------
    if st.checkbox("Show Raw Data"):
        st.subheader("📋 Data Preview")
        st.dataframe(data.tail(20))

# ------------------------------
# Sidebar footer
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.write("**Instructions:**")
st.sidebar.write("1. Select stock symbol")
st.sidebar.write("2. Click 'Refresh Live Price' to force update")
st.sidebar.write("3. View updated charts and prediction")
