# dashboard.py - Corrected Streamlit dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="AI Stock Prediction", page_icon="📈", layout="wide")
st.title("📈 AI Stock Price Forecasting")
st.markdown("**MCA Project by Nikhil B Nair**")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Controls")
stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]
selected_stock = st.sidebar.selectbox("Select Stock", stocks)


# ------------------------------
# Load data function
# ------------------------------
@st.cache_data
def load_data(ticker):
    file_path = f"data/processed/processed_{ticker}_stock_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # Remove extra spaces
        return df
    return None


# ------------------------------
# Load model function
# ------------------------------
@st.cache_data
def load_model(ticker):
    model_path = f"models/{ticker}_better_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    model_path = f"models/{ticker}_xgboost.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return {"model": model, "metrics": {}}
    return None


# ------------------------------
# Main dashboard
# ------------------------------
if selected_stock:
    data = load_data(selected_stock)

    if data is not None and not data.empty:
        st.success(f"✅ Loaded {selected_stock}: {len(data)} records")

        # ------------------------------
        # Detect key columns dynamically
        # ------------------------------
        def find_col(keywords):
            for col in data.columns:
                if any(k.lower() in col.lower() for k in keywords):
                    return col
            return None

        close_col = find_col(["close"])
        open_col = find_col(["open"])
        high_col = find_col(["high"])
        low_col = find_col(["low"])
        volume_col = find_col(["volume"])
        date_col = find_col(["date"])

        x_vals = data[date_col] if date_col else data.index

        # ------------------------------
        # Current metrics
        # ------------------------------
        col1, col2, col3, col4 = st.columns(4)

        current_price = data[close_col].iloc[-1] if close_col else np.nan
        prev_price = (
            data[close_col].iloc[-2] if close_col and len(data) > 1 else current_price
        )
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0

        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}" if close_col else "N/A",
                f"{change:.2f} ({change_pct:.2f}%)" if close_col else "N/A",
            )
        with col2:
            st.metric(
                "Volume", f"{data[volume_col].iloc[-1]:,.0f}" if volume_col else "N/A"
            )
        with col3:
            if "RSI" in data.columns:
                st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
        with col4:
            if "sentiment_compound" in data.columns:
                sentiment = data["sentiment_compound"].iloc[-1]
                label = (
                    "Positive"
                    if sentiment > 0.05
                    else "Negative" if sentiment < -0.05 else "Neutral"
                )
                st.metric("Sentiment", label, f"{sentiment:.3f}")

        # ------------------------------
        # Price chart
        # ------------------------------
        st.subheader("📊 Stock Price Chart")
        fig = go.Figure()
        if all([open_col, high_col, low_col, close_col]):
            fig.add_trace(
                go.Candlestick(
                    x=x_vals,
                    open=data[open_col],
                    high=data[high_col],
                    low=data[low_col],
                    close=data[close_col],
                    name="Price",
                )
            )
        # Moving averages
        for ma in ["SMA_10", "SMA_50"]:
            if ma in data.columns:
                fig.add_trace(go.Scatter(x=x_vals, y=data[ma], name=ma))
        fig.update_layout(title=f"{selected_stock} Price Chart", height=500)
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
                    name="Compound",
                    marker_color="royalblue",
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
        model_data = load_model(selected_stock)

        if model_data:
            model = model_data["model"]
            metrics = model_data.get("metrics", {})

            col1, col2 = st.columns([2, 1])
            with col1:
                # Model performance metrics
                if metrics:
                    st.write("**Model Performance:**")
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        if "mae" in metrics:
                            st.metric("MAE", f"${metrics['mae']:.2f}")
                        if "rmse" in metrics:
                            st.metric("RMSE", f"${metrics['rmse']:.2f}")
                    with perf_col2:
                        if "mape" in metrics:
                            mape = metrics["mape"]
                            status = "✅" if mape <= 2.0 else "⚠️"
                            st.metric(
                                "MAPE", f"{mape:.2f}%", help=f"{status} Target: ≤2%"
                            )
                        if "r2" in metrics:
                            st.metric("R² Score", f"{metrics['r2']:.3f}")

            # Prediction
            try:
                latest_data = data.iloc[-1:].copy()
                features = []

                # Technical indicators
                for col in data.columns:
                    if any(ind in col for ind in ["SMA", "EMA", "RSI"]):
                        features.append(col)

                # Normalized price & volume
                for col in [open_col, high_col, low_col, close_col, volume_col]:
                    if col:
                        norm_col = f"{col}_Norm"
                        latest_data[norm_col] = (
                            latest_data[col] - data[col].mean()
                        ) / data[col].std()
                        features.append(norm_col)

                # Sentiment features
                for col in [
                    "sentiment_compound",
                    "sentiment_positive",
                    "sentiment_negative",
                    "sentiment_strength",
                    "positive_ratio",
                    "negative_ratio",
                ]:
                    if col in data.columns:
                        features.append(col)

                if features:
                    if hasattr(model, "feature_names_in_"):
                        X_pred = latest_data.reindex(
                            columns=model.feature_names_in_, fill_value=0
                        )
                    else:
                        X_pred = latest_data[features].fillna(0)

                    prediction = model.predict(X_pred)[0]
                    pred_change = prediction - current_price
                    pred_change_pct = (
                        (pred_change / current_price) * 100 if current_price != 0 else 0
                    )
                    direction = (
                        "📈" if pred_change > 0 else "📉" if pred_change < 0 else "➡️"
                    )

                    st.success(
                        f"**Tomorrow's Prediction:**\n\n"
                        f"{direction} **${prediction:.2f}**\n\n"
                        f"Expected Change: ${pred_change:.2f} ({pred_change_pct:.2f}%)\n\n"
                        f"Current: ${current_price:.2f}"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        else:
            st.warning(f"No model found for {selected_stock}")

        # ------------------------------
        # Data preview
        # ------------------------------
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Data Preview")
            st.dataframe(data.tail(10))

    else:
        st.error(f"No data found for {selected_stock}")

# ------------------------------
# Sidebar footer
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.write("**Instructions:**")
st.sidebar.write("1. Select stock symbol")
st.sidebar.write("2. View price charts")
st.sidebar.write("3. See AI predictions")
st.sidebar.write("4. Check model metrics")
st.sidebar.warning("⚠️ Educational use only")
