"""
Crypto Prediction UI - Streamlit Interface
A user-friendly web interface for crypto price movement prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import sys
import contextlib
from io import StringIO

warnings.filterwarnings("ignore")

# Suppress Streamlit ScriptRunContext warnings
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Lazy import to avoid ScriptRunContext warnings
# Import only when needed, inside functions
def get_crypto_module():
    """Lazy import crypto_simple_enhance to avoid ScriptRunContext warnings"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import crypto_simple_enhance as crypto
    return crypto

# Page configuration
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-up {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .prediction-down {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .prediction-neutral {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_price_ui(value):
    """Format price for UI display"""
    if value is None or pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1:
        return f"{value:,.2f}"
    elif abs_val >= 0.01:
        return f"{value:,.4f}"
    elif abs_val >= 0.0001:
        return f"{value:,.6f}"
    else:
        return f"{value:,.8f}"


def create_price_chart(df, current_price, prediction=None):
    """Create interactive price chart with prediction"""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Chart", "Volume"),
    )

    # Price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Add prediction line if available
    if prediction is not None:
        last_time = df["timestamp"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_time, last_time],
                y=[df["low"].min(), df["high"].max()],
                mode="lines",
                name="Current",
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Volume chart
    colors = ["red" if df["close"].iloc[i] < df["open"].iloc[i] else "green" for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def main():
    # Lazy import crypto module
    crypto = get_crypto_module()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Crypto Price Movement Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Symbol input
        symbol_input = st.text_input(
            "Trading Pair",
            value="BTC",
            help="Enter symbol like 'BTC', 'ETH', or 'BTC/USDT'",
        )
        
        quote_currency = st.selectbox(
            "Quote Currency",
            options=["USDT", "USD", "BTC", "ETH"],
            index=0,
        )
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            options=["30m", "45m", "1h", "2h", "4h", "6h", "12h", "1d"],
            index=2,
        )
        
        # Data limit
        limit = st.slider(
            "Number of Candles",
            min_value=500,
            max_value=3000,
            value=crypto.DEFAULT_LIMIT,
            step=100,
            help="More candles = more training data but slower processing",
        )
        
        # Exchange selection
        st.subheader("Exchanges")
        selected_exchanges = []
        for exchange in crypto.DEFAULT_EXCHANGES:
            if st.checkbox(exchange.capitalize(), value=True, key=f"exch_{exchange}"):
                selected_exchanges.append(exchange)
        
        if not selected_exchanges:
            st.warning("⚠️ Please select at least one exchange!")
            selected_exchanges = crypto.DEFAULT_EXCHANGES
        
        # Predict button
        predict_button = st.button("🚀 Predict", type="primary", use_container_width=True)

    # Main content area
    if predict_button:
        # Normalize symbol
        symbol = crypto.normalize_symbol(symbol_input, quote_currency)
        
        # Show loading
        with st.spinner(f"Fetching data for {symbol}..."):
            # Fetch data
            df, exchange_id = crypto.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                exchanges=selected_exchanges,
            )
        
        if df is None or df.empty:
            st.error("❌ Failed to fetch data. Please try again or check your internet connection.")
            return
        
        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"
        
        # Show data info
        st.success(f"✅ Data fetched from {exchange_label}")
        
        # Calculate indicators
        with st.spinner("Calculating technical indicators..."):
            df = crypto.calculate_indicators(df, apply_labels=False)
        
        # Save latest data before applying labels
        latest_data = df.iloc[-1:].copy()
        latest_data = latest_data.ffill().bfill()
        
        # Apply labels for training
        with st.spinner("Preparing training data..."):
            df = crypto.apply_directional_labels(df)
            latest_threshold = df["DynamicThreshold"].iloc[-1] if len(df) > 0 else 0.01
            df.dropna(inplace=True)
            latest_data["DynamicThreshold"] = latest_threshold
        
        # Train model
        with st.spinner(f"Training model on {len(df)} samples..."):
            # Suppress print statements during training by redirecting stdout
            @contextlib.contextmanager
            def suppress_stdout():
                with open(os.devnull, "w") as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    try:
                        yield
                    finally:
                        sys.stdout = old_stdout
            
            with suppress_stdout():
                model = crypto.train_and_predict(df)
        
        # Make prediction
        proba = crypto.predict_next_move(model, latest_data)
        proba_percent = {
            label: proba[crypto.LABEL_TO_ID[label]] * 100 for label in crypto.TARGET_LABELS
        }
        best_idx = int(np.argmax(proba))
        direction = crypto.ID_TO_LABEL[best_idx]
        probability = proba_percent[direction]
        
        # Get current market data
        current_price = latest_data["close"].values[0]
        atr = latest_data["ATR_14"].values[0]
        prediction_window = crypto.get_prediction_window(timeframe)
        threshold_value = latest_data["DynamicThreshold"].iloc[0]
        
        # Display results
        st.markdown("---")
        
        # Main prediction box
        if direction == "UP":
            box_class = "prediction-up"
            emoji = "🟢"
            color = "#28a745"
        elif direction == "DOWN":
            box_class = "prediction-down"
            emoji = "🔴"
            color = "#dc3545"
        else:
            box_class = "prediction-neutral"
            emoji = "🟡"
            color = "#ffc107"
        
        st.markdown(
            f"""
            <div class="prediction-box {box_class}">
                <h2 style="text-align: center; margin-bottom: 1rem;">
                    {emoji} Prediction: <strong>{direction}</strong>
                </h2>
                <h3 style="text-align: center; color: {color};">
                    Confidence: {probability:.2f}%
                </h3>
                <p style="text-align: center; margin-top: 1rem;">
                    Prediction Window: {prediction_window} | {crypto.TARGET_HORIZON} candles >= {threshold_value*100:.2f}% move
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", format_price_ui(current_price))
        
        with col2:
            st.metric("Market Volatility (ATR)", format_price_ui(atr))
        
        with col3:
            st.metric("Data Source", exchange_label)
        
        with col4:
            st.metric("Timeframe", timeframe)
        
        # Probability breakdown
        st.subheader("📊 Probability Breakdown")
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.metric("UP", f"{proba_percent['UP']:.2f}%")
        
        with prob_col2:
            st.metric("NEUTRAL", f"{proba_percent['NEUTRAL']:.2f}%")
        
        with prob_col3:
            st.metric("DOWN", f"{proba_percent['DOWN']:.2f}%")
        
        # Probability bar chart
        prob_df = pd.DataFrame({
            "Direction": crypto.TARGET_LABELS,
            "Probability": [proba_percent[label] for label in crypto.TARGET_LABELS],
        })
        
        fig_prob = go.Figure(
            data=[
                go.Bar(
                    x=prob_df["Direction"],
                    y=prob_df["Probability"],
                    marker_color=["#dc3545", "#ffc107", "#28a745"],
                    text=[f"{p:.2f}%" for p in prob_df["Probability"]],
                    textposition="auto",
                )
            ]
        )
        fig_prob.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Direction",
            yaxis_title="Probability (%)",
            height=300,
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Price targets if not neutral
        if direction != "NEUTRAL":
            st.subheader("🎯 Price Targets (ATR Multiples)")
            atr_sign = 1 if direction == "UP" else -1
            
            targets = []
            for multiple in [1, 2, 3]:
                target_price = current_price + atr_sign * multiple * atr
                move_abs = abs(target_price - current_price)
                move_pct = (move_abs / current_price) * 100 if current_price else 0
                targets.append({
                    "ATR Multiple": f"x{multiple}",
                    "Target Price": format_price_ui(target_price),
                    "Move": format_price_ui(move_abs),
                    "Move %": f"{move_pct:.2f}%",
                })
            
            targets_df = pd.DataFrame(targets)
            st.dataframe(targets_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                f"Market expected to stay within +/-{threshold_value*100:.2f}% over the next {crypto.TARGET_HORIZON} candles."
            )
        
        # Price chart
        st.subheader("📈 Price Chart")
        fig_chart = create_price_chart(df, current_price, direction)
        st.plotly_chart(fig_chart, use_container_width=True)
        
        # Technical indicators summary
        with st.expander("📊 Technical Indicators Summary"):
            indicators_col1, indicators_col2 = st.columns(2)
            
            with indicators_col1:
                st.write("**Trend Indicators**")
                st.write(f"- SMA 20: {format_price_ui(latest_data['SMA_20'].values[0])}")
                st.write(f"- SMA 50: {format_price_ui(latest_data['SMA_50'].values[0])}")
                st.write(f"- SMA 200: {format_price_ui(latest_data['SMA_200'].values[0])}")
                
                st.write("**Momentum Indicators**")
                st.write(f"- RSI 9: {format_price_ui(latest_data['RSI_9'].values[0])}")
                st.write(f"- RSI 14: {format_price_ui(latest_data['RSI_14'].values[0])}")
                st.write(f"- RSI 25: {format_price_ui(latest_data['RSI_25'].values[0])}")
            
            with indicators_col2:
                st.write("**Volatility Indicators**")
                st.write(f"- ATR 14: {format_price_ui(latest_data['ATR_14'].values[0])}")
                st.write(f"- Bollinger Bands %: {format_price_ui(latest_data['BBP_5_2.0'].values[0])}")
                
                st.write("**Volume Indicators**")
                st.write(f"- OBV: {format_price_ui(latest_data['OBV'].values[0])}")
        
        # Data info
        with st.expander("ℹ️ Data Information"):
            st.write(f"**Total Samples:** {len(df)}")
            st.write(f"**Training Samples:** {len(df)}")
            st.write(f"**Data Range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
            st.write(f"**Prediction Horizon:** {crypto.TARGET_HORIZON} candles")
            st.write(f"**Threshold:** {threshold_value*100:.2f}%")
    
    else:
        # Welcome message
        st.info("👈 Please configure your settings in the sidebar and click 'Predict' to start!")
        
        # Instructions
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Enter Trading Pair**: Type the symbol (e.g., BTC, ETH) or full pair (BTC/USDT)
        2. **Select Timeframe**: Choose your preferred timeframe (1h recommended)
        3. **Adjust Settings**: 
           - Increase candles for more training data (but slower)
           - Select exchanges to fetch data from
        4. **Click Predict**: The model will fetch data, train, and make predictions
        
        ### 💡 Tips
        - Use at least 1500 candles for better model performance
        - Multiple exchanges provide better data reliability
        - The model predicts price movement over the next 24 candles
        """)


if __name__ == "__main__":
    main()

