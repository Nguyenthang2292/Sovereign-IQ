"""
Crypto Prediction UI - Gradio Interface
A simple and user-friendly web interface for crypto price movement prediction
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import sys
import contextlib

warnings.filterwarnings("ignore")

# Lazy import crypto module
def get_crypto_module():
    """Lazy import crypto_simple_enhance"""
    import crypto_simple_enhance as crypto
    return crypto


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


def create_price_chart(df):
    """Create interactive price chart"""
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


def predict_crypto(
    symbol_input,
    quote_currency,
    timeframe,
    limit,
    *exchange_selections
):
    """Main prediction function"""
    try:
        crypto = get_crypto_module()
        
        # Get selected exchanges
        all_exchanges = crypto.DEFAULT_EXCHANGES
        selected_exchanges = [
            all_exchanges[i] for i, selected in enumerate(exchange_selections) if selected
        ]
        
        if not selected_exchanges:
            selected_exchanges = crypto.DEFAULT_EXCHANGES
        
        # Normalize symbol
        symbol = crypto.normalize_symbol(symbol_input, quote_currency)
        
        # Fetch data
        df, exchange_id = crypto.fetch_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            exchanges=selected_exchanges,
        )
        
        if df is None or df.empty:
            return (
                "❌ Failed to fetch data. Please try again or check your internet connection.",
                None,
                None,
                None,
                None,
            )
        
        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"
        
        # Calculate indicators
        df = crypto.calculate_indicators(df, apply_labels=False)
        
        # Save latest data before applying labels
        latest_data = df.iloc[-1:].copy()
        latest_data = latest_data.ffill().bfill()
        
        # Apply labels for training
        df = crypto.apply_directional_labels(df)
        latest_threshold = df["DynamicThreshold"].iloc[-1] if len(df) > 0 else 0.01
        df.dropna(inplace=True)
        latest_data["DynamicThreshold"] = latest_threshold
        
        # Train model (suppress stdout)
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
        
        # Build result text
        result_text = f"""
# 📈 Prediction Results for {symbol}

## 🎯 Prediction: **{direction}** ({probability:.2f}%)

**Data Source:** {exchange_label}  
**Timeframe:** {timeframe}  
**Prediction Window:** {prediction_window}  
**Horizon:** {crypto.TARGET_HORIZON} candles >= {threshold_value*100:.2f}% move

### 📊 Market Data
- **Current Price:** {format_price_ui(current_price)}
- **Market Volatility (ATR):** {format_price_ui(atr)}

### 📈 Probability Breakdown
- **UP:** {proba_percent['UP']:.2f}%
- **NEUTRAL:** {proba_percent['NEUTRAL']:.2f}%
- **DOWN:** {proba_percent['DOWN']:.2f}%

### 📊 Technical Indicators
**Trend:**
- SMA 20: {format_price_ui(latest_data['SMA_20'].values[0])}
- SMA 50: {format_price_ui(latest_data['SMA_50'].values[0])}
- SMA 200: {format_price_ui(latest_data['SMA_200'].values[0])}

**Momentum:**
- RSI 9: {format_price_ui(latest_data['RSI_9'].values[0])}
- RSI 14: {format_price_ui(latest_data['RSI_14'].values[0])}
- RSI 25: {format_price_ui(latest_data['RSI_25'].values[0])}

**Volatility:**
- ATR 14: {format_price_ui(latest_data['ATR_14'].values[0])}
- Bollinger Bands %: {format_price_ui(latest_data['BBP_5_2.0'].values[0])}

**Volume:**
- OBV: {format_price_ui(latest_data['OBV'].values[0])}

### 📝 Data Information
- **Total Samples:** {len(df)}
- **Data Range:** {df['timestamp'].min()} to {df['timestamp'].max()}
"""
        
        # Price targets
        targets_text = ""
        if direction != "NEUTRAL":
            targets_text = "\n### 🎯 Price Targets (ATR Multiples)\n"
            atr_sign = 1 if direction == "UP" else -1
            for multiple in [1, 2, 3]:
                target_price = current_price + atr_sign * multiple * atr
                move_abs = abs(target_price - current_price)
                move_pct = (move_abs / current_price) * 100 if current_price else 0
                targets_text += f"- **ATR x{multiple}:** {format_price_ui(target_price)} (Move: {format_price_ui(move_abs)}, {move_pct:.2f}%)\n"
        else:
            targets_text = f"\n### ℹ️ Market Expected to Stay Within +/-{threshold_value*100:.2f}% over the next {crypto.TARGET_HORIZON} candles.\n"
        
        result_text += targets_text
        
        # Create chart
        chart = create_price_chart(df)
        
        # Create probability chart
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
        
        return result_text, chart, fig_prob, f"✅ Success! Data from {exchange_label}", None
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease check your inputs and try again."
        return error_msg, None, None, "❌ Error occurred", str(e)


def create_ui():
    """Create Gradio interface"""
    crypto = get_crypto_module()
    
    with gr.Blocks(title="Crypto Price Prediction", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📈 Crypto Price Movement Predictor
            
            Predict cryptocurrency price movements using machine learning and technical indicators.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                symbol_input = gr.Textbox(
                    label="Trading Pair",
                    value="BTC",
                    placeholder="Enter symbol like BTC, ETH, or BTC/USDT",
                    info="Enter symbol (e.g., BTC) or full pair (BTC/USDT)",
                )
                
                quote_currency = gr.Dropdown(
                    label="Quote Currency",
                    choices=["USDT", "USD", "BTC", "ETH"],
                    value="USDT",
                )
                
                timeframe = gr.Dropdown(
                    label="Timeframe",
                    choices=["30m", "45m", "1h", "2h", "4h", "6h", "12h", "1d"],
                    value="1h",
                    info="Timeframe for OHLCV data",
                )
                
                limit = gr.Slider(
                    label="Number of Candles",
                    minimum=500,
                    maximum=3000,
                    value=crypto.DEFAULT_LIMIT,
                    step=100,
                    info="More candles = more training data but slower processing",
                )
                
                gr.Markdown("### 📊 Exchanges")
                exchange_checkboxes = []
                for i, exchange in enumerate(crypto.DEFAULT_EXCHANGES):
                    checkbox = gr.Checkbox(
                        label=exchange.capitalize(),
                        value=True,
                    )
                    exchange_checkboxes.append(checkbox)
                
                predict_btn = gr.Button("🚀 Predict", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                status = gr.Textbox(
                    label="Status",
                    value="👈 Configure settings and click 'Predict' to start!",
                    interactive=False,
                )
                
                result_markdown = gr.Markdown(label="Prediction Results")
                
                with gr.Tabs():
                    with gr.Tab("📈 Price Chart"):
                        price_chart = gr.Plot(label="Price Chart")
                    
                    with gr.Tab("📊 Probability Chart"):
                        prob_chart = gr.Plot(label="Probability Chart")
                
                error_output = gr.Textbox(
                    label="Error Details (if any)",
                    visible=False,
                    interactive=False,
                )
        
        # Event handlers
        predict_btn.click(
            fn=predict_crypto,
            inputs=[symbol_input, quote_currency, timeframe, limit] + exchange_checkboxes,
            outputs=[result_markdown, price_chart, prob_chart, status, error_output],
        )
        
        gr.Markdown(
            """
            ---
            ### 📖 How to Use
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
            
            ### ⚠️ Disclaimer
            This tool is for educational purposes only. Not financial advice. Trading cryptocurrency involves risk.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    
    # Launch with localhost (127.0.0.1) instead of 0.0.0.0
    # 0.0.0.0 is for binding, not for browser access
    print("\n" + "="*60)
    print("🚀 Starting Crypto Prediction UI...")
    print("="*60)
    print(f"📱 Access the UI at: http://localhost:7860")
    print(f"   or: http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
        server_port=7860,
        share=False,
        show_error=True,
    )

