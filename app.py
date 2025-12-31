import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline

# ================= PAGE LAYOUT =================
st.set_page_config(page_title="MarketMind AI", layout="wide")
st.title("🤖 MarketMind: Hybrid AI Stock Advisor")
st.markdown("### Merging LSTM Price Prediction with News Sentiment (BERT)")

# Sidebar for user input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
days_back = st.sidebar.slider("Days for Prediction", 30, 90, 60)

# ================= FUNCTION 1: FETCH DATA =================
@st.cache_data
def get_data(symbol, days):
    try:
        # Download extra data to ensure we have enough for the lookback
        df = yf.download(symbol, period='2y', progress=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ================= FUNCTION 2: TRAIN LSTM (Live) =================
# Note: In a real app, you would load a pre-trained model. 
# Here we train on the fly to keep it self-contained.
@st.cache_resource
def train_lstm(df):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data
    train_len = int(len(scaled_data) * 0.8)
    lookback = 60
    
    x_train, y_train = [], []
    for i in range(lookback, train_len):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train (Fast for demo)
    model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=0)
    
    return model, scaler

# ================= FUNCTION 3: GET SENTIMENT =================
@st.cache_resource
def get_sentiment_model():
    # Added framework="pt" to force PyTorch and avoid the TensorFlow crash
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")

# ================= MAIN LOGIC =================
if st.button("Analyze Stock"):
    with st.spinner('Fetching data and training AI models...'):
        # 1. Get Data
        df = get_data(stock_symbol, days_back)
        
        if df is not None:
            # Display Chart
            st.subheader(f"Price History: {stock_symbol}")
            st.line_chart(df['Close'])

            # 2. LSTM Prediction
            model, scaler = train_lstm(df)
            
            # Prep last 60 days for prediction
            last_60 = df[['Close']].tail(60).values
            last_60_scaled = scaler.transform(last_60)
            X_input = np.reshape(last_60_scaled, (1, 60, 1))
            
            pred_scaled = model.predict(X_input)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            current_price = df['Close'].iloc[-1].item()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Predicted Price (Next Day)", f"${pred_price:.2f}")
            price_change = ((pred_price - current_price) / current_price) * 100
            col3.metric("Expected Change", f"{price_change:.2f}%")

            # 3. Sentiment Analysis
            st.markdown("---")
            st.subheader("📰 News Sentiment Analysis")
            finbert = get_sentiment_model()
            
            news = yf.Ticker(stock_symbol).news
            headlines = [i.get('title', i.get('content', {}).get('title')) for i in news if i]
            
            if headlines:
                # Limit to top 5 for speed
                sentiment_scores = finbert(headlines[:5])
                
                # Calculate Average Score
                score_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                total_score = sum([score_map[s['label']] * s['score'] for s in sentiment_scores])
                avg_score = total_score / len(sentiment_scores)
                
                st.write(f"**Analyzed {len(sentiment_scores)} recent headlines.**")
                st.progress((avg_score + 1) / 2) # Normalize -1..1 to 0..1 for progress bar
                
                if avg_score > 0.2:
                    st.success(f"Market Sentiment: POSITIVE ({avg_score:.2f})")
                elif avg_score < -0.2:
                    st.error(f"Market Sentiment: NEGATIVE ({avg_score:.2f})")
                else:
                    st.info(f"Market Sentiment: NEUTRAL ({avg_score:.2f})")

                # Show headlines
                with st.expander("Read Analyzed Headlines"):
                    for i, h in enumerate(headlines[:5]):
                        st.write(f"- {h} ({sentiment_scores[i]['label']})")

                # 4. FINAL RECOMMENDATION
                st.markdown("### 🔮 Final Trade Signal")
                
                signal = "HOLD"
                color = "orange"
                
                if price_change > 0.5 and avg_score > 0.2:
                    signal = "STRONG BUY"
                    color = "green"
                elif price_change < -0.5 and avg_score < -0.2:
                    signal = "STRONG SELL"
                    color = "red"
                
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>{signal}</h2>", unsafe_allow_html=True)
                
            else:
                st.warning("No news found for sentiment analysis.")
