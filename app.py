import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Algo Trading System", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main {background-color:#020617;}
h1,h2,h3 {color:white;}
.metric {
    background:#020617;
    padding:18px;
    border-radius:16px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Algorithmic Trading & Research System")

# ---------------- SIDEBAR ----------------
stock = st.sidebar.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS"]
)

# ---------------- LOAD DATA ----------------
data = yf.download(stock, period="1y", auto_adjust=True)

# 🔑 FIX: FLATTEN MULTIINDEX COLUMNS
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.dropna(inplace=True)

# ---------------- INDICATORS ----------------
data["SMA20"] = data["Close"].rolling(20).mean()
data["EMA20"] = data["Close"].ewm(span=20).mean()

delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
data["RSI"] = 100 - (100 / (1 + rs))

# ---------------- SIGNALS ----------------
data["Signal"] = 0
data.loc[(data["RSI"] < 30) & (data["SMA20"] > data["EMA20"]), "Signal"] = 1
data.loc[(data["RSI"] > 70) & (data["EMA20"] > data["SMA20"]), "Signal"] = -1

# ---------------- RETURNS ----------------
data["Return"] = data["Close"].pct_change().fillna(0)
data["Strategy Return"] = data["Return"] * data["Signal"].shift(1).fillna(0)

data["Market Curve"] = (1 + data["Return"]).cumprod()
data["Strategy Curve"] = (1 + data["Strategy Return"]).cumprod()

# ---------------- RISK METRICS ----------------
risk_free = 0.05 / 252
sharpe = (
    (data["Strategy Return"].mean() - risk_free)
    / data["Strategy Return"].std()
) * np.sqrt(252)

data["Peak"] = data["Strategy Curve"].cummax()
data["Drawdown"] = (data["Strategy Curve"] - data["Peak"]) / data["Peak"]
max_drawdown = data["Drawdown"].min() * 100

# ---------------- MOCK BROKER ----------------
class MockBroker:
    def __init__(self, balance=100000):
        self.balance = float(balance)
        self.position = 0
        self.trades = []

    def buy(self, price, date):
        price = float(price)
        if self.balance >= price:
            self.balance -= price
            self.position += 1
            self.trades.append([date, "BUY", price, self.balance])

    def sell(self, price, date):
        price = float(price)
        if self.position > 0:
            self.balance += price
            self.position -= 1
            self.trades.append([date, "SELL", price, self.balance])

broker = MockBroker()

for i in range(len(data)):
    signal = int(data["Signal"].iloc[i])
    price = float(data["Close"].iloc[i])
    date = data.index[i]

    if signal == 1:
        broker.buy(price, date)
    elif signal == -1:
        broker.sell(price, date)

final_value = broker.balance + broker.position * data["Close"].iloc[-1]
profit = final_value - 100000

journal = pd.DataFrame(
    broker.trades,
    columns=["Date", "Action", "Price", "Balance"]
)

# ---------------- METRICS ----------------
c1, c2, c3 = st.columns(3)

c1.metric("Market Return", f"{(data['Market Curve'].iloc[-1]-1)*100:.2f}%")
c2.metric("Strategy Return", f"{(data['Strategy Curve'].iloc[-1]-1)*100:.2f}%")
c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")
st.write(f"**Bot P/L:** ₹{profit:,.2f}")

# ---------------- CANDLESTICK ----------------
st.subheader("📊 Candlestick Chart")

fig = go.Figure()
fig.add_candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
)
fig.add_scatter(x=data.index, y=data["SMA20"], name="SMA 20")
fig.add_scatter(x=data.index, y=data["EMA20"], name="EMA 20")
fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

# ---------------- STRATEGY COMPARISON (FIXED) ----------------
st.subheader("📈 Strategy Comparison")

curve_df = data[["Market Curve", "Strategy Curve"]].copy()
st.line_chart(curve_df)

# ---------------- PROPHET PREDICTION ----------------
st.subheader("🔮 Price Prediction")

df_p = data.reset_index()[["Date", "Close"]]
df_p.columns = ["ds", "y"]

model = Prophet()
model.fit(df_p)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

st.line_chart(
    forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
)

actual = df_p["y"].iloc[-30:]
predicted = forecast["yhat"].iloc[-30:]

st.write(f"MAE: {mean_absolute_error(actual, predicted):.2f}")
st.write(f"R² Score: {r2_score(actual, predicted):.2f}")

# ---------------- TRADING JOURNAL ----------------
st.subheader("📘 Trading Journal")
st.dataframe(journal)

st.download_button(
    "Download Trading Journal",
    journal.to_csv(index=False),
    "trading_journal.csv"
)
