import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
st.set_page_config(page_title="AI Trading Model",layout="wide")
st.title("Nifty 50 & Sensex | AI Trading Model")
st.caption("Research only • delayed/unofficial market data • no automated orders")
symbol=st.selectbox("Index",["Nifty 50","Sensex"])
ticker={"Nifty 50":"^NSEI","Sensex":"^BSESN"}[symbol]
@st.cache_data(ttl=300)
def load(ticker):
    d=yf.download(ticker,period="5y",interval="1d",progress=False,auto_adjust=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])
try:
    d=load(ticker)
except Exception as e:
    st.error("Market data unavailable. Try again later."); st.stop()
if len(d)<250:
    st.error("Not enough market data."); st.stop()
c=d.Close
d["EMA20"]=c.ewm(span=20,adjust=False).mean()
d["EMA50"]=c.ewm(span=50,adjust=False).mean()
delta=c.diff()
g=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
l=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
d["RSI"]=100-100/(1+g/(l+1e-9))
d["MACD"]=c.ewm(span=12).mean()-c.ewm(span=26).mean()
d["ret"]=c.pct_change()
d["trend"]=c/d.EMA50-1
d["macd_rel"]=d.MACD/c
d["ema_rel"]=c/d.EMA20-1
d["ATR"]=pd.concat([d.High-d.Low,(d.High-c.shift()).abs(),(d.Low-c.shift()).abs()],axis=1).max(axis=1).rolling(14).mean()
f=["RSI","ret","trend","macd_rel","ema_rel"]
x=d.dropna(subset=f).copy()
x["next"]=x.Close.shift(-1)
train=x.dropna(subset=["next"]).copy()
y=(train.next>train.Close).astype(int)
cut=int(len(train)*.8)
m=RandomForestClassifier(n_estimators=150,max_depth=5,min_samples_leaf=12,random_state=42,n_jobs=1)
m.fit(train[f].iloc[:cut],y.iloc[:cut])
acc=accuracy_score(y.iloc[cut:],m.predict(train[f].iloc[cut:]))
m.fit(train[f],y)
p=m.predict_proba(x[f].iloc[[-1]])[0][list(m.classes_).index(1)]
last=x.iloc[-1]
signal="CE watch" if p>=.60 else "PE watch" if p<=.40 else "WAIT"
a,b,c1=st.columns(3)
a.metric("Index close",f"{last.Close:,.2f}")
b.metric("Next-session bias",signal)
c1.metric("Holdout directional accuracy",f"{acc:.1%}")
st.caption(f"Last daily candle: {x.index[-1]} | Model bullish probability: {p:.1%}. Daily candle may be incomplete during market hours.")
if signal!="WAIT":
    direction=1 if signal=="CE watch" else -1
    st.write(f"Illustrative underlying index levels: reference {last.Close:,.2f}, target {last.Close+direction*1.5*last.ATR:,.2f}, invalidation {last.Close-direction*last.ATR:,.2f}. Not option premium targets.")
v=d.tail(120)
fig=go.Figure(go.Candlestick(x=v.index,open=v.Open,high=v.High,low=v.Low,close=v.Close,name="Price"))
for col in ["EMA20","EMA50"]: fig.add_scatter(x=v.index,y=v[col],name=col)
fig.update_layout(xaxis_rangeslider_visible=False,height=450)
st.plotly_chart(fig,use_container_width=True)
st.line_chart(v[["RSI"]])
st.line_chart(v[["MACD"]])
st.warning("Daily directional research, not live tick data or a validated options strategy. No option chain, expiry, premiums, slippage or execution backtest. Paper trade first.")
