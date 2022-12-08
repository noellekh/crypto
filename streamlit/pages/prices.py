import streamlit as st
import time
import numpy as np
import pandas as pd

st.set_page_config(page_title="Live Prices", page_icon="📈")

st.markdown("# Live Prices")
st.sidebar.header("Live Prices")

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

#big_title = '<p style="font-family:Courier; color:#01a781; font-size: 50px;">Live Prices</p>'
#st.markdown(big_title, unsafe_allow_html=True)
#st.title("cryPredict")
c1, c2 = st.columns([1, 8])



st.markdown(
    """# **Crypto actual prices**
From the [Binance API](https://www.binance.com/en/support/faq/360002502072).
"""
)
#st.header("**Selected Price**")

# Load market data from Binance API
df = pd.read_json("https://api.binance.com/api/v3/ticker/24hr")

crpytoList = {
    "Price 1": "BTCBUSD",
    "Price 2": "ETHBUSD",
    "Price 3": "BNBBUSD",
    "Price 4": "XRPBUSD",
    "Price 5": "ADABUSD",

}

col1, col2, col3 = st.columns(3)

for i in range(len(crpytoList.keys())):
    selected_crypto_label = list(crpytoList.keys())[i]
    selected_crypto_index = list(df.symbol).index(crpytoList[selected_crypto_label])
    selected_crypto = st.selectbox(
        selected_crypto_label, df.symbol, selected_crypto_index, key=str(i)
    )
    col_df = df[df.symbol == selected_crypto]
    col_price = (col_df.weightedAvgPrice)
    col_percent = f"{float(col_df.priceChangePercent)}%"
    if i < 3:
        with col1:
            st.metric(selected_crypto, col_price, col_percent)
    if 2 < i < 6:
        with col2:
            st.metric(selected_crypto, col_price, col_percent)