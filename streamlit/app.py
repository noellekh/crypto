import streamlit as st
import pandas as pd
import numpy as np
import requests
####################
#   Actual prices  #
####################

st.set_page_config(page_icon="📈", page_title="cryPredict")
st.image(
    "https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje",
    width=50,
)
c1, c2 = st.columns([1, 8])
with c1:
    st.image(
        "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/chart-increasing_1f4c8.png",
        width=90,
    )
st.markdown(
    """# **Crypto Dashboard**
A simple cryptocurrency price app pulling price data from the [Binance API](https://www.binance.com/en/support/faq/360002502072).
"""
)
st.header("**Selected Price**")

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
    if i > 5:
        with col3:
            st.metric(selected_crypto, col_price, col_percent)
####################
#   Prediction  #
####################

#test branche

#image backgroud
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{

             color: #01a781;
             backgroundColor: #01a781
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()


#add title

st.title('Crypto')


#add head
add_bg_from_url()
st.header("What crypto do you already have ? ")


genres =['Green', 'Yellow', 'Red', 'Blue','Orange','Noelle', 'JohnyCrypt','GuiCrypt']

with st.form(key='my_form'):
    genre=st.multiselect('Select crypto', genres)

    submit_button = st.form_submit_button(label='Submit')



#add head
add_bg_from_url()
st.header("What do you have to do with those one...")

result=st.button("Predict")
wagon_cab_api_url = 'https://cryptogcloud-wfw7tigrma-ew.a.run.app/predict'
response = requests.get(wagon_cab_api_url)

prediction = response.json()['prediction']
#pred = prediction["prediction"]
st.markdown(prediction)
st.header("list what to do Keep/Sell/Rebuy..")
##list what to do Keep/Sell/Rebuy

df = pd.DataFrame(np.random.randn(5, 3),
    columns=['past', 'today', 'future'])

st.dataframe(df)


##give shart from each crypto
#for crypto in options:
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['past', 'today', 'future'])

st.area_chart(chart_data)

st.header("The sentiment of each crypt")
###

st.write("___________________________________________________________________")

#add head
add_bg_from_url()
st.header("Want to invest in new crypto ?")


df = pd.DataFrame(np.random.randn(5, 3),
    columns=['Propostion', 'DL/FT'])

st.dataframe(df)
