import streamlit as st
import pandas as pd
import numpy as np
import requests
import streamlit.components.v1 as components
from transformers import pipeline
import tweepy as tw
import os


####################
#   Actual prices  #
####################

st.set_page_config(page_icon="📈", page_title="cryPredict")
big_title = '<p style="font-family:Courier; color:#01a781; font-size: 80px;">cryPredict</p>'
st.markdown(big_title, unsafe_allow_html=True)
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

####################
#   Twitter #
####################
def theTweet(tweet_url):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
    response = requests.get(api)
    res = response.json()["html"]
    components.html(res,height= 700)
    return res
input = st.text_input("Enter your tweet url")
if input:
    res = theTweet(input)
    st.write(res)

#######################
#   Twitter Sentiment #
#######################

consumer_key = os.environ.get("api_key")
print(consumer_key)
consumer_secret = os.environ.get("api_key_secret")
access_token = os.environ.get("access_token")
access_token_secret = os.environ.get("access_token_secret")
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

classifier = pipeline('sentiment-analysis')
st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')

def run():
    with st.form(key="Enter name"):
        search_words = st.text_input("Enter the name for which you want to know the sentiment")
        number_of_tweets = st.number_input("Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)", 0,50,10)
        submit_button = st.form_submit_button(label="Submit")
        if submit_button:
            tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
            tweet_list = [i.text for i in tweets]
        p = [i for i in classifier(tweet_list)]
        q=[p[i]["label"] for i in range(len(p))]
        df = pd.DataFrame(list(zip(tweet_list, q)),columns =["Latest "+str(number_of_tweets)+" Tweets"+" on "+search_words, "sentiment"])
        st.write(df)


if __name__=="__main__":
    run()

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
