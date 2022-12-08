import streamlit as st
import pandas as pd
import numpy as np
import requests
import streamlit.components.v1 as components
from transformers import pipeline
import tweepy as tw
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import binance
#from binance import Client

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from nltk.stem.porter import *




#######################
#   Binance           #
#######################



# api_key = os.environ.get("BINANCE_KEY")
# api_secret = os.environ.get("BINANCE_SECRET")

# client = Client(api_key, api_secret)

# asset = 'BTCUSDT'

# def getminutedata(symbol, interval, lookback):
#     frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback))
#     frame = frame.iloc[:,:6]
#     frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
#     frame = frame.set_index('Time')
#     frame.index = pd.to_datetime(frame.index, units='ms')
#     frame = frame.astype(float)
#     return frame

# # df = getminutedata(asset, '1m', '120m')

# def animate(i):
#     data =  getminutedata(asset, '1m', '120m')
#     plt.cla()
#     plt.plot(data.index, data.Close)
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.title(asset)
#     plt.gcf().autofmt_xdate()
#     plt.tight_layout()


# fig, ax = plt.subplots()
# ani = FuncAnimation(plt.gcf(), animate, 1000)
# plt.tight_layout()
# plt.show()



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
consumer_secret = os.environ.get("api_key_secret")
access_token = os.environ.get("access_token")
access_token_secret = os.environ.get("access_token_secret")
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

classifier = pipeline('sentiment-analysis')
st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')

def clean (text):

    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation

    lowercased = text.lower() # Lower Case

    unaccented_string = unidecode.unidecode(lowercased) # remove accents
    words = unaccented_string.split()

    #tokenized = word_tokenize(unaccented_string) # Tokenize

    words_only = [word for word in words if word.isalpha()] # Remove numbers

    stop_words = set(stopwords.words('english')) # Make stopword list

    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    word = [PorterStemmer().stem(w) for w in without_stopwords]
    # print(words)

    return " ".join(word)

def run():
    with st.form(key="Enter name"):
        search_words = st.text_input("Enter the name for which you want to know the sentiment")
        search_users = st.text_input("Enter the user name for which you want to know the sentiment")
        number_of_tweets = st.number_input("Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)", 0,50,10)
        submit_button = st.form_submit_button(label="Submit")
        if submit_button:

            tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
            tweets_user = tw.Cursor(api.user_timeline, screen_name=search_users,
                              count=200, tweet_mode="extended").items((number_of_tweets))
            tweet_list = [clean(i.text) for i in tweets]
            user_list = [user.full_text for user in tweets_user]
            p = [i for i in classifier(tweet_list)]
            u = [j for j in classifier(user_list)]
            q=[p[i]["label"] for i in range(len(p))]
            v=[u[i]["label"] for i in range(len(u))]
            df = pd.DataFrame(list(zip(tweet_list, q)),columns =["Latest "+str(number_of_tweets)+" Tweets"+" on "+search_words, "sentiment"])
            df2 = pd.DataFrame(list(zip(user_list, v)),columns =["Latest "+str(number_of_tweets)+" Tweets"+" on "+search_users, "sentiment"])

        st.write(df)
        st.write(df2)


if __name__=="__main__":
    run()


