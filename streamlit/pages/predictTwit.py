import streamlit as st
import pandas as pd
import numpy as np
import requests
import tweepy as tw
import streamlit.components.v1 as components
from transformers import pipeline
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from nltk.stem.porter import *

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

st.title('Predictions')


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
