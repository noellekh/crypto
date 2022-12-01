import pandas as pd
from nltk.corpus import stopwords
import string
import unidecode
from nltk.stem.porter import *


def clean(text: str) -> str:
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

# df = pd.read_csv("../../raw_data/Bitcoin_tweets.csv")
# df = df[df["user_verified"]==True]
# df = df[["user_name", "date", "text"]]

def apply(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_text"] = df["text"].apply(clean)
    #print(df["clean_text"])
    return None
#print(apply(df))
