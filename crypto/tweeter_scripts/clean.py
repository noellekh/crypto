import pandas as pd
from nltk.corpus import stopwords
import string
import unidecode
from nltk.stem.porter import *

def clean (X: pd.DataFrame) -> pd.DataFrame:
    for punctuation in string.punctuation:
        X = X.replace(punctuation, ' ') # Remove Punctuation

    lowercased = X.lower() # Lower Case

    unaccented_string = unidecode.unidecode(lowercased) # remove accents
    words = unaccented_string.split()

    #tokenized = word_tokenize(unaccented_string) # Tokenize

    words_only = [word for word in words if word.isalpha()] # Remove numbers

    stop_words = set(stopwords.words('english')) # Make stopword list

    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    word = [PorterStemmer().stem(w) for w in without_stopwords]
    # print(words)

    return " ".join(word)
def apply (X: pd.DataFrame) -> pd.DataFrame:
    X["clean_text"] = X["text"].apply(clean)
