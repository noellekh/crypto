
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

max_words = 5000
max_len=50

def tokenize_pad_sequences(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''

    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(df)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(df)
    # Pad sequences to the same length
    X = pad_sequences(df, padding='post', maxlen=max_len)
    # return sequences

    return X, tokenizer
