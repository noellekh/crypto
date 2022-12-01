import numpy as np
import pandas as pd
import os

from crypto.tweeter_scripts.data import get_data
from crypto.tweeter_scripts.clean import (clean, apply)
from crypto.tweeter_scripts.sentiment import compute_vader_scores
from crypto.tweeter_scripts.model import (f1_score, initialize_model, compile, train)
from crypto.tweeter_scripts.tokenizer import tokenize_pad_sequences

VALIDATION_DATASET_SIZE = "10k"  # ["1k", "10k", "100k", "500k"]
CHUNK_SIZE = 2000
DATASET_SIZE = "10k"
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "noellekh", "crypto", "raw_data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "noellekh", "crypto", "models")

def preprocess_and_train():
    data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
    data = pd.read_csv(data_raw_path)

    cleaned_data = clean (data)
    new_df = cleaned_data["text"].apply(cleaned_data)

    df_sentiment = compute_vader_scores(new_df, "clean_text")

    class0=[]
    for i in range(len(df_sentiment)):
        if df_sentiment.loc[i,'vader_neg']>0:
            class0+=["neg"]
        elif df_sentiment.loc[i,'vader_pos']>0:
            class0+=["pos"]
        else:
            class0+=["else"]
    df_sentiment['class']=class0
    #df_sentiment['class'].value_counts()
    y = pd.get_dummies(new_df['class'])
    X, tokenizer = tokenize_pad_sequences(new_df['clean_text'])
    vocab_size = 5000
    embedding_size = 32
    epochs = 10
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    max_len=50

    model = initialize_model(X)
    model = compile(model)
    model, history = train(model, X,y,
                            batch_size=64,
                            epochs = 10,
                            verbose = 1,
                            validation_split=0.25,
                            validation_data=None
                           )

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    # model = load_model()
    # X_pred_peprocessed = tokenize_pad_sequences(X_pred)

    # y_pred = model.predict(X_pred)
    # return y_pred
    pass
