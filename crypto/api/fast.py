import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from cry.ml_logic.registry import load_model
from tensorflow.keras import Model, models
import pandas as pd
from jh_crypto.ml_logic.registry import load_model,save_model,get_model_version
import numpy as np
from jh_crypto.ml_logic.params import  DATASET_FREQ,CHUNK_SIZE,LOCAL_DATA_PATH
from jh_crypto.ml_logic.data import clean_data, get_chunk, save_chunk


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# MODEL_DIR = os.path.join(PACKAGE_DIR, 'raw_data')

# def load_model():
#     return models.load_model(os.path.join(MODEL_DIR, 'Lastmodel0212.h5'))

app = FastAPI()
app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(pair:str="BTC-USDT"):      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    # Iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    data_processed_chunk = get_chunk(
        source_name=f"{pair}_processed_{DATASET_FREQ}",
        index=chunk_id * CHUNK_SIZE,
        chunk_size=CHUNK_SIZE
    )


    data_processed_chunk = data_processed_chunk.to_numpy()

    SEQ_LEN = 100

    def to_sequences(data, seq_len):
        d = []

        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])

        return np.array(d)

    def preprocess(data_raw, seq_len, train_split):

        data = to_sequences(data_raw, seq_len)

        num_train = int(train_split * data.shape[0])

        X_train = data[:num_train, :-1, :]
        y_train = data[:num_train, -1, :]

        X_test = data[num_train:, :-1, :]
        y_test = data[num_train:, -1, :]

        return np.asarray(X_train).astype(np.float32), np.asarray(y_train).astype(np.float32), np.asarray(X_test).astype(np.float32), np.asarray(y_test).astype(np.float32)
    _, _, X_test, y_test = preprocess(data_processed_chunk[:,-1].reshape(-1, 1), SEQ_LEN, train_split = 0.8)
    # print(X_test.shape)
    # X_test[-1,:-1,:] = X_test[-1,1:,:]
    # X_test[-1,-1,:] = y_test[-1]
    # X_test = X_test[None,-1,:,:]
    # print(X_test)



    res = app.state.model.predict(X_test)

    return {"prediction":float(res[0][0])
            }




@app.get("/")
def root():
    return {
    'working': 1
}
