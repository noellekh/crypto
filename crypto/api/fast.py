import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import Model, models
from crypto.ml_logic.registry import load_model,save_model,get_model_version
import numpy as np
from crypto.ml_logic.params import  DATASET_FREQ,CHUNK_SIZE
from crypto.ml_logic.data import get_chunk
from crypto.ml_logic.model import SEQ_LEN
from crypto.ml_logic.utils import preprocess_custom

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

    _, _, X_test, y_test = preprocess_custom(data_processed_chunk[:,-1].reshape(-1, 1), SEQ_LEN, train_split = 0.95)
    X_test[-1,:-1,:] = X_test[-1,1:,:]
    X_test[-1,-1,:] = y_test[-1]
    X_test = X_test[None,-1,:,:]

    res = app.state.model.predict(X_test)

    return {"prediction":float(res[0][0])
            }




@app.get("/")
def root():
    return {
    'working': 1
}
