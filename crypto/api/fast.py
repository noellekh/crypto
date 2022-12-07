import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from crypto.interface.main import pred



app = FastAPI()
# app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(pair:str="BTC-USDT",freq:str="1d"):

    return {"prediction":float(pred(pair=pair,freq=freq)[0][0])
             }


@app.get("/")
def root():
    return {
    'working': 1
}




# ---------------------------------------------------- #

# PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# MODEL_DIR = os.path.join(PACKAGE_DIR, 'raw_data')
# def load_model():
#     return models.load_model(os.path.join(MODEL_DIR, 'Lastmodel0212.h5'))


    # # Iterate on the full dataset per chunks
    # chunk_id = 0
    # row_count = 0
    # metrics_val_list = []

    # data_processed_chunk = get_chunk(
    #     source_name=f"{pair}_processed_{DATASET_FREQ}",
    #     index=chunk_id * CHUNK_SIZE,
    #     chunk_size=CHUNK_SIZE
    # )


    # data_processed_chunk = data_processed_chunk.to_numpy()

    # _, _, X_test, y_test = preprocess_custom(data_processed_chunk[:,-1].reshape(-1, 1), SEQ_LEN, train_split = 0.95)
    # X_test[-1,:-1,:] = X_test[-1,1:,:]
    # X_test[-1,-1,:] = y_test[-1]
    # X_test = X_test[None,-1,:,:]

    # res = app.state.model.predict(X_test)

    # return {"prediction":float(res[0][0])
    #         }
