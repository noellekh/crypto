"""
taxifare model package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np

DATASET_SIZE = "10k"             # ["1k","10k", "100k", "500k"]
VALIDATION_DATASET_SIZE = "10k"  # ["1k", "10k", "100k", "500k"]
CHUNK_SIZE = 2000
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

# Use this to optimize loading of raw_data with headers: pd.read_csv(..., dtypes=..., headers=True)
DTYPES_RAW_OPTIMIZED = {
    "key": "O",
    "user_name": "O",
    "date": "O",
    "text": "O",
}

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

# Use this to optimize loading of raw_data without headers: pd.read_csv(..., dtypes=..., headers=False)
DTYPES_RAW_OPTIMIZED_HEADLESS = {
    0: "O",
    1: "O",
    2: "O",
    3: "O",

}

DTYPES_PROCESSED_OPTIMIZED = np.string_



################## VALIDATIONS #################

env_valid_options = dict(
    DATASET_SIZE=["1k", "10k", "100k", "500k", "50M", "new"],
    VALIDATION_DATASET_SIZE=["1k", "10k", "100k", "500k", "500k", "new"],
    DATA_SOURCE=["local", "big query"],
    MODEL_TARGET=["local", "gcs", "mlflow"],)
