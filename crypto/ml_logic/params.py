import os
import numpy as np

DATASET_FREQ = os.environ.get("DATASET_FREQ")
VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


COLUMN_NAMES_RAW = ["open_time","close"]
SLICING_TIME = 60*24



DTYPES_RAW_OPTIMIZED = {
    "open_time": 'O',
    "open": 'float64',
    "high": 'float64',
    "low": 'float64',
    "close": 'float64',
    "volume": 'float64',
    "quote_asset_volume": 'float64',
    "number_of_trades": 'int64',
    "taker_buy_base_asset_volume": 'float64',
    "taker_buy_quote_asset_volume": 'float64'
    }

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

# Use this to optimize loading of raw_data without headers: pd.read_csv(..., dtypes=..., headers=False)
DTYPES_RAW_OPTIMIZED_HEADLESS = {
    0: 'O',
    1: 'float64',
    2: 'float64',
    3: 'float64',
    4: 'float64',
    5: 'float64',
    6: 'float64',
    7: 'int64',
    8: 'float64',
    9: 'float64'
    }

DTYPES_PROCESSED_OPTIMIZED = {
    "open_time": "O",
    "close": "float64",
}

COLUMN_NAMES_PROCESSED = DTYPES_PROCESSED_OPTIMIZED.keys()

DTYPES_PROCESSED_OPTIMIZED_HEADLESS = {
    0: "O",
    1: "float64",

}


DTYPES_SCALED_OPTIMIZED = {
    "open_time": "O",
    "close": "float64",
    "scaled": "float64"
}

COLUMN_NAMES_SCALED = DTYPES_SCALED_OPTIMIZED.keys()

DTYPES_SCALED_OPTIMIZED_HEADLESS = {
    0: "O",
    1: "float64",
    2:"float64"
}
