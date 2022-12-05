from crypto.ml_logic.params import  COLUMN_NAMES_RAW,SLICING_TIME,DTYPES_RAW_OPTIMIZED,DTYPES_RAW_OPTIMIZED_HEADLESS,DTYPES_PROCESSED_OPTIMIZED,DTYPES_PROCESSED_OPTIMIZED_HEADLESS,COLUMN_NAMES_PROCESSED
import os
import pandas as pd

from crypto.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)
from crypto.data_sources.big_query import (get_bq_chunk, save_bq_chunk)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """

    df = df[["open_time","close"]]
    df = df.iloc[::SLICING_TIME,:]


    print("\n✅ data cleaned")

    return df


def get_chunk(source_name: str,
              index: int = 0,
              chunk_size: int = None,
              verbose=False) -> pd.DataFrame:
    """
    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
    Always assumes `source_name` (CSV or Big Query table) have headers,
    and do not consider them as part of the data `index` count.
    """

    if "processed" in source_name:
        columns =COLUMN_NAMES_PROCESSED
        dtypes = DTYPES_PROCESSED_OPTIMIZED
    else:
        columns = COLUMN_NAMES_RAW
        if os.environ.get("DATA_SOURCE") == "bigquery":
            dtypes = DTYPES_RAW_OPTIMIZED
        else:
            dtypes = DTYPES_RAW_OPTIMIZED_HEADLESS

    if os.environ.get("DATA_SOURCE") == "bigquery":

        chunk_df = get_bq_chunk(table=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                verbose=verbose)

        return chunk_df

    chunk_df = get_pandas_chunk(path=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                columns=columns,
                                verbose=verbose)

    return chunk_df


def save_chunk(destination_name: str,
               is_first: bool,
               data: pd.DataFrame) -> None:
    """
    save chunk
    """

    if os.environ.get("DATA_SOURCE") == "bigquery":

        save_bq_chunk(table=destination_name,
                      data=data,
                      is_first=is_first)

        return

    save_local_chunk(path=destination_name,
                     data=data,
                     is_first=is_first)
