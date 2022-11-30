from tweeter_scripts.params import (COLUMN_NAMES_RAW,
                                            DTYPES_RAW_OPTIMIZED,
                                            DTYPES_RAW_OPTIMIZED_HEADLESS,
                                            DTYPES_PROCESSED_OPTIMIZED
                                            )

import os
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """

    # remove useless/redundant columns
    df = df.drop(columns=['key'])
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    #Keep only the verified users
    df = df[df["user_verified"]==True]

    df = df.reset_index()



    return df
