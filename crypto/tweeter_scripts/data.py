import os
import pandas as pd


def get_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """

    df. pd.read_csv("../../raw_data/Bitocin_tweets.csv")

    # remove useless/redundant columns
    df = df[["user_name", "date", "text"]]
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    #Keep only the verified users
    df = df[df["user_verified"]==True]

    df = df.reset_index()
    return df
