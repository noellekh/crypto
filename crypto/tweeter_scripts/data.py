import pandas as pd

def get_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """
    df = df[df["user_verified"]==True]

    # remove useless/redundant columns
    df = df[["user_name", "date", "text"]]
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    #Keep only the verified users

    df = df.reset_index()
    return df

# df = pd.read_csv("../../raw_data/Bitcoin_tweets.csv")
# print(get_data(df))
