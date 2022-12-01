import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def compute_vader_scores(df: pd.DataFrame) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    idx = 0
    for sentence in df["clean_text"]:
        # print(sid.polarity_scores(sentence))
        df.loc[idx,"vader_neg"] = sid.polarity_scores(sentence)["neg"]
        df.loc[idx,"vader_neu"] = sid.polarity_scores(sentence)["neu"]
        df.loc[idx,"vader_pos"] = sid.polarity_scores(sentence)["pos"]
        df.loc[idx,"vader_comp"] = sid.polarity_scores(sentence)["compound"]
        idx+=1
    #tweets['cleantext2'] = tweets[label].apply(lambda x: unlist(x))
    return df
