import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def compute_vader_scores(X: pd.DataFrame) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    idx = 0
    for sentence in X["clean_text"]:
        # print(sid.polarity_scores(sentence))
        tweets.loc[idx,"vader_neg"] = sid.polarity_scores(sentence)["neg"]
        tweets.loc[idx,"vader_neu"] = sid.polarity_scores(sentence)["neu"]
        tweets.loc[idx,"vader_pos"] = sid.polarity_scores(sentence)["pos"]
        tweets.loc[idx,"vader_comp"] = sid.polarity_scores(sentence)["compound"]
        idx+=1
    #tweets['cleantext2'] = tweets[label].apply(lambda x: unlist(x))
    return tweets
