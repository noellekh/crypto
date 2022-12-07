from binance import Client
import os
import pandas as pd

LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
DATASET_FREQ  = os.environ.get("DATASET_FREQ")


def getminutedata(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback))
    frame = frame.iloc[:,:5]
    frame.columns = ['open_time', 'Open', 'High', 'Low', 'close']
    frame = frame.set_index('open_time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame

pairs = ["BTCUSDT","MATICUSDT","DOGEUSDT",
             "ATOMUSDT","ETHUSDT","BNBUSDT","ADAUSDT","LTCUSDT","UNIUSDT"]

api_key="lQWIgmYNBdkjEgBnyMnT6xtCOeBn7hZgeAcstl0FeVJ8YufJqGeLNkXf5GZiMCfD"
api_secret="jmVds0lScfUAC6mvKEJ7ozqb95ZLnQNPB3UADegQgOyuSbun1gNQqp7JfNGAmUOO"

client = Client(api_key, api_secret)



for asset in pairs:
    df = getminutedata(asset, '1d', '2000d').reset_index()
    df = df[["open_time","close"]]
    df = df[df["open_time"]>"2019"].reset_index(drop=True)
    df.to_csv(LOCAL_DATA_PATH+"/processed/"+"-".join([asset[:-4],asset[-4:]])+f"_processed_{DATASET_FREQ}.csv",index=False)
