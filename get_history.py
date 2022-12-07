from binance import Client
import os
import pandas as pd

LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")



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

freq = ["1d","12h","6h"]

api_key=os.environ.get("BINANCE_KEY")
api_secret=os.environ.get("BINANCE_SECRET")

client = Client(api_key, api_secret)



for asset in pairs:
    for frequence in freq:
        print(f"saving {asset} @ freq {frequence}...")
        df = getminutedata(asset, frequence, '2000d').reset_index()
        df = df[["open_time","close"]]
        df = df[df["open_time"]>"2019"].reset_index(drop=True)
        df.to_csv(LOCAL_DATA_PATH+"/processed/"+"-".join([asset[:-4],asset[-4:]])+f"_processed_{frequence}.csv",index=False)
        print(f"saving {asset} @ freq {frequence} is DONE")
