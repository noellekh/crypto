from crypto.ml_logic.params import LOCAL_DATA_PATH
import pandas as pd
from os import listdir


files = [f for f in listdir(LOCAL_DATA_PATH+"/raw") if ".parquet" in f]
# print(LOCAL_DATA_PATH)
# print(files)

# file = "BTC-USDT.parquet"
# df = pd.read_parquet(LOCAL_DATA_PATH+"/raw/"+file).reset_index()
# df=df[df["open_time"]>"2019"].reset_index(drop=True)
# df.to_csv(LOCAL_DATA_PATH+"/raw/"+file[:-8]+".csv",index=False)
for file in files:
    df = pd.read_parquet(LOCAL_DATA_PATH+"/raw/"+file).reset_index()
    df=df[df["open_time"]>"2019"].reset_index(drop=True)
    df.to_csv(LOCAL_DATA_PATH+"/raw/"+file[:-8]+".csv",index=False)
