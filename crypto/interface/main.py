from crypto.ml_logic.params import  DATASET_FREQ,CHUNK_SIZE,LOCAL_DATA_PATH
import numpy as np
import pandas as pd
from os import listdir

from colorama import Fore, Style

from crypto.ml_logic.data import clean_data, get_chunk, save_chunk, scaler_custom
from crypto.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model,SEQ_LEN
from crypto.ml_logic.registry import load_model,save_model,get_model_version
from crypto.ml_logic.utils import preprocess_custom
from sklearn.preprocessing import MinMaxScaler


def preprocess(pair:str="BTC-USDT"):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    """

    print("\n⭐️ Use case: preprocess")

    # Iterate on the dataset, in chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{pair}"
    destination_name = f"{pair}_processed_{DATASET_FREQ}"

    while (True):
        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(
            source_name=source_name,
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # Break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break


        # Save and append the chunk
        is_first = chunk_id == 0

        save_chunk(
            destination_name=destination_name,
            is_first=is_first,
            data=data_chunk_cleaned
        )

        chunk_id += 1

    if row_count == 0:
        print("\n✅ No new data for the preprocessing 👌")
        return None

    print(f"\n✅ Data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

def scaling(pair:str="BTC-USDT"):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    """

    print("\n⭐️ Use case: preprocess")

    # Iterate on the dataset, in chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{pair}_processed_{DATASET_FREQ}"
    destination_name = f"{pair}_scaled_{DATASET_FREQ}"

    while (True):
        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(
            source_name=source_name,
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = scaler_custom(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # Break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break


        # Save and append the chunk
        is_first = chunk_id == 0

        save_chunk(
            destination_name=destination_name,
            is_first=is_first,
            data=data_chunk_cleaned
        )

        chunk_id += 1

    if row_count == 0:
        print("\n✅ No new data for the preprocessing 👌")
        return None

    print(f"\n✅ Data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

def train(pair:str="BTC-USDT"):
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print("\n⭐️ Use case: train")

    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # # Load a validation set common to all chunks, used to early stop model training
    # data_val_processed = get_chunk(
    #     source_name=f"val_processed_{VALIDATION_DATASET_SIZE}",
    #     index=0,  # retrieve from first row
    #     chunk_size=None
    # )  # Retrieve all further data

    # if data_val_processed is None:
    #     print("\n✅ no data to train")
    #     return None

    # data_val_processed = data_val_processed.to_numpy()

    # X_val_processed = data_val_processed[:, :-1]
    # y_val = data_val_processed[:, -1]

    model = None
    # model = load_model()  # production model

    # Model params
    learning_rate = 0.01
    batch_size = 64
    patience = 3

    # Iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    while (True):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_processed_chunk = get_chunk(
            source_name=f"{pair}_scaled_{DATASET_FREQ}",
            index=chunk_id * CHUNK_SIZE,
            chunk_size=CHUNK_SIZE
        )

        # Check whether data source contain more data
        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        data_processed_chunk = data_processed_chunk.to_numpy()

        data_processed_chunk = data_processed_chunk[:,[0,-1]]

        X_train_chunk, y_train_chunk, _, _ = preprocess_custom(data_processed_chunk[:,-1].reshape(-1, 1), SEQ_LEN, train_split = 0.95)

        # Increment trained row count
        chunk_row_count = data_processed_chunk.shape[0]
        row_count += chunk_row_count

        # Initialize model
        if model is None:
            model = initialize_model(X_train_chunk)

        # (Re-)compile and train the model incrementally
        model = compile_model(model, learning_rate)
        model, history = train_model(
            model,
            X_train_chunk,
            y_train_chunk,
            batch_size=batch_size,
            patience=patience,
            # validation_data=(X_val_processed, y_val)
            validation_split=0.1,
        )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)
        print(f"Chunk MAE: {round(metrics_val_chunk,2)}")

        # Check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    print(f"\n✅ trained on {row_count} rows with MAE: {round(val_mae, 2)}")

    params = dict(
        # Model parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,

        # Package behavior
        context="train",
        chunk_size=CHUNK_SIZE,

        # Data source
        training_set_size=DATASET_FREQ,
        # val_set_size=VALIDATION_DATASET_SIZE,
        row_count=row_count,
        model_version=get_model_version(pair=pair),
        # dataset_timestamp=get_dataset_timestamp(),
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(mae=val_mae),pair=pair)

    return val_mae


# def evaluate():
#     """
#     Evaluate the performance of the latest production model on new data
#     """

#     print("\n⭐️ Use case: evaluate")

#     # Load new data
#     new_data = get_chunk(
#         source_name=f"val_processed_{DATASET_SIZE}",
#         index=0,
#         chunk_size=None
#     )  # Retrieve all further data

#     if new_data is None:
#         print("\n✅ No data to evaluate")
#         return None

#     new_data = new_data.to_numpy()

#     X_new = new_data[:, :-1]
#     y_new = new_data[:, -1]

#     model = load_model()

#     metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
#     mae = metrics_dict["mae"]

#     # Save evaluation
#     params = dict(
#         dataset_timestamp=get_dataset_timestamp(),
#         model_version=get_model_version(),

#         # Package behavior
#         context="evaluate",

#         # Data source
#         training_set_size=DATASET_SIZE,
#         val_set_size=VALIDATION_DATASET_SIZE,
#         row_count=len(X_new)
#     )

#     save_model(params=params, metrics=dict(mae=mae))

#     return mae


def pred(pair:str="BTC-USDT") -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # Iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []



    print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

    data_processed_chunk = get_chunk(
        source_name=f"{pair}_scaled_{DATASET_FREQ}",
        index=chunk_id * CHUNK_SIZE,
        chunk_size=CHUNK_SIZE
    )


    data_processed_chunk = data_processed_chunk.to_numpy()

    scaler = MinMaxScaler()
    training_scaler = data_processed_chunk[:,1].reshape(-1, 1)
    scaler.fit(training_scaler)


    data_processed_chunk = data_processed_chunk[:,[0,-1]]
    _, _, X_test, y_test = preprocess_custom(data_processed_chunk[:,-1].reshape(-1, 1), SEQ_LEN, train_split = 0.95)
    X_test[-1,:-1,:] = X_test[-1,1:,:]
    X_test[-1,-1,:] = y_test[-1]
    X_test = X_test[None,-1,:,:]

    model = load_model(pair=pair)

    y_pred = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape,pair)

    return y_pred


if __name__ == '__main__':
    # files = [f for f in listdir(LOCAL_DATA_PATH+"/raw") if ".csv" in f]
    # files = [f for f in files if "USDT" in f]

    pairs = ["BTC-USDT","MATIC-USDT","DOGE-USDT",
             "ATOM-USDT","ETH-USDT","BNB-USDT","ADA-USDT","LTC-USDT","UNI-USDT"]
    for pair in pairs:
        preprocess(pair)
        scaling(pair)
        train(pair)
        pred(pair=pair)

    # train()
    # pred()
    # evaluate()





# import os

# from crypto.tweeter_scripts.data import get_data
# from crypto.tweeter_scripts.clean import (clean, apply)
# from crypto.tweeter_scripts.sentiment import compute_vader_scores
# from crypto.tweeter_scripts.model import (f1_score, initialize_model, compile, train)
# from crypto.tweeter_scripts.tokenizer import tokenize_pad_sequences

# VALIDATION_DATASET_SIZE = "10k"  # ["1k", "10k", "100k", "500k"]
# CHUNK_SIZE = 2000
# DATASET_SIZE = "10k"
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "noellekh", "crypto", "raw_data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "noellekh", "crypto", "models")

# def preprocess_and_train():
#     data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
#     data = pd.read_csv(data_raw_path)

#     cleaned_data = clean (data)
#     new_df = cleaned_data["text"].apply(cleaned_data)

#     df_sentiment = compute_vader_scores(new_df, "clean_text")

#     class0=[]
#     for i in range(len(df_sentiment)):
#         if df_sentiment.loc[i,'vader_neg']>0:
#             class0+=["neg"]
#         elif df_sentiment.loc[i,'vader_pos']>0:
#             class0+=["pos"]
#         else:
#             class0+=["else"]
#     df_sentiment['class']=class0
#     #df_sentiment['class'].value_counts()
#     y = pd.get_dummies(new_df['class'])
#     X, tokenizer = tokenize_pad_sequences(new_df['clean_text'])
#     vocab_size = 5000
#     embedding_size = 32
#     epochs = 10
#     learning_rate = 0.1
#     decay_rate = learning_rate / epochs
#     momentum = 0.8
#     max_len=50

#     model = initialize_model(X)
#     model = compile(model)
#     model, history = train(model, X,y,
#                             batch_size=64,
#                             epochs = 10,
#                             verbose = 1,
#                             validation_split=0.25,
#                             validation_data=None
#                            )
#     print("yes")

# def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

#     # model = load_model()
#     # X_pred_peprocessed = tokenize_pad_sequences(X_pred)

#     # y_pred = model.predict(X_pred)
#     # return y_pred
#     pass