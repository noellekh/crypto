
from colorama import Fore, Style

import time
print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import LSTM,Dropout,Dense,Bidirectional,Activation
from tensorflow.keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")

from typing import Tuple

import numpy as np


def initialize_model(X: np.ndarray) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    SEQ_LEN=100

    DROPOUT = 0.2
    WINDOW_SIZE = SEQ_LEN - 1

    model = Sequential()

    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                            input_shape=(WINDOW_SIZE, X.shape[-1])))
    model.add(Dropout(rate=DROPOUT))

    # model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    # model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

    model.add(Dense(units=1))

    model.add(Activation('linear'))

    print("\n✅ model initialized")

    return model


def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("\n✅ model compiled")
    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                patience=2,
                validation_split=0.3,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=50,#50
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1,
                        shuffle=False)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history


def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    return metrics
