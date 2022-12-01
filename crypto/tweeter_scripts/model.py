import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import History
from tensorflow.keras import losses
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple


def f1_score(precision, recall):
    ''' Function to calculate f1 score '''

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

vocab_size = 5000
embedding_size = 32
epochs = 10
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
max_len=50

def initialize_model(X):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    model= Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    return model

def compile(model):
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy', Precision(), Recall()])

    return model

def train(model,
            X: np.ndarray,
            y: np.ndarray,
            batch_size=64,
            epochs = 10,
            verbose = 1,
            validation_split=0.25,
            validation_data=None
            ):

    history = model.fit(
        X,y,
        validation_data=validation_data,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose)
    return model, history
