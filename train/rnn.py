import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN # type: ignore

from resources import rnn_reshape

def rnn_build(X_train, X_test, y_train) -> tuple[Sequential, np.ndarray, np.ndarray]:
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=(16, 1)),
        Dense(1)
    ])

    X_train_rnn = rnn_reshape(X_train)
    X_test_rnn = rnn_reshape(X_test)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train_rnn, y_train, validation_split=0.2, epochs=50, batch_size=32)

    predict_train = model.predict(X_train_rnn, verbose=0)
    predict_test = model.predict(X_test_rnn, verbose=0)

    return (model, predict_train, predict_test)

