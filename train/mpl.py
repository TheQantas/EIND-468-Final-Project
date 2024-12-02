import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

def mpl_build(X_train, X_test, y_train) -> tuple[Sequential, np.ndarray, np.ndarray]:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(16,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

    predict_train = model.predict(X_train, verbose=0)
    predict_test = model.predict(X_test, verbose=0)

    return (model, predict_train, predict_test)
