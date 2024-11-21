import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X = np.random.rand(2000, 16)  # Shape (2000, 16)
y = np.random.rand(2000)      # Shape (2000,)

from tensorflow.keras.layers import LSTM

# LSTM Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(16, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_rnn, y, validation_split=0.2, epochs=50, batch_size=32)
