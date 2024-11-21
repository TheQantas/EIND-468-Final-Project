import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print(sys.version)

# Example data
X = np.random.rand(2000, 16)  # Shape (2000, 16)
y = np.random.rand(2000)      # Shape (2000,)

# MLP Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(16,)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X, y, validation_split=0.2, epochs=50, batch_size=32)
