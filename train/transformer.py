import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention # type: ignore
from tensorflow.keras.models import Model # type: ignore

def __positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Transformer Block
def __transformer_block(inputs, d_model, num_heads):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    dense_output = Dense(d_model, activation='relu')(attn_output)
    dense_output = LayerNormalization(epsilon=1e-6)(attn_output + dense_output)
    return dense_output

def transformer_build(X_train, X_test, y_train) -> tuple[Model, np.ndarray, np.ndarray]:
    seq_len = 16
    d_model = 64
    num_heads = 8

    X_train_transformer = X_train.reshape((-1, seq_len, 1))
    X_test_transformer = X_test.reshape((-1, seq_len, 1))

    inputs = Input(shape=(seq_len, 1))
    x = Dense(d_model)(inputs)  # Project to d_model dimensions
    x += __positional_encoding(seq_len, d_model)
    x = __transformer_block(x, d_model, num_heads)
    x = Dense(1)(x[:, -1, :])  # Output prediction for the last time step
    model = Model(inputs, x)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train_transformer, y_train, validation_split=0.2, epochs=50, batch_size=32)

    predict_train = model.predict(X_train_transformer, verbose=0)
    predict_test = model.predict(X_test_transformer, verbose=0)

    return (model, predict_train, predict_test)

