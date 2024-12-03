import sys
import numpy as np
from typing import Any

from resources import split_data

from lsr import lsr_build
from rnn import rnn_build
from mpl import mpl_build
from lstm import lstm_build
from transformer import transformer_build
from holt import holt_build

def evaluate(model_name: str, X_train, X_test, y_train) -> tuple[Any, np.ndarray, np.ndarray]:
    model_name = model_name.lower()

    if model_name == "lsr":
        return lsr_build(X_train, X_test, y_train)
    elif model_name == "rnn":
        return rnn_build(X_train, X_test, y_train)
    elif model_name == "lstm":
        return lstm_build(X_train, X_test, y_train)
    elif model_name == "mpl":
        return mpl_build(X_train, X_test, y_train)
    elif model_name == "trans":
        return transformer_build(X_train, X_test, y_train)
    elif model_name == "holt":
        return holt_build(X_train, X_test, y_train)
    else:
        raise Exception(f"Unknown model {model_name} (Options: lsr, rnn, lstm, mpl, trans, holt)")

if __name__ == "__main__":
    [_, model_name, score, side] = sys.argv

    assert(score == "td" or score == "fg")
    assert(side == "off" or side == "def")

    X_train, X_test, y_train, y_test = split_data(score == "td", side == "off", 0.8)
    X_train, X_test, y_train, y_test = split_data(score == "td", side == "off", 0.8)

    _, predict_train, predict_test = evaluate(model_name, X_train, X_test, y_train)
    
    train_mad = np.mean(np.abs(predict_train - y_train))
    test_mad = np.mean(np.abs(predict_test - y_test))

    print(f"Train MAE of {model_name} of \033[33m{train_mad:.4f}\033[0m")
    print(f"Test MAE of {model_name} of \033[32m{test_mad:.4f}\033[0m")
    print(f"Train MAE of {model_name} of \033[33m{train_mad:.4f}\033[0m")
    print(f"Test MAE of {model_name} of \033[32m{test_mad:.4f}\033[0m")




