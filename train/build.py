import sys
import numpy as np

from train_test_split import split_data

from lsr import lsr_build
from rnn import rnn_build
from mpl import mpl_build
from rstn import rstn_build
from transformer import transformer_build

if __name__ == "__main__":
    [_, model_name, score, side] = sys.argv

    assert(score == "td" or score == "fg")
    assert(side == "off" or side == "def")

    X_train, X_test, y_train, y_test = split_data(score=="td", side == "off", 0.8)

    if model_name == "lsr":
        (model, predict_train, predict_test) = lsr_build(X_train, X_test, y_train)
    elif model_name == "rnn":
        (model, predict_train, predict_test) = rnn_build(X_train, X_test, y_train)
    elif model_name == "rstn":
        (model, predict_train, predict_test) = rstn_build(X_train, X_test, y_train)
    elif model_name == "mpl":
        (model, predict_train, predict_test) = mpl_build(X_train, X_test, y_train)
    elif model_name == "trans":
        (model, predict_train, predict_test) = transformer_build(X_train, X_test, y_train)
    else:
        raise Exception(f"Unknown model {model_name}")
    
    train_mad = np.mean(np.abs(predict_train - y_train))
    test_mad = np.mean(np.abs(predict_test - y_test))

    print(f"Train MAD of {model_name} of {train_mad:.4f}")
    print(f"Test MAD of {model_name} of {test_mad:.4f}")




