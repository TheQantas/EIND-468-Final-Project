from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(do_td: bool, do_off: bool, train_perc: float):
    infix = "off" if do_off else "def"
    score = "touchdown" if do_td else "field_goal"
    data = pd.read_csv(f"../series/{score}_perc_{infix}.csv")
    X = data.iloc[:, :-1].values   # Shape (x, 16)
    y = data.iloc[:, -1].values      # Shape (x,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_perc, random_state=42)   
    return (X_train, X_test, y_train, y_test)

def rnn_reshape(x):
    return x.reshape((len(x), 16, 1))
