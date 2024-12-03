from statsmodels.tsa.holtwinters import Holt
import numpy as np

def holt_build(X_train: np.ndarray, X_test: np.ndarray, _y_train: np.ndarray) -> tuple[None, np.ndarray, np.ndarray]:
    predict_train = np.zeros(len(X_train))
    predict_test = np.zeros(len(X_test))
    
    for i in range(len(X_train)):
        model = Holt(X_train[i])
        model_fit = model.fit()
        seventeenth = model_fit.forecast(1)[0]
        predict_train[i] = seventeenth

    for i in range(len(X_test)):
        model = Holt(X_test[i])
        model_fit = model.fit()
        seventeenth = model_fit.forecast(1)[0]
        predict_test[i] = seventeenth

    return (None, predict_train, predict_test)
