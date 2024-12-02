import numpy as np

def lsr_build(X_train, X_test, y_train) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

    y_train_pred = X_train_bias @ theta
    y_test_pred = X_test_bias @ theta

    return (theta, y_train_pred, y_test_pred)



