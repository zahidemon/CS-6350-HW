import numpy as np
import pandas as pd


def get_data(train_filename, test_filename, column_names):
    train_data = pd.read_csv(train_filename, names=column_names).astype(float)
    test_data = pd.read_csv(test_filename, names=column_names).astype(float)
    y_train = train_data[column_names[-1]]
    y_test = test_data[column_names[-1]]
    x_train = train_data.drop(column_names[-1], axis=1)
    x_test = test_data.drop(column_names[-1], axis=1)

    return x_train, y_train, x_test, y_test


def get_prediction(x, weights):
    return np.dot(x, weights)


def get_error(y, y_pred):
    return 0.5 * np.sum((y_pred - y) ** 2)
