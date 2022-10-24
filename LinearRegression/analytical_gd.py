import numpy as np
from LinearRegression.utils import get_data, get_error, get_prediction


def optimize_analytical_gd(x, y):
    return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))


if __name__ == "__main__":
    train_filename = "../Data/Concrete/train.csv"
    test_filename = "../Data/Concrete/test.csv"
    column_names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'slump']
    x_train, y_train, x_test, y_test = get_data(train_filename, test_filename, column_names)

    weights = optimize_analytical_gd(x_train, y_train)
    y_pred = get_prediction(x_test, weights)
    test_error = get_error(y_test, y_pred)
    print(test_error)
    print(weights)
