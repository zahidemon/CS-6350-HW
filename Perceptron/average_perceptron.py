import numpy as np
import pandas as pd

average_perceptron_logs = open("average_perceptron_logs.txt", 'w')


def average_perceptron(x_train, y_train, x_test, y_test, iteration, learning_rate):
    num_features = len(x_train[0])
    weights = np.zeros(num_features)
    average = np.zeros(num_features)
    for _ in range(iteration):
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        for i in range(len(x_train)):
            if y_train[i] * np.dot(x_train[i], weights) <= 0:
                weights = weights + learning_rate * y_train[i] * x_train[i]
            average = average + weights
    prediction = np.sign(np.dot(x_test, average))
    return weights, np.mean(prediction != y_test)


if __name__ == "__main__":
    train_dataframe = pd.read_csv('../Data/bank-note/train.csv', header=None)
    test_dataframe = pd.read_csv('../Data/bank-note/test.csv', header=None)
    train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    train_x = train_dataframe.iloc[:, :-1].values
    train_y = train_dataframe.iloc[:, -1].values
    train_y[train_y == 0] = -1
    test_x = test_dataframe.iloc[:, :-1].values
    test_y = test_dataframe.iloc[:, -1].values
    test_y[test_y == 0] = -1

    learned_weights, test_error = average_perceptron(train_x, train_y, test_x, test_y, 10, 0.1)
    print("Learned weights", learned_weights)
    print("Test Error", test_error)

    average_perceptron_logs.write(f"Learned weights: {learned_weights}\n")
    average_perceptron_logs.write(f"Test error: {test_error}\n")
    average_perceptron_logs.close()
