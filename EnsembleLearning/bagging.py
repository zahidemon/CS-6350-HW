import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DecisionTree.id3 import DecisionTree
from DecisionTree.utils.data import get_attributes_and_labels, apply_thresholding

bagging_file = open("bagging_logs.txt", 'w')
bagging_file.write("n_of_trees\t training_error\t testing_error\n")


class Bagging:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, number_of_trees, impurity_type='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.number_of_trees = number_of_trees
        self.impurity_type = impurity_type
        self.trees = []
        self.train_error, self.test_error = self.build_trees()

    def build_trees(self):
        train_error = []
        test_error = []
        subset_size = len(self.train_data)

        for i in range(self.number_of_trees):
            bootstrap = self.train_data.sample(n=subset_size, replace=True)
            tree = DecisionTree(bootstrap, self.attributes, bootstrap['y'], self.max_depth, self.impurity_type)
            self.trees.append(tree)
            training_error = self.evaluate(self.train_data, 'y')
            testing_error = self.evaluate(self.test_data, 'y')
            train_error.append(training_error)
            test_error.append(testing_error)
            bagging_file.write(f"{i}\t {training_error}\t {testing_error}\n")
        return train_error, test_error

    def predict(self, row):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(row))
        return max(set(predictions), key=predictions.count)

    def predictions(self, data):
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        predictions = self.predictions(data)
        actual = data[label]
        return np.mean(predictions != actual)


if __name__ == "__main__":
    train_filename = "../Data/Bank/train.csv"
    test_filename = "../Data/Bank/test.csv"

    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
               'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    all_train, x_train, y_train = get_attributes_and_labels(filename=train_filename, columns=columns)
    all_test, x_test, y_test = get_attributes_and_labels(filename=test_filename, columns=columns)

    all_train['y'] = all_train['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
    all_test['y'] = all_test['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
    y_train = y_train.apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
    y_test = y_test.apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

    non_categorical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    all_train = apply_thresholding(
        all_train,
        threshold=all_train[non_categorical_columns].median(),
        columns=non_categorical_columns
    )
    all_test = apply_thresholding(
        all_test,
        threshold=all_test[non_categorical_columns].median(),
        columns=non_categorical_columns
    )

    number_of_trees = 500
    bgt = Bagging(all_train, all_test, columns[-1], y_train, 100, number_of_trees)

    fig1 = plt.figure(1)
    ax1 = plt.axes()
    ax1.plot(list(range(1, number_of_trees + 1))*2, bgt.train_error, c='b', label='Training Error')
    ax1.plot(list(range(1, number_of_trees + 1))*2, bgt.test_error, c='r', label='Testing Error')
    ax1.set_title("Bagged Decision Tree")
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error Rate', fontsize=16)
    plt.legend(['train', 'test'])
    plt.savefig("bagging.png")
    plt.show()
    bagging_file.close()
