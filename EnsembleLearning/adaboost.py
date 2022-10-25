import numpy as np
from matplotlib import pyplot as plt

from DecisionTree.id3 import DecisionTree
from DecisionTree.utils.data import get_attributes_and_labels, apply_thresholding


class Adaboost:
    def __init__(self, dataframe, features, labels, number_of_trees, test_x, test_y, impurity_type='entropy'):
        self.dataframe = dataframe
        self.features = features
        self.labels = labels
        self.max_depth = 2
        self.number_of_trees = number_of_trees
        self.impurity_type = impurity_type
        self.test_x = test_x
        self.test_y = test_y
        self.stumps = []
        self.stump_training_errors = []
        self.stump_testing_errors = []
        self.build_trees()

    def build_trees(self, save_errors=True):
        weights = np.ones(len(self.dataframe)) / len(self.dataframe)
        for _ in range(self.number_of_trees):
            stump = DecisionTree(self.dataframe, self.features, self.labels, self.max_depth, self.impurity_type)
            predictions = stump.predictions(self.dataframe)
            error = np.sum(weights[predictions != self.labels])
            tree_weight = 0.5 * np.log((1 - error) / error)  # alpha_t
            tmp = predictions.apply(lambda row: 1 if row == "yes" else -1).astype(float)
            weights *= np.exp(-tree_weight * tmp)
            weights /= np.sum(weights)
            self.stumps.append((stump, tree_weight))

            if save_errors:
                self.stump_training_errors.append(stump.training_error("y"))
                self.stump_testing_errors.append(stump.evaluate(self.test_x, self.test_y))

    def predict(self, row):
        return np.sign(np.sum([tree.predict(row) * weight for tree, weight in self.stumps]))

    def predictions(self, data):
        return data.apply(self.predict, axis=1)

    def evaluate(self, data, label):
        predictions = self.predictions(data)
        return np.mean(predictions != data[label])

    def training_error(self, label: str):
        return self.evaluate(self.dataframe, label)


if __name__ == "__main__":
    train_filename = "../Data/Bank/train.csv"
    test_filename = "../Data/Bank/test.csv"
    adaboost_file = open("Adaboost_logs.txt", 'w')
    adaboost_file.write("iteration\t training_error\t testing_error\n")

    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
               'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    all_train, x_train, y_train = get_attributes_and_labels(filename=train_filename, columns=columns)
    all_test, x_test, y_test = get_attributes_and_labels(filename=test_filename, columns=columns)

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
    y_train = y_train.apply(lambda row: 1 if row == "yes" else -1).astype(float)
    all_train["y"] = y_train
    y_test = y_test.apply(lambda row: 1 if row == "yes" else -1).astype(float)
    all_test["y"] = y_test
    number_of_iterations = 500
    adaboost_training_errors = []
    adaboost_testing_errors = []
    for i in range(number_of_iterations):
        boost_classifier = Adaboost(
            dataframe=all_train,
            number_of_trees=i,
            features=x_train,
            labels=y_train,
            test_x=all_test,
            test_y=columns[-1]
        )
        training_error = boost_classifier.training_error(columns[-1])
        testing_error = boost_classifier.evaluate(all_test, columns[-1])
        adaboost_training_errors.append(training_error)
        adaboost_testing_errors.append(testing_error)
        adaboost_file.write(f"{i}\t {training_error}\t {testing_error}\n")

    fig1 = plt.figure(1)
    ax2 = plt.axes()
    ax2.plot(range(1, number_of_iterations), adaboost_training_errors, c='b', label='Train Error')
    ax2.plot(range(1, number_of_iterations), adaboost_testing_errors, c='r', label='Test Error')
    ax2.set_title("Random Forest, Feature Subset Size = 2")
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error Rate', fontsize=16)
    plt.legend(['train', 'test'])
    plt.savefig("adaboost.png")
    plt.show()