import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DecisionTree.utils.data import apply_thresholding


class RandomDecisionTree:
    def __init__(self, data, attributes, labels, max_depth, subset_size, criterion='entropy'):
        self.data = data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.criteria = criterion
        self.subset_size = subset_size
        self.tree = self.build_tree(data, attributes, labels)

    def build_tree(self, data: pd.DataFrame, attributes: list, labels: pd.Series, depth=0):
        if len(np.unique(labels)) == 1:
            return labels.iloc[0]

        if len(attributes) == 0:
            return np.unique(labels).tolist()[0]

        if depth == self.max_depth:
            return np.unique(labels).tolist()[0]

        samples = data.sample(n=self.subset_size, replace=True)

        best_attribute = self.choose_attribute(samples, samples['y'], attributes)
        tree = {best_attribute: {}}  # Create a root node

        for value in set(data[best_attribute]):
            new_data = data[data[best_attribute] == value]
            new_label = labels[data[best_attribute] == value]
            new_attributes = list(attributes[:])
            new_attributes.remove(best_attribute)
            subtree = self.build_tree(new_data, new_attributes, new_label, depth + 1)
            tree[best_attribute][value] = subtree

        return tree

    def choose_attribute(self, data: pd.DataFrame, labels: pd.Series, attributes: list):
        gains = []

        for attribute in attributes:
            gains.append(self.information_gain(data, labels, attribute))
        return attributes[gains.index(max(gains))]

    def information_gain(self, data: pd.DataFrame, labels: pd.Series, attribute: str):
        if self.criteria == 'entropy':
            first_term = self.entropy(labels)
        elif self.criteria == 'gini':
            first_term = self.gini_index(labels)
        else:
            first_term = self.majority_error(labels)

        values, counts = np.unique(data[attribute], return_counts=True)
        second_term = 0

        for value, count_val in zip(values, counts):
            if self.criteria == 'entropy':
                second_term += (count_val / len(data)) * self.entropy(labels[data[attribute] == value])
            elif self.criteria == 'gini':
                second_term += (count_val / len(data)) * self.gini_index(labels[data[attribute] == value])
            else:
                second_term += (count_val / len(data)) * self.majority_error(labels[data[attribute] == value])

        return first_term - second_term

    @staticmethod
    def entropy(label: pd.Series):
        _, counts = np.unique(label, return_counts=True)
        entropy = 0

        for count in counts:
            entropy += (-count / len(label)) * np.log2(count / len(label))

        return entropy

    @staticmethod
    def gini_index(label):
        _, counts = np.unique(label, return_counts=True)
        gini = 1

        for count in counts:
            gini -= (count / len(label)) ** 2

        return gini

    @staticmethod
    def majority_error(label):
        _, counts = np.unique(label, return_counts=True)
        majority_error = 1 - max(counts) / len(label)

        return majority_error

    def predict(self, row):
        node = self.tree  # Start at the root node

        while isinstance(node, dict):
            attribute = list(node.keys())[0]
            attribute_value = row[attribute]

            if attribute_value not in node[attribute].keys():
                return None
            node = node[attribute][attribute_value]  # Move to the next node

        return node

    def predictions(self, data):
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        predictions = self.predictions(data)
        actual = data[label]
        return np.mean(predictions != actual)

    def training_error(self, label: str):
        return self.evaluate(self.data, label)


class RandomForest:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, num_trees, subset_size,
                 criterion='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = min(len(attributes), max_depth)
        self.subset_size = subset_size
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = []
        self.train_error, self.test_error = self.build_trees()

    def build_trees(self):
        train_error = []
        test_error = []
        sample_subset_size = len(self.train_data)

        for _ in range(self.num_trees):
            # get samples uniformly with replacement
            samples = self.train_data.sample(n=sample_subset_size, replace=True)
            # build a decision tree
            tree = RandomDecisionTree(samples, self.attributes, samples['y'], self.max_depth, self.subset_size,
                                      self.criteria)
            # add the tree to the forest
            self.trees.append(tree)
            # calculate the training error
            train_error.append(self.evaluate(self.train_data, 'y'))
            # calculate the test error
            test_error.append(self.evaluate(self.test_data, 'y'))

        return train_error, test_error

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def predict(self, row):
        predictions = [tree.predict(row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def evaluate(self, data: pd.DataFrame, label: str):
        predictions = self.predictions(data)
        actual = data[label]
        return np.mean(predictions != actual)


if __name__ == "__main__":
    train_data = "../Data/Bank/train.csv"
    test_data = "../Data/Bank/test.csv"

    bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                         'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

    bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    train_df = pd.read_csv(train_data, names=bank_column_names)
    test_df = pd.read_csv(test_data, names=bank_column_names)

    train_df['y'] = train_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
    test_df['y'] = test_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

    train_numerical_thresholds = train_df[bank_numerical_columns].median()
    test_numerical_thresholds = test_df[bank_numerical_columns].median()

    preprocessed_bank_train_df = apply_thresholding(train_df, train_numerical_thresholds, bank_numerical_columns)
    preprocessed_bank_test_df = apply_thresholding(test_df, test_numerical_thresholds, bank_numerical_columns)

    print("Bank Dataset Evaluation (with unknown considered as value):")
    T = 500
    x = range(1, T + 1)

    rf_2 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                        list(preprocessed_bank_train_df.columns[:-1]),
                        preprocessed_bank_train_df['y'], 16, T, 2)

    fig1 = plt.figure(1)
    ax2 = plt.axes()
    ax2.plot(x, rf_2.train_error, c='b', label='Train Error')
    ax2.plot(x, rf_2.test_error, c='r', label='Test Error')
    ax2.set_title("Random Forest, Feature Subset Size = 2")
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error Rate', fontsize=16)
    plt.legend(['train', 'test'])
    plt.savefig("rf2.png")
    plt.show()

    rf_4 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                        list(preprocessed_bank_train_df.columns[:-1]),
                        preprocessed_bank_train_df['y'], 16, T, 4)

    fig2 = plt.figure(2)
    ax3 = plt.axes()
    ax3.plot(x, rf_4.train_error, c='b', label='Train Error')
    ax3.plot(x, rf_4.test_error, c='r', label='Test Error')
    ax3.set_title("Random Forest, Feature Subset Size = 4")
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error Rate', fontsize=16)
    plt.legend(['train', 'test'])
    plt.savefig("")
    plt.show()

    rf_6 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                        list(preprocessed_bank_train_df.columns[:-1]),
                        preprocessed_bank_train_df['y'], 16, T, 6)

    fig3 = plt.figure(3)
    ax4 = plt.axes()
    ax4.plot(x, rf_6.train_error, c='b', label='Train Error')
    ax4.plot(x, rf_6.test_error, c='r', label='Test Error')
    ax4.set_title("Random Forest, Feature Subset Size = 6")
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Error Rate', fontsize=16)
    plt.legend(['train', 'test'])
    plt.savefig("rf6.png")
    plt.show()
