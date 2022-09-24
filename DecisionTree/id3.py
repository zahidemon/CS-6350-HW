import numpy as np
import pandas as pd
import argparse

from utils.data import get_attributes_and_labels, apply_thresholding, fill_missing_values


class DecisionTree:
    def __init__(self, dataframe, attributes, labels, max_depth, impurity_type='entropy'):
        self.dataframe = dataframe
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.impurity_type = impurity_type
        self.tree = self.build_decision_tree(dataframe, attributes, labels)

    def build_decision_tree(self, dataframe, attributes, labels, depth=0):
        if len(np.unique(labels)) == 1:
            return labels.iloc[0]

        if len(attributes) == 0:
            return np.unique(labels).tolist()[0]

        if depth == self.max_depth:
            return np.unique(labels).tolist()[0]

        best_attribute = self.choose_attribute(dataframe, labels, attributes)
        tree = {best_attribute: {}}  # Create a root node

        for value in set(dataframe[best_attribute]):
            split_dataframe = dataframe[dataframe[best_attribute] == value]
            new_label = labels[dataframe[best_attribute] == value]
            new_attributes = list(attributes[:])
            new_attributes.remove(best_attribute)
            subtree = self.build_decision_tree(split_dataframe, new_attributes, new_label, depth + 1)
            tree[best_attribute][value] = subtree

        return tree

    def choose_attribute(self, dataframe, labels, attributes):
        gains = []

        for attribute in attributes:
            gains.append(self.information_gain(dataframe, labels, attribute))

        return attributes[gains.index(max(gains))]

    def information_gain(self, dataframe, labels, attribute: str):
        first_term = 0
        if self.impurity_type == 'entropy':
            first_term = self.entropy(labels)
        elif self.impurity_type == 'gini':
            first_term = self.gini_index(labels)
        elif self.impurity_type == 'majority':
            first_term = self.majority_error(labels)

        values, counts = np.unique(dataframe[attribute], return_counts=True)
        weighted_entropy = 0
        for value, count in zip(values, counts):
            if self.impurity_type == 'entropy':
                weighted_entropy += (count / len(dataframe)) * self.entropy(labels[dataframe[attribute] == value])
            elif self.impurity_type == 'gini':
                weighted_entropy += (count / len(dataframe)) * self.gini_index(labels[dataframe[attribute] == value])
            else:
                weighted_entropy += (count / len(dataframe)) * self.majority_error(labels[dataframe[attribute] == value])

        return first_term - weighted_entropy

    @staticmethod
    def entropy(label):
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
        node = self.tree
        while isinstance(node, dict):
            attribute = list(node.keys())[0]
            attribute_value = row[attribute]
            if attribute_value not in node[attribute].keys():
                return None
            node = node[attribute][attribute_value]
        return node

    def predictions(self, dataframe):
        return dataframe.apply(self.predict, axis=1)

    def evaluate(self, dataframe, label):
        predictions = self.predictions(dataframe)
        actual = dataframe[label]
        return np.mean(predictions != actual)

    def training_error(self, label):
        return self.evaluate(self.dataframe, label)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Decision Tree Implementation')
    parser.add_argument('dataset', metavar='D', type=str, default="car",
                        help='3 choices: car, bank, bank-handling-missing-data')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    dataset = str(args.dataset)  # can be either car, bank or bank-handling-missing-data
    impurity_types = ["entropy", "gini", "majority"]
    report_columns = ['train_error(entropy)', 'test_error(entropy)', 'train_error(gini)', 'test_error(gini)',
                      'train_error(majority)', 'test_error(majority)']

    if dataset == "car":
        train_filename = "Data/Car/train.csv"
        test_filename = "Data/Car/test.csv"
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
        max_depth_size = 6
        all_train, x_train, y_train = get_attributes_and_labels(filename=train_filename, columns=columns)
        all_test, x_test, y_test = get_attributes_and_labels(filename=test_filename, columns=columns)
    else:
        train_filename = "Data/Bank/train.csv"
        test_filename = "Data/Bank/test.csv"
        columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
        max_depth_size = 16
        all_train, x_train, y_train = get_attributes_and_labels(filename=train_filename, columns=columns)
        all_test, x_test, y_test = get_attributes_and_labels(filename=test_filename, columns=columns)
        if dataset == "bank":
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
        else:
            missing_columns = ['job', 'education', 'contact', 'poutcome']
            all_train = fill_missing_values(all_train, missing_columns)
            all_test = fill_missing_values(all_test, missing_columns)

    error_table = np.zeros((max_depth_size, len(report_columns)))
    for d in range(1, max_depth_size + 1):
        for i in range(len(impurity_types)):
            car_decision_tree = DecisionTree(
                dataframe=all_train,
                attributes=x_train,
                labels=y_train,
                max_depth=d,
                impurity_type=impurity_types[i]
            )
            training_error = car_decision_tree.training_error(columns[-1])
            error_table[d - 1, 2 * i] = training_error
            test_error = car_decision_tree.evaluate(all_test, columns[-1])
            error_table[d - 1, 2 * i + 1] = test_error

    table = pd.DataFrame(error_table, columns=report_columns)
    table.insert(0, 'max_depth', value=np.arange(1, max_depth_size+1))
    print(table.to_string(index=False))
