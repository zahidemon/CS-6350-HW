import math

import numpy as np
import pandas as pd

from DecisionTree.id3 import DecisionTree
from DecisionTree.utils.data import get_attributes_and_labels, apply_thresholding

if __name__ == "__main__":
    train_filename = "../Data/Bank/train.csv"
    test_filename = "../Data/Bank/test.csv"
    adaboost_file = open("Adaboost_logs.txt", 'w')
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

    T = 1
    train_size = len(all_train)
    test_size = len(all_test)
    alphas = [0 for x in range(T)]
    weights = np.array([1 / train_size for x in range(train_size)])

    training_error = [0 for x in range(T)]
    test_error = [0 for x in range(T)]
    train_error_overall = [0 for x in range(T)]
    test_error_overall = [0 for x in range(T)]
    train_y = np.array([0 for x in range(train_size)])
    test_y = np.array([0 for x in range(train_size)])

    for t in range(T):
        decision_tree = DecisionTree(
                dataframe=all_train,
                attributes=x_train,
                labels=y_train,
                max_depth=1,
                weights=pd.Series(weights),
                impurity_type="weighted_entropy"
            )
        # training error
        train_preds = decision_tree.predictions(all_train)
        print(decision_tree.weighted_evaluate(all_train, columns[-1], weights))
        # print(type(train_preds))
        # print(train_preds)
        # train_preds[train_preds == 'yes'] = 1
        # train_preds[train_preds == 'no'] = 0
        # err = 1 - train_preds.sum() / train_size
        # training_error[t] = err
        # #
        # # # weighted error and alpha
        # tmp = np.array(train_preds.tolist())
        # w = weights[tmp == 0]
        # err = np.sum(w)
        # alpha = 0.5 * math.log((1 - err) / err)
        # alphas[t] = alpha
        # #
        # # # updated weights
        # weights = np.exp(tmp * -alpha) * weights
        # total = np.sum(weights)
        # weights = weights / total
        # #
        # # # testing error
        # test_preds = decision_tree.predictions(all_test)
        # test_preds[test_preds == 'yes'] = 1
        # test_preds[test_preds == 'no'] = 0
        # test_error[t] = 1 - test_preds.sum() / test_size
        # #
        # # # combined prediction so far
        # # # train
        # train_py = train_y + train_preds * alpha
        # err = 1 - train_y.sum() / train_size
        # train_error_overall[t] = err
        # #
        # # # test
        # test_py = test_y + test_preds * alpha
        # err = 1 - test_y.sum() / train_size
        # test_error_overall[t] = err
        #
        # adaboost_file.write(f't: {t}, train_stump_err: {training_error[t]}, test_stump_err: {test_error[t]}, '
        #                     f'train_overall_err: {train_error_overall[t]} test_overall_err: {test_error_overall[t]}\n')

