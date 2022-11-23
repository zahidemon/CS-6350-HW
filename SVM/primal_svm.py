import numpy as np
import pandas as pd

primal_svm_file = open("p_svm.txt", "w")
primal_svm_file.write(f"C\ta_training_error\ta_testing_error\tepoch_training_error\tepoch_testing_error\ta_weights\te_weights\n")


class PrimalSVM:
    def __init__(self, lr_type, a, bias=0, learning_rate=0.001, c=1.0):
        self.c = c
        self.w = None
        self.a = a
        self.bias = bias
        if lr_type == "lr_a":
            self.lr_inc = self.learning_rate_increase_on_a
        elif lr_type == "lr_epoch":
            self.lr_inc = self.learning_rate_increase_on_epoch

        self.learning_rate = learning_rate

    def fit(self, x, y, epochs=100):
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        else:
            x = np.hstack((x, np.zeros((x.shape[0], 1))))

        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # initialize weights

        for epoch in range(epochs):
            lr_epoch = self.lr_inc(epoch)
            # shuffle the data
            idx = np.random.permutation(n_samples)
            x = x[idx]
            y = y[idx]

            for i in range(n_samples):
                xi = x[i]
                yi = y[i]
                if yi * np.dot(xi, self.w) <= 1:
                    dw = np.append(self.w[:len(self.w) - 1], 0) - self.c * n_samples * yi * xi
                    self.w = self.w - lr_epoch * dw
                else:
                    self.w[:len(self.w) - 1] = (1 - lr_epoch) * self.w[:len(self.w) - 1]

    def learning_rate_increase_on_epoch(self, epoch):
        return self.learning_rate / (1 + epoch)

    def learning_rate_increase_on_a(self, epoch):
        return self.learning_rate / (1 + (self.learning_rate * epoch) / self.a)

    def predict(self, x):
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        else:
            x = np.hstack((x, np.zeros((x.shape[0], 1))))

        return np.sign(np.dot(x, self.w))

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)


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

    lr = 0.001
    a = 0.001
    T = 100
    C_list = [100 / 873, 500 / 873, 700 / 873]
    a_weights = []
    e_weights = []

    for C in C_list:
        p_svm_a = PrimalSVM("lr_a", a, learning_rate=lr, c=C)
        p_svm_a.fit(train_x, train_y)
        a_pred_train = p_svm_a.predict(train_x)
        a_pred_test = p_svm_a.predict(test_x)

        a_training_error = p_svm_a.evaluate(train_x, train_y)
        a_testing_error = p_svm_a.evaluate(test_x, test_y)
        a_weights.append(p_svm_a.w)

        p_svm_e = PrimalSVM("lr_epoch", a, learning_rate=lr, c=C)
        p_svm_e.fit(train_x, train_y)
        pred_train = p_svm_e.predict(train_x)
        pred_test = p_svm_e.predict(test_x)

        e_training_error = p_svm_e.evaluate(train_x, train_y)
        e_testing_error = p_svm_e.evaluate(test_x, test_y)
        e_weights.append(p_svm_e.w)
        primal_svm_file.write(f"{C}\t{a_training_error}\t{a_testing_error}\t{e_training_error}\t{e_testing_error}\t"
                              f"{p_svm_a.w}\t{p_svm_e.w}\n")

    primal_svm_file.close()
