import numpy as np
import pandas as pd

primal_svm_file = open("q5.txt", "w")
primal_svm_file.write(f"\tLearning rate\tweights\n")


class PrimalSVM:
    def __init__(self, a, bias=0, learning_rates=None, c=1.0):
        if learning_rates is None:
            learning_rates = [0.01, 0.005, 0.0025]
        self.c = c
        self.w = None
        self.a = a
        self.bias = bias

        self.learning_rates = learning_rates

    def fit(self, x, y, epochs=3):
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        else:
            x = np.hstack((x, np.zeros((x.shape[0], 1))))

        n_samples, n_features = x.shape
        self.w = np.zeros(n_features)  # initialize weights

        for epoch in range(epochs):
            lr_epoch = self.learning_rates[epoch]
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
                primal_svm_file.write(f"{lr_epoch}\t{dw}\n")

    def predict(self, x):
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        else:
            x = np.hstack((x, np.zeros((x.shape[0], 1))))

        return np.sign(np.dot(x, self.w))

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)


if __name__ == "__main__":
    train_x = np.array([
        [0.5, -1, 0.3],
        [-1, -2, -2],
        [1.5, 0.2, -2.5]
    ])
    train_y = np.array([1, -1, 1])

    lr = [0.01, 0.005, 0.0025]
    a = 0
    T = len(lr)
    a_weights = []
    e_weights = []

    p_svm_a = PrimalSVM(a, learning_rates=lr)
    p_svm_a.fit(train_x, train_y, epochs=T)
    a_pred_train = p_svm_a.predict(train_x)

    a_training_error = p_svm_a.evaluate(train_x, train_y)
    a_weights.append(p_svm_a.w)

    primal_svm_file.close()
