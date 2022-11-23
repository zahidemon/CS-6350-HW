import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

dual_svm_file_linear = open("d_svm_linear.txt", "w")
dual_svm_file_gaussian = open("d_svm_gaussian.txt", "w")
dual_svm_file_linear.write(f"C\ttraining_error\ttesting_error\tweights\n")
dual_svm_file_gaussian.write(f"C\tgamma\ttraining_error\ttesting_error\tepochs\tSV\n")


class DualSVM:
    def __init__(self, kernel_type, gamma=0.0, c=1.0):
        self.c = c
        self.lambdas = None
        self.w = None 
        self.b = None
        self.x = None 
        self.y = None
        self.gamma = gamma
        self.support_vectors = None
        self.overlapping_support_vectors = None
        if kernel_type == "linear":
            self.kernel = self.linear_kernel 
        elif kernel_type == "gaussian":
            self.kernel = self.gaussian_kernel

    def fit(self, x, y: np.ndarray):
        n_samples, n_features = x.shape
        self.x = x
        self.y = y
        self.lambdas = np.zeros(n_samples)

        constraints = ({'type': 'eq', 'fun': self.constraints})
        bounds = Bounds(0, self.c)

        initial_guess = np.zeros(n_samples) 
        solution = minimize(fun=self.objective_function, x0=initial_guess, bounds=bounds,
                            method='SLSQP', constraints=constraints)
        self.lambdas = solution.x
        self.support_vectors = np.where(self.lambdas > 1e-5)[0]
        self.overlapping_support_vectors = np.where((self.lambdas > 1e-5) & (self.lambdas < self.c))[0] 
        self.w = np.dot(self.lambdas * self.y, self.x)
        self.b = np.dot(self.lambdas, self.y)

    def constraints(self, lambdas):
        return np.dot(lambdas.T, self.y)

    def predict(self, x):
        prediction_res = []

        for i in range(len(x)):
            prediction = np.sign(sum(self.lambdas[self.support_vectors] * self.y[self.support_vectors] *
                                     self.kernel(self.x[self.support_vectors], x[i])))
            if prediction > 0:
                prediction_res.append(1)
            else:
                prediction_res.append(-1)

        return np.array(prediction_res)

    def evaluate(self, x, y):
        return np.mean(self.predict(x) != y)

    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2.T)

    def objective_function(self, lambdas):
        out = -np.sum(lambdas) + 0.5 * np.dot(self.lambdas,
                                              np.dot(self.lambdas.T,
                                              (self.y @ self.y.T) * self.kernel(self.x, self.x)))
        return out

    def gaussian_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return np.exp(-np.linalg.norm(x1-x2)**2 / self.gamma)


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
    gamma_list = [0.1, 0.5, 1, 5, 100]
    linear_weights = []
    gaussian_weights = []

    for C in C_list:
        d_svm_linear = DualSVM("linear", c=C)
        d_svm_linear.fit(train_x, train_y)
        a_pred_train = d_svm_linear.predict(train_x)
        a_pred_test = d_svm_linear.predict(test_x)

        linear_training_error = d_svm_linear.evaluate(train_x, train_y)
        linear_testing_error = d_svm_linear.evaluate(test_x, test_y)
        linear_weights.append(d_svm_linear.w)
        dual_svm_file_linear.write(f"{C}\t{linear_training_error}\t{linear_testing_error}\t{d_svm_linear.w}\n")
    dual_svm_file_linear.close()
    
    for gamma in gamma_list:
        for C in C_list:
            dual = DualSVM("gaussian", c=C, gamma=gamma)
            dual.fit(train_x, train_y)
            pred_train = dual.predict(train_x)
            pred_test = dual.predict(test_x)
            print('C: ' + str(C) + ', gamma: ' + str(gamma))
            print('Train Error: ' + str(dual.evaluate(train_x, train_y)))
            print('Test Error: ' + str(dual.evaluate(test_x, test_y)))
            gaussian_weights.append(np.append(dual.w, dual.b))
            # print support vectors
            print("Number of support vectors: " + str(len(dual.support_vectors)))

            if C == 500 / 873:
                # print overlapping support vectors
                print("Number of overlapping support vectors: " + str(len(dual.overlapping_support_vectors)))

            dual_svm_file_gaussian.write(f"{C}\t{gamma}\t{dual.evaluate(train_x, train_y)}\t{dual.evaluate(test_x, test_y)}\t{np.append(dual.w, dual.b)}\t{len(dual.support_vectors)}\n")

    dual_svm_file_gaussian.close()
