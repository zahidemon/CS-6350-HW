import numpy as np
import pandas as pd

nn_2bc = open("nn_2.txt", "w")
nn_2bc.write(f"#nodes\t random training_error\trandom testing_error \t zero training_error\tzero"
             f" testing_error\n")
class ANN:

    def __init__(self, number_of_features, number_of_nodes, weights_init, d, learning_rate):
        self.learning_rate = learning_rate
        self.d = d
        if weights_init == "random":
            self.weights = [
                np.random.randn(number_of_features + 1, number_of_nodes),
                np.random.randn(number_of_nodes + 1, number_of_nodes),
                np.random.randn(number_of_nodes + 1, 1)
            ]
        else:
            self.weights = [
                np.zeros((number_of_features + 1, number_of_nodes)),
                np.zeros((number_of_nodes + 1, number_of_nodes)),
                np.zeros((number_of_nodes + 1, 1))
            ]


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            lr_a = self.learning_rate_increase_on_a(epoch)

            idx = np.random.permutation(len(x)) 
            train_x = x[idx]
            train_y = y[idx]
            for i in range(len(x)):
                activation_outputs = [train_x[i]] 
                z = []  
                for j in range(len(self.weights)):  
                    input_x = np.hstack((activation_outputs[j], np.ones(1)))
                    z.append(np.dot(input_x, self.weights[j]))  
                    activation_outputs.append(self.sigmoid(z[j]))

                # backward pass
                delta = [activation_outputs[-1] - train_y[i]]  
                for j in range(len(self.weights) - 1, 0, -1):
                    delta.append(np.dot(delta[-1], self.weights[j][:-1, :].T) * self.sigmoid_derivative(
                        activation_outputs[j + 1])) 
                delta.reverse()

                for j in range(len(self.weights)):
                    input_x = np.hstack((activation_outputs[j], np.ones(1)))
                    d_loss_w = np.dot(input_x[:, np.newaxis], delta[j][np.newaxis, :])
                    self.weights[j] -= lr_a * d_loss_w
            loss = 0
            for i in range(len(x)):
                a = [x[i]]
                for j in range(len(self.weights)):
                    input_x = np.hstack((a[j], np.ones(1)))
                    a.append(self.sigmoid(np.dot(input_x, self.weights[j])))
                loss +=  0.5 * ((a[-1] - y[i]) ** 2)
            loss /= len(x)
            print(f"Epoch: {epoch + 1}\t Training Loss: {loss}")

    def learning_rate_increase_on_a(self, epoch):
        return self.learning_rate / (1 + (self.learning_rate / self.d) * epoch)

    def predict(self, x):
        for i in range(len(x)):
            a = [x[i]] 
            for j in range(len(self.weights)):
                input_x = np.hstack((a[j], np.ones(1))) 
                a.append(self.sigmoid(np.dot(input_x, self.weights[j])))  
        return 1 if a[-1] >= 0.5 else 0

    def evaluate(self, x, y):
        return np.mean(self.predict(x) != y)

if __name__ == "__main__":
    train_dataframe = pd.read_csv('../Data/bank-note-2/train.csv', header=None)
    test_dataframe = pd.read_csv('../Data/bank-note-2/test.csv', header=None)
    train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    train_x = train_dataframe.iloc[:, :-1].values
    train_y = train_dataframe.iloc[:, -1].values
    test_x = test_dataframe.iloc[:, :-1].values
    test_y = test_dataframe.iloc[:, -1].values

    lr = 0.001
    d = 0.01
    T = 100
    number_of_nodes = [5, 10, 25, 50, 100]

    for number_of_node in number_of_nodes:
        print("Number of nodes: " + str(number_of_node))
        nn_model_random = ANN(train_x.shape[1], number_of_node, "random", d, lr)
        nn_model_random.train(train_x, train_y, T)

        random_training_error = nn_model_random.evaluate(train_x, train_y)
        random_testing_error = nn_model_random.evaluate(test_x, test_y)
        print('Training Error: ' + str(random_training_error))
        print('Testing Error: ' + str(random_testing_error))
        print()

        nn_model_zeros = ANN(train_x.shape[1], number_of_node, "zeros", d, lr)
        nn_model_zeros.train(train_x, train_y, T)

        zero_training_error = nn_model_zeros.evaluate(train_x, train_y)
        zero_testing_error = nn_model_zeros.evaluate(test_x, test_y)
        print('Training Error: ' + str(zero_training_error))
        print('Testing Error: ' + str(zero_testing_error))
        print()

        del nn_model_random, nn_model_zeros

        nn_2bc.write(f"{number_of_node}\t\t{random_training_error}\t\t{random_testing_error} \t\t{zero_training_error}"
                     f"\t\t{zero_testing_error}\n")

    nn_2bc.close()