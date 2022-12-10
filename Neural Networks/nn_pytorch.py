import pandas as pd
import torch.nn as nn
import torch

nn_bonus = open("nn_bonus.txt", "w")
nn_bonus.write(f"#nodes\t#layer\ttanh_training_error\ttanh_testing_error \t relu_training_error\trelu_testing_error\n")

class ANNPyTorch:
    def __init__(self, input_size, no_of_layers, act_func, hidden_size, lr=0.001):
        self.lr = lr
        self.model = Model(input_size, no_of_layers, act_func, hidden_size)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X, Y, T):
        for t in range(T):
            self.optimizer.zero_grad()
            Y_pred = self.model(X)
            Y = Y.view(-1, 1)
            loss = self.loss_fn(Y_pred, Y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            Y_pred = self.model(X)
            Y_pred = (Y_pred > 0.5).float()
            return Y_pred

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        return torch.mean((Y_pred != Y).float())

    def get_weights(self):
        return self.model.state_dict()


class Model(nn.Module):
    def __init__(self, input_size, no_of_layers, activation_function, hidden_size):
        super(Model, self).__init__()
        self.no_of_layers = no_of_layers
        for i in range(no_of_layers):
            if i == 0:
                setattr(self, f"fc{i + 1}", nn.Linear(input_size, hidden_size))
            else:
                setattr(self, f"fc{i + 1}", nn.Linear(hidden_size, hidden_size))

        setattr(self, f"fc{no_of_layers + 1}", nn.Linear(hidden_size, 1))

        if activation_function == "relu":
            self.act = nn.ReLU()
            for i in range(1, len(self._modules)):
                nn.init.kaiming_uniform_(getattr(self, f"fc{i}").weight)
        elif activation_function == "tanh":
            self.act = nn.Tanh()
            for i in range(1, len(self._modules)):
                nn.init.xavier_uniform_(getattr(self, f"fc{i}").weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(1, self.no_of_layers + 1):
            x = getattr(self, f"fc{i}")(x)
            x = self.act(x)
        x = getattr(self, f"fc{self.no_of_layers + 1}")(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    train_dataframe = pd.read_csv('../Data/bank-note-2/train.csv', header=None)
    test_dataframe = pd.read_csv('../Data/bank-note-2/test.csv', header=None)
    train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    train_x = train_dataframe.iloc[:, :-1].values
    train_y = train_dataframe.iloc[:, -1].values
    test_x = test_dataframe.iloc[:, :-1].values
    test_y = test_dataframe.iloc[:, -1].values
    torch_trainX = torch.from_numpy(train_x).float()
    torch_trainY = torch.from_numpy(train_y).float()
    torch_testX = torch.from_numpy(test_x).float()
    torch_testY = torch.from_numpy(test_y).float()

    number_of_layers = [3, 5, 9]
    number_of_nodes = [5, 10, 25, 50, 100]
    lr = 0.001
    d = 0.01
    T = 100

    for number_of_layer in number_of_layers:
        for number_of_node in number_of_nodes:
            print("Number of nodes: " + str(number_of_node))
            nn_model_tanh = ANNPyTorch(train_x.shape[1], number_of_layer, "tanh", number_of_node, lr)
            nn_model_tanh.train(torch_trainX, torch_trainY, T)

            tanh_training_error = nn_model_tanh.evaluate(torch_trainX, torch_trainY)
            tanh_testing_error = nn_model_tanh.evaluate(torch_testX, torch_testY)
            print('Training Error: ' + str(tanh_training_error))
            print('Testing Error: ' + str(tanh_testing_error))
            print()


            # NeuralNetworkPytorch(input_size, no_of_layers, act_func, hidden_size, lr = 0.001):
            nn_model_relu = ANNPyTorch(train_x.shape[1], number_of_layer, "relu", number_of_node, lr)
            nn_model_relu.train(torch_trainX, torch_trainY, T)

            relu_training_error = nn_model_relu.evaluate(torch_trainX, torch_trainY)
            relu_testing_error = nn_model_relu.evaluate(torch_testX, torch_testY)
            print('Training Error: ' + str(relu_training_error))
            print('Testing Error: ' + str(relu_testing_error))
            print()

            del nn_model_tanh, nn_model_relu

            nn_bonus.write(
                f"{number_of_node}\t\t{number_of_layer}\t\t{tanh_training_error}\t\t{tanh_testing_error} "
                f"\t\t{relu_training_error}\t\t{relu_testing_error}\n")

    nn_bonus.close()