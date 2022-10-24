import numpy as np
from matplotlib import pyplot as plt
from LinearRegression.utils import get_data, get_error, get_prediction

bgd_file = open("bgd_logs.txt", 'w')
bgd_file.write("iteration\t cost\t weight_diff\n")


def optimize_batch_gd(x, y, learning_rate=0.001, convergence=0.000001, max_iteration=10000):
    number_of_features = x.shape[1]
    gd_weights = np.zeros(number_of_features)
    cost_values = []

    for i in range(max_iteration):
        y_prediction = np.dot(x, gd_weights)
        dw = np.dot(x.T, (y_prediction - y))
        delta = - learning_rate * dw
        weight_diff = np.sqrt(np.sum(np.square(delta)))
        gd_weights += delta
        error = get_error(y, y_prediction)
        cost_values.append(error)
        bgd_file.write(f"{str(i)}\t {str(error)}\t {weight_diff}\n")
        if weight_diff <= convergence:
            break
    return cost_values, gd_weights


if __name__ == "__main__":
    train_filename = "../Data/Concrete/train.csv"
    test_filename = "../Data/Concrete/test.csv"
    column_names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'slump']
    x_train, y_train, x_test, y_test = get_data(train_filename, test_filename, column_names)

    lr = 0.01
    costs, weights = optimize_batch_gd(x_train, y_train, learning_rate=lr)
    y_pred = get_prediction(x_test, weights)
    test_error = get_error(y_test, y_pred)

    bgd_file.write(f"\n Final weights: {weights}\n")
    bgd_file.write(f"\n Test error: {test_error}\n")

    plot_filename = "cost_change_bgd.jpeg"
    plot_title = f"cost change after each iteration for lr={lr}"
    plt.plot(range(len(costs)), costs)
    plt.title(plot_title)
    plt.savefig(plot_filename)
    plt.show()
    plt.clf()
    plt.close()
    bgd_file.close()

