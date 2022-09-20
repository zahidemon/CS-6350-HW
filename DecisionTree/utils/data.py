import numpy as np


def get_attributes_and_labels(filename):
    x, y = [], []
    with open(filename, "r") as f:
        for line in f:
            row = line.strip().split(",")
            x.append(row[:-1])
            y.append([row[-1]])

    x = np.array(x)
    y = np.array(y)

    return x, y
