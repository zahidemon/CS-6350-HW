import numpy as np
import pandas as pd


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


def create_dataframe(filename):
    data = {
        "buying": [],
        "maint": [],
        "doors": [],
        "persons": [],
        "lug_boot": [],
        "safety": [],
        "labels": []
    }
    data_keys = list(data.keys())
    with open(filename, "r") as f:
        for line in f:
            row = line.strip().split(",")
            for i in range(len(row)):
                data.get(data_keys[i]).append(row[i])

    return pd.DataFrame(data)


def filter_dataframe(dataframe, column, value):
    return dataframe[dataframe[column] == value]


if __name__ == "__main__":
    print(create_dataframe("../../Data/Car/train.csv"))
