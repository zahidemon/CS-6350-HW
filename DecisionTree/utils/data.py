from collections import Counter

import pandas as pd


def get_attributes_and_labels(filename, columns):
    data = pd.read_csv(filename)
    data.columns = columns
    return data, list(data.columns[:-1]), data[columns[-1]]


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
    return dataframe[dataframe[column] == str(value)]


def apply_thresholding(dataframe, threshold, columns):
    for column in columns:
        dataframe[column] = dataframe[column].apply(lambda x: 0 if x <= threshold[column] else 1)
    return dataframe


def fill_missing_values(data, missing_columns):
    for column in missing_columns:
        value_and_frequency = Counter(data[column]).most_common(2)
        mode = [value for value, frequency in value_and_frequency if value != 'unknown'][0]
        data[column] = data[column].apply(lambda x: mode if x == 'unknown' else x)
    return data


if __name__ == "__main__":
    print(create_dataframe("../../Data/Car/train.csv"))
