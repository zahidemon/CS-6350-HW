import numpy as np


def calculate_entropy(counts):
    total_entropy = 0
    for count in counts:
        total_entropy += -(count/sum(counts))*np.log2((count/sum(counts)))
    return total_entropy


def calculate_gini_index(counts):
    total_gini = 1
    for count in counts:
        total_gini -= (count/sum(counts))**2
    return total_gini


def calculate_majority_error(counts):
    return abs(counts[0] - counts[1])/(counts[0] + counts[1])


def calculate_feature_gain(feature_counts, splitting_type="entropy"):
    total_data = sum([sum(features) for features in feature_counts])
    total_gain = 0
    for feature_count in feature_counts:
        if splitting_type == "entropy":
            total_gain += (sum(feature_count) / total_data) * calculate_entropy(feature_count)
        elif splitting_type == "gini":
            total_gain += (sum(feature_count) / total_data) * calculate_gini_index(feature_count)
        else:
            total_gain += (sum(feature_count) / total_data) * calculate_majority_error(feature_count)
    return total_gain


def get_feature_counts(data):
    unique_values = list(set(data))
    feature_counts = []
    for unique_value in unique_values:
        feature_counts.append(data.count(unique_value))
    return feature_counts


def choose_best_feature(dataframe, splitting_type, label):
    class_label_counts = dataframe[label].tolist()
    if splitting_type == "entropy":
        information = calculate_entropy(class_label_counts)
    elif splitting_type == "gini":
        information = calculate_gini_index(class_label_counts)
    else:
        information = calculate_majority_error(class_label_counts)
    gains = []
    column_names = []
    for (column_name, column_data) in dataframe.iteritems():
        if column_name == label:
            continue
        column_names.append(column_name)
        feature_counts = get_feature_counts(column_data)
        gains.append(information - calculate_feature_gain(feature_counts, splitting_type))

    return column_names[gains.index(max(gains))]



