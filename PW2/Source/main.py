import pandas as pd
import numpy as np
from preprocessing import preprocess
from sklearn.model_selection import StratifiedKFold
import time
import sys




def load_data(data_path, class_name):
    data = pd.read_csv(data_path)  # doctest: +SKIP
    new_names = [a.translate(str.maketrans({"-": r"_"})) for a in list(data)]
    data.columns = new_names
    labels = data[class_name]
    data = data.loc[:, data.columns != class_name]
    # labels = data.iloc[:, -1]
    # data = data.iloc[:, :-1]
    data = preprocess(data)

    return data, labels


if __name__ == "__main__":

    # Run PRISM
    # data, labels = load_data('./data/horse.csv', 'outcome')
    # data, labels = load_data('./data/breast-cancer.data', 'Class')
    data, labels = load_data('../Data/iris.data', 'class')
    # data, labels = load_data('./data/test.csv','t')

    # data, labels = load_data(sys.argv[1], sys.argv[2])

    attributes = list(data)
    classes = labels.unique()

    pXc_i = labels[labels==labels[0]].size / data.size
    print(pXc_i)



