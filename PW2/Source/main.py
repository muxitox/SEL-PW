import pandas as pd
import numpy as np
from preprocessing import preprocess
from Random_Forest import RF
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#from id3 import Id3Estimator
#from id3 import export_graphviz
import math
import time
import sys




def load_data(data_path, class_name):
    data = pd.read_csv(data_path)  # doctest: +SKIP
    new_names = [a.translate(str.maketrans({"-": r"_"})) for a in list(data)]
    data.columns = new_names
    labels = data[class_name]
    data = data.loc[:, data.columns != class_name]
    data = preprocess(data)

    return data, labels


if __name__ == "__main__":

    # Run Random Forest
    # data, labels = load_data('../data/breast-cancer.data', 'Class')
    # data, labels = load_data('../Data/iris.data', 'class')
    # data, labels = load_data('../data/hepatitis.data', 'Class')

    print('Loading Dataset...')
    data, labels, NT, F = load_data(sys.argv[1], sys.argv[2], sys.argv[4], sys.argv[4])
    # Holdout
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=103, stratify=labels)

    print('Training the Random Forest...')
    rf = RF(NT, F)
    start_time = time.time()
    rf = rf.fit(X_train, y_train, NT, F)
    elapsed_time = time.time() - start_time

    print('Predicting...')
    # Predict the result
    pred, acc = rf.predict(X_test, y_test)
    feat_importance = rf.importance_list()
    print('')
    print('F:', F, ', NT:', NT, ', Accuracy: ', acc, ', Time: ', elapsed_time)
    print('Feature importance ordering', feat_importance)



