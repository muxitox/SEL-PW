import pandas as pd
import numpy as np
from preprocessing import preprocess
from sklearn.model_selection import StratifiedKFold
from PRISM import PRISM
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
    # data, labels = load_data('./data/iris.data', 'class')
    # data, labels = load_data('./data/test.csv','t')

    data, labels = load_data(sys.argv[1], sys.argv[2])



    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)

    train = []
    test = []
    verbose = False
    errors = []
    NaN_rates = []
    times = []
    for (train, test) in cv.split(data, labels):
        data_train = data.loc[train]
        labels_train = labels.loc[train]
        data_test = data.loc[test]
        labels_test = labels.loc[test]

        my_prism = PRISM(data_train, labels_train)
        start_time = time.time()
        my_prism.fit(verbose)
        elapsed_time = time.time() - start_time

        error, NaNrate = my_prism.predict(data_test, labels_test)
        errors.append(error)
        NaN_rates.append(NaNrate)
        times.append(elapsed_time)

    print('Error', errors)
    print('Mean error', np.mean(errors))
    print('Std error', np.std(errors))
    print('')
    print('NaNs', NaN_rates)
    print('Mean NaN rate', np.mean(NaN_rates))
    print('Std NaN rate', np.std(NaN_rates))
    print('')
    print('Time', times)
    print('Mean time', np.mean(times))
    print('Std time', np.std(times))


    my_prism = PRISM(data, labels)
    my_prism.fit(True)
    


