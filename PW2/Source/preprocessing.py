import pandas as pd
import numpy as np


"""

Change NaN and discretize

"""


def preprocess(data):

    data.replace('?', np.NaN, inplace=True)
    data.replace('NA', np.NaN, inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='pad', inplace=True)
    # data = to_numbers(data)
    data = discretize(data)


    return data


"""

Creates bins for continuous variables

"""


def discretize(data):
    for column in range(0, len(data.columns)):
        if (np.issubdtype(data.iloc[:, column].dtype, np.number) and (data.iloc[:, column].nunique() > 8)):
            data.iloc[:, column] = pd.cut(data.iloc[:, column], 3)
            data.iloc[:, column] = data.iloc[:, column].astype(str)
    return data
