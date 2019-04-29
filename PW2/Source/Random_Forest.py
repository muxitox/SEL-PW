import numpy as np
import math
from Tree import Tree
import pandas as pd
from ID3 import ID3
import copy
from sklearn.utils import resample
from scipy.stats import mstats
import random


'''
Performs Random Forest on  the provided dataset
'''


class RF:
    # Initializes the the Random Forest
    def __init__(self, NT, F):
        self.NT = NT
        self.F = F
        self.estimators = []

    # Generates
    def fit(self, data, labels, NT, F):

        for n in range(0,NT):
            dataR, labelsR = resample(data, labels)

            id3 = ID3()
            id3.fit(dataR, labelsR,F)
            self.estimators.append(id3)

        return self

    def predict_instance(self, instance):
        pred_list = []
        for tree in self.estimators:
            pred = tree.predict_instance(instance)
            pred_list.append(pred)

        # Solves ties randomly
        mode = mstats.mode(pred)
        return random.sample(mode, 1)

    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
        prediction = copy.deepcopy(test_labels)
        for index, instance in test_data.iterrows():
            value = self.predict_instance(instance)
            prediction.loc[index] = value

        accuracy = sum(test_labels.eq(prediction)) / len(test_data.index)
        return prediction, accuracy

