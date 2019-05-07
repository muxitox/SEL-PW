import numpy as np
import math
from Tree import Tree
import pandas as pd
from ID3 import ID3
import copy
from sklearn.utils import resample
import statistics
import random
from operator import itemgetter



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
        pred_list = np.empty(1)
        for tree in self.estimators:
            pred = tree.predict_instance(instance)
            pred_list = np.append(pred_list, pred)

        # Solves ties randomly
        unique, counts = np.unique(pred_list, return_counts=True)
        higher_freq = max(counts)

        mode = []
        for (u, c) in zip(unique, counts):
            if c == higher_freq:
                mode.append(u)

        return random.sample(mode, 1)

    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
        prediction = copy.deepcopy(test_labels)
        for index, instance in test_data.iterrows():
            value = self.predict_instance(instance)
            prediction.loc[index] = value[0]

        accuracy = sum(test_labels.eq(prediction)) / len(test_data.index)
        return prediction, accuracy

    def importance_list(self):
        feat_list = []
        for tree in self.estimators:
            feat_list.append(tree.root.children[0].attribute)

        unique, counts = np.unique(feat_list, return_counts=True)

        counts = [a/self.NT for a in counts]
        feat_count = list(zip(unique, counts))
        feat_order = sorted(feat_count, key=lambda tup: -tup[1])

        return feat_order



