import copy
import numpy as np
import math

'''
Performs PRISM over the provided dataset
'''

class ID3:
    # Loads data from a path
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.rules = []

    # Generates all the possible combinations of attribute/value pairs remaining in the dataframe
    @staticmethod
    def generate_all_pairs(data):
        pairs_list = []

        for attribute in list(data):
            for value in data[attribute].unique():
                pairs_list.append((attribute, value))

        return pairs_list

    # Calculates The information Gain
    # X are the data, Y are the labels and Ak the attribute for which to calculate the information gain
    @staticmethod
    def calculate_ig(X,Y,Ak):
        classes = Y.unique()

        # Calculate the Entropy of X with respect to C
        InfoXC = 0
        for c_i in classes:
            pXc_i = Y[Y == c_i].size / X.size
            InfoXC -= (pXc_i * math.log2(pXc_i))

        # Calculate the conditional entropy given Ak
        InfoXA = 0
        for vi in X[Ak].unique():
            Xvi = X.loc[X[Ak] == vi]

            pXvi = Xvi.size / X.size

            Yvi = Y.loc[Xvi.index]

            InfoXCVi = 0
            for c_i in classes:
                pXvici = Yvi[Yvi == c_i].size / Xvi.size
                InfoXCVi -= (pXvici * math.log2(pXvici))

            InfoXA += pXvi * InfoXCVi

        return InfoXC-InfoXA

    @staticmethod
    # Generates the decision tree following the ID3 algorithm for the dataset X, labesl Y and the set of attributes A
    def generate_tree(self, X, Y, A):
        max_ig = -1
        for Ak in A:
            ig = self.calculate_ig(X, Ak)
            if ig > max_ig:
                best_attribute = Ak


        # TODO: take vi as root and recursevily build the tree


    # Generates the ID3 tree
    def fit(self, verbose):
        labels = self.labels
        data = self.data
        attributes = list(data)
        self.generate_tree(data, labels, attributes)



    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
       return
