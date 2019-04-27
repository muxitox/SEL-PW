import copy
import numpy as np
import math
from Tree import Tree

'''
Performs PRISM over the provided dataset
'''

class ID3:
    # Loads data from a path
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

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
            print(Xvi)

            pXvi = Xvi.size / X.size

            Yvi = Y.loc[Xvi.index]

            InfoXCVi = 0
            for c_i in classes:
                pXvici = Yvi[Yvi == c_i].size / Xvi.size
                if pXvici > 0:
                    InfoXCVi -= (pXvici * math.log2(pXvici))

            InfoXA += pXvi * InfoXCVi

        return InfoXC-InfoXA

    @staticmethod
    # Generates the decision tree following the ID3 algorithm for the dataset X, labels Y and the set of attributes A
    def generate_tree(self, X, Y, A):
        if Y[Y == Y.mode].size == Y.size or not A:
            child = Tree(is_leaf=True)
            child.set_class(Y.mode)
            return child

        else:
            max_ig = -math.inf
            for Ak in A:
                ig = self.calculate_ig(X, Ak)
                if ig > max_ig:
                    best_attribute = Ak
                    max_ig = ig

            # Generate subtrees splitting by best_attribute
            values_list = X[best_attribute].unique()

            A = [a1 for a1 in A if a1 != best_attribute]
            root = Tree()
            for vi in values_list:
                Xvi = X.loc[X[best_attribute] == vi]
                Yvi = Y.loc[Xvi.index]

                if Xvi.empty:
                    child = Tree()
                    child.set_class(Yvi.mode)
                else:
                    child = self.generate_tree(Xvi, Yvi, A)

                child.set_attribute(best_attribute)
                child.set_value(vi)
                root.add_node(child)

            return root

    # Generates the ID3 tree
    def fit(self, verbose):
        labels = self.labels
        data = self.data
        attributes = list(data)
        root = self.generate_tree(data, labels, attributes)



    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
       return
