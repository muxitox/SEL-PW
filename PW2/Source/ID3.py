import numpy as np
import math
from Tree import Tree
import pandas as pd
import copy
import random

'''
Performs PRISM over the provided dataset
'''

class ID3:
    # Loads data from a path
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.root = Tree()

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
            pXc_i = Y[Y == c_i].size / Y.size
            if pXc_i > 0:
                InfoXC -= (pXc_i * math.log2(pXc_i))

        # Calculate the conditional entropy given Ak
        InfoXA = 0
        for vi in X[Ak].unique():
            Xvi = X.loc[X[Ak] == vi]
            pXvi = len(Xvi.index) / len(X.index)

            Yvi = Y.loc[Xvi.index]

            InfoXCVi = 0
            for c_i in classes:
                pXvici = Yvi[Yvi == c_i].size / Yvi.size
                if pXvici > 0:
                    InfoXCVi -= (pXvici * math.log2(pXvici))

            InfoXA += pXvi * InfoXCVi

        return InfoXC-InfoXA

    # Generates the decision tree following the ID3 algorithm for the dataset X, labels Y and the set of attributes A
    def generate_tree(self, X, Y, A, F):
        if Y.value_counts().size == 1 or not A:
            child = Tree()
            child.set_class(Y.mode()[0])
            return child

        else:
            if len(A) > F:
                atr_selection = random.sample(A, F)
            else:
                atr_selection = A

            max_ig = -math.inf
            for Ak in atr_selection:
                ig = self.calculate_ig(X, Y, Ak)
                if ig > max_ig:
                    best_attribute = Ak
                    max_ig = ig

            # Generate subtrees splitting by best_attribute
            values_list = self.data[best_attribute].unique()

            A = [a1 for a1 in A if a1 != best_attribute]
            root = Tree()
            for vi in values_list:
                Xvi = X.loc[X[best_attribute] == vi]
                Yvi = Y.loc[Xvi.index]

                if Xvi.empty:
                    child = Tree()
                    child.set_class(Y.mode()[0])
                else:
                    child = self.generate_tree(Xvi, Yvi, A, F)

                child.set_attribute(best_attribute)
                child.set_value(vi)
                root.add_node(child)

            # This is made in order to predict an output, for generalization, when a new value of an instance that had
            # never appeared in the training appears in the test
            root.set_class(Y.mode()[0])

            return root

    # Generates the ID3 tree
    def fit(self, F):
        labels = self.labels
        data = self.data
        attributes = list(data)
        root = self.generate_tree(data, labels, attributes, F)
        self.root = root

        return root

    def predict_instance(self, instance, label):
        root = self.root

        while root.children:
            children = root.children
            attribute = children[0].attribute
            found = False
            for child in children:
                if child.value == instance[attribute]:
                    root = child
                    found = True
                    break

            if not found:
                print('mierdaaa')
                # None of the child had the same label than the instance
                return root.clss

        return root.clss

    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
        count = 0
        for index, instance in test_data.iterrows():
            label = test_labels.loc[index]
            value = self.predict_instance(instance, label)
            if label == value:
                count += 1

        print(count/len(test_data.index))

