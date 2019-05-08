import math
from Tree import Tree
import random
import copy

'''
Performs ID3 on the provided dataset
'''

class ID3:
    # Initializes de ID3 tree
    def __init__(self):
        self.root = Tree()
        self.data = None

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
        # All the instances have the same label or we have not more attributes to split with A
        if Y.value_counts().size == 1 or not A:
            child = Tree()
            child.set_class(Y.mode()[0])
            return child

        else:
            # Sampling F features from A if F<|A|
            if len(A) > F:
                atr_selection = random.sample(A, F)
            else:
                atr_selection = A

            # Computing the Information Gain of each attribute Ak
            max_ig = -math.inf
            for Ak in atr_selection:
                ig = self.calculate_ig(X, Y, Ak)
                if ig > max_ig:
                    best_attribute = Ak
                    max_ig = ig

            values_list = self.data[best_attribute].unique()
            # Remove the selected attribute from the list of attributes A
            A = [a1 for a1 in A if a1 != best_attribute]
            root = Tree()

            # Generate subtrees splitting by best_attribute
            for vi in values_list:
                # Load the instances containing the value vi for Ak
                Xvi = X.loc[X[best_attribute] == vi]
                Yvi = Y.loc[Xvi.index]

                # There is no instances containing vi anymore, generalize creating a leaf with the mode of the parent
                if Xvi.empty:
                    child = Tree()
                    child.set_class(Y.mode()[0])
                # Create a child tree subsplitting Xvi
                else:
                    child = self.generate_tree(Xvi, Yvi, A, F)

                # Add the subtrees to the root
                child.set_attribute(best_attribute)
                child.set_value(vi)
                root.add_node(child)

            # This is made in order to predict an output, for generalization, when a new value of an instance that had
            # never appeared in the training appears in the test
            root.set_class(Y.mode()[0])

            return root

    # Generates the ID3 tree
    def fit(self, data, labels, F):
        self.data = data
        attributes = list(data)
        root = self.generate_tree(data, labels, attributes, F)
        self.root = root

        return root

    def predict_instance(self, instance):
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
                # None of the child had the same label than the instance
                return root.clss

        return root.clss

    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
        prediction = copy.deepcopy(test_labels)
        for index, instance in test_data.iterrows():
            value = self.predict_instance(instance)
            prediction.loc[index] = value

        accuracy = sum(test_labels.eq(prediction))/len(test_data.index)
        return prediction, accuracy


