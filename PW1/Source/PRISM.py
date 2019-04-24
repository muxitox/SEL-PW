from Rule import Rule
import copy
import numpy as np

'''
Performs PRISM over the provided dataset
'''

class PRISM:
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

    # Mines the PRISM rules
    def fit(self, verbose):

        labels = self.labels
        data = self.data
        classes = labels.unique()

        for curr_class in classes:

            index_class_left = labels.loc[labels[:] == curr_class]
            instances_left = data.loc[index_class_left.index]


            # While there are instances available
            while not instances_left.empty:

                # Create empty rule
                all_pairs = self.generate_all_pairs(instances_left)
                curr_rule = Rule(curr_class)
                # While rule is not empty and there are attribute/value pairs available
                while (not curr_rule.is_perfect()) and len(all_pairs) > 0:
                    best_pair = ()
                    # Dummy rule with 0 precision and 0 coverage
                    best_pair_rule = Rule(curr_class)
                    # Combine current rule with the rest of attribute/value pairs
                    for pair in all_pairs:
                        # Calculate precision
                        test_rule = Rule(curr_rule.consequent)
                        test_rule.set_antecedent(copy.copy(curr_rule.antecedent))
                        test_rule.add_pair(copy.copy(pair))
                        test_rule.calculate_precision(data, labels, instances_left)
                        if test_rule.precision > best_pair_rule.precision:
                            best_pair_rule.set(copy.copy(test_rule))
                            best_pair = copy.copy(pair)
                        elif test_rule.precision == best_pair_rule.precision and \
                                test_rule.coverage > best_pair_rule.coverage:
                            best_pair_rule.set(copy.copy(test_rule))
                            best_pair = copy.copy(pair)

                    curr_rule.set(copy.copy(best_pair_rule))
                    # Remove the selector of the list of pairs
                    all_pairs = [(a, b) for (a, b) in all_pairs if a != best_pair[0]]


                # Out of the while loop, the rule is perfect or there are not more pairs
                # Remove the covered instances out of the dataset
                self.rules.append(curr_rule)
                instances_left, index_class_left = curr_rule.find_negative(instances_left, index_class_left)


        # Order rules
        if verbose:
            print('')
            print(len(self.rules), ' Rules (discovery order)')
            print('-------------------------------------------------------------------------------------')
            for rule in self.rules:
                ant_str = ""
                for ant in rule.antecedent:
                    ant_str = ant_str + '  and  ' + ant[0] + '=\'' + str(ant[1]) + '\''

                ant_str = ant_str[7:]
                print("# ", ant_str, '=>', rule.consequent, 'Prec: ', rule.precision, 'Coverage: ', rule.coverage)




        self.rules = sorted(self.rules, key=lambda x: (-x.precision, -x.coverage))

        if verbose:
            print('')
            print(len(self.rules), ' Rules (ordered by precision and coverage)')
            print('-------------------------------------------------------------------------------------')
            for rule in self.rules:
                ant_str = ""
                for ant in rule.antecedent:
                    ant_str = ant_str + '  and  ' + ant[0] + '=\'' + str(ant[1]) + '\''

                ant_str = ant_str[7:]
                print("# ", ant_str, '=>', rule.consequent, 'Prec: ', rule.precision, 'Coverage: ', rule.coverage)

                # print(rule.antecedent, '=>', rule.consequent, 'Prec: ', rule.precision, 'Cover: ', rule.coverage)

    # Returns the label of the first rule he finds or NaN if not found
    def predict_instance(self, instance):
        for rule in self.rules:
            if rule.match(instance):
                return rule.consequent
        return np.NaN

    # Predicts the values for the test data
    def predict(self, test_data, test_labels):
        # Predict each instance
        error_count = 0
        NaN_count = 0
        for i in range(0, test_data.shape[0]):
            prediction = self.predict_instance(test_data.iloc[i])
            if prediction != test_labels.iloc[i]:
                error_count += 1
            if prediction != prediction:
                NaN_count += 1

        error = error_count/test_labels.shape[0]
        NaN_rate = NaN_count/test_labels.shape[0]
        return error, NaN_rate
