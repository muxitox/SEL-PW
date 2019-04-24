
class Rule:

    # Loads data from a path
    def __init__(self, consequent):
        self.antecedent = []
        self.consequent = consequent
        self.precision = 0
        self.coverage = 0

    def add_pair(self, pair):
        self.antecedent.append(pair)

    def set_antecedent(self, antecedent):
        self.antecedent = antecedent

    def set(self, rule):
        self.antecedent = rule.antecedent
        self.consequent = rule.consequent
        self.precision = rule.precision
        self.coverage = rule.coverage

    def get_antecedent(self):
        return self.antecedent

    def is_perfect(self):
        return self.precision == 1.0

    # Calculates the precission of the rule according to data and labels
    # Intances_left is only used to see if the rule is valid in the current frame
    def calculate_precision(self, data, labels, instances_left):
        # Create the query in a string with all the tuples
        my_query = ''

        for pair in self.antecedent:

            my_query = my_query + ' and ' + pair[0] + ' == \'' + str(pair[1]) + '\''

        my_query = my_query[5:]

        # Make the query to PANDAS
        queried_data = data.query(my_query)

        # Calculate total
        total = queried_data.shape[0]

        # Look for the positives of the query in the labels dataframe
        query_labels = labels.loc[queried_data.index]
        positive_labels = query_labels.loc[query_labels == self.consequent]
        positive = positive_labels.shape[0]

        if self.is_present(instances_left):
            if total == 0:
                self.precision = 0
                self.coverage = 0
            else:
                self.precision = positive / total
                self.coverage = positive / data.shape[0]
        else:
            self.precision = 0
            self.coverage = 0

    # Make sure that the rule is an instance of the dataframe
    def is_present(self, data):
        # Create the query in a string with all the tuples
        my_query = ''

        for pair in self.antecedent:

            my_query = my_query + ' and ' + pair[0] + ' == \'' + str(pair[1]) + '\''

        my_query = my_query[5:]

        # Make the query to PANDAS
        queried_data = data.query(my_query)

        # Calculate total
        total = queried_data.shape[0]

        return total>0

    # Subtracts from data and labels the positive
    def find_negative(self, data, labels):
        # Create the query in a string with all the tuples
        my_query = ''

        for pair in self.antecedent:
            my_query = my_query + ' and ' + pair[0] + ' == \'' + str(pair[1]) + '\''

        my_query = my_query[5:]

        # Make the query to PANDAS
        queried_data = data.query(my_query)
        valid_index = data.index.difference(queried_data.index)
        valid_data = data.loc[valid_index,:]
        valid_labels = labels.loc[valid_index]

        return valid_data, valid_labels

    def match(self, instance):
        for (antecedent, consequent) in self.antecedent:
            if instance[antecedent] != consequent:
                return False
        return True





