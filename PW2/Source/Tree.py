class Tree(object):
    def __init__(self, attribute='root', value=None, clss=None, is_leaf=False):
        self.attribute = attribute
        self.value = value
        self.clss = clss
        self.is_leaf = is_leaf
        self.children = []

    def add_node(self, node):
        self.children.append(node)

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_value(self, value):
        self.value = value

    def set_class(self, clss):
        self.clss = clss
