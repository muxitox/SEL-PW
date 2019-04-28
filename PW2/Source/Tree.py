class Tree(object):
    def __init__(self, attribute='root', value=None, clss=None):
        self.attribute = attribute
        self.value = value
        self.clss = clss
        self.children = []

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.attribute)+ ":" + repr(self.value)
        if not self.children:
            ret += " Class:" + repr(self.clss)

        ret += "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def add_node(self, node):
        self.children.append(node)

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_value(self, value):
        self.value = value

    def set_class(self, clss):
        self.clss = clss
