class Tree(object):
    def __init__(self, name='root'):
        self.name = name
        self.children = []

    def add_node(self, node):
        self.children.append(node)
