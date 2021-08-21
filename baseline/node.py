class Node(object):
    def __init__(self, attribute, value=None, cost=None, father=None):
        self.childs = []
        self.attribute = int(attribute)
        self.father = father
        self.value = value
        self.cost = cost
        self.depth = 1
        if father:
            self.cost = self.cost + self.father.get_cost()
            self.depth = self.depth + self.father.get_depth()
        self.leaf = True
        self.identity = {}
        self.identity["Attribute"] = self.attribute
        self.identity["Value"] = self.value
        self.identity["Cost"] = self.cost

    def __str__(self):
        self.identity["Is-leaf"] = self.leaf
        return "Node({})".format(self.identity)

    def is_leaf(self):
        return self.leaf

    def add_child(self, node):
        self.leaf = False
        self.childs += [node]

    def get_father(self):
        return self.father

    def get_childs(self):
        return self.childs

    def get_attribute(self):
        return self.attribute

    def get_value(self):
        return self.value

    def get_cost(self):
        return self.cost

    def get_value_plus_cost(self):
        return self.value + self.cost

    def get_depth(self):
        return self.depth

    def __eq__(self, node):
        if not isinstance(node, Node):
            print("Node: comparing two different classes")
            print("Wrong input: {}".format(str(node)))
            return None

        if node.get_attribute() == self.get_attribute():
            return True
        return False

    def __hash__(self):
        return self.get_attribute()

    def __ne__(self, node):
        return not node == self

    def __lt__(self, node):
        return self.get_value_plus_cost() < node.get_value_plus_cost()
