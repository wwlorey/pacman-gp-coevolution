import math
import random


STARTING_TREE_SIZE = 64


class TreeNode:
    def __init__(self, index, value):
        """Initializes the TreeNode class."""
        self.value = value

        # Note: index represents this node's index in the Tree list
        self.index = index
    

    def __str__(self):
        return str(self.value)
        

class Tree:
    def __init__(self, config, root_value=None):
        """Initializes the (binary) Tree class."""
        self.list = [TreeNode(index, None) for index in range(STARTING_TREE_SIZE)]
        self.list[0] = TreeNode(0, root_value)

    
    def __str__(self):
        ret = '['
        for item in self.list:
            ret += str(item) + ', '
        
        ret += ']'

        return ret
    

    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        return self.list[index]


    def __setitem__(self, index, value):
        self.list[index] = value


    def get_parent(self, node):
        """Returns the parent node of the given node.

        If the parent node does not exist, None is returned.
        """
        parent_index = node.index // 2

        if parent_index < 0:
            return TreeNode(-1, None)
        
        return self.list[parent_index]
    

    def get_left_child(self, node):
        """Returns the left child of the given node.

        If the left child does not exist, None is returned.
        """
        left_child_index = self.get_left_child_index(node)

        if left_child_index >= len(self.list) or not self.list[left_child_index]:
            return TreeNode(-1, None)
        
        return self.list[left_child_index]


    def get_left_child_index(self, node):
        """Returns the left child index of the given node. """
        return 2 * (1 + node.index) - 1


    def get_right_child(self, node):
        """Returns the right child of the given node.

        If the right child does not exist, None is returned.
        """
        right_child_index = self.get_right_child_index(node)

        if right_child_index >= len(self.list) or not self.list[right_child_index]:
            return TreeNode(-1, None)
        
        return self.list[right_child_index]


    def get_right_child_index(self, node):
        """Returns the right child index of the given node."""
        return 2 * (node.index + 1)


    def get_height(self):
        """Returns the maximum depth (height) of this tree."""
        counter = 0

        for n in self.list[::-1]:
            if n.value != None:
                return math.ceil(math.log2(len(self.list) - counter))

            counter += 1
        
        return 1


    def add_node_at_index(self, index, value):
        """Creates a new TreeNode object in self.list at index.

        Allocates more space for the tree list as needed.
        """
        if index >= len(self.list):
            # Allocate more space for this list
            self.list = self.list + [TreeNode(index, None) for index in range(len(self.list), len(self.list) * 2)]

        self.list[index].value = value


    def add_node_left(self, parent_node, value=None):
        """Adds a new node with provided value to the left node of parent_node."""
        self.add_node_at_index(self.get_left_child_index(parent_node), value)

        
    def add_node_right(self, parent_node, value=None):
        """Adds a new node with provided value to the right node of parent_node."""
        self.add_node_at_index(self.get_right_child_index(parent_node), value) 


    def get_root(self):
        """Returns the root node of the tree."""
        return self.list[0]


    def is_leaf(self, node):
        """Returns if this node is a leaf node (i.e. it has no children)."""
        return not self.get_left_child(node).value or not self.get_right_child(node).value
    