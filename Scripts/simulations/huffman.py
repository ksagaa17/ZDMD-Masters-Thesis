""""
Author: Andreas J. Fuglsig

This script containts the functions required to produce a Huffman code for a
 given set of frequencies/probabilites.
Source: https://stackoverflow.com/questions/11587044/how-can-i-create-a-tree-for-huffman-encoding-and-decoding
"""

import queue
import numpy as np


class HuffmanNode(object):
    def __init__(self, left=None, right=None, root=None):
        self.left = left
        self.right = right
        self.root = root     # Why?  Not needed for anything.

    def children(self):
        return((self.left, self.right))


def create_tree(frequencies, labels=None):
    if labels is None:
        labels = np.arange(0, frequencies.size)
    P = queue.PriorityQueue()
    for value, lab in zip(frequencies, labels):  # 1. Create a leaf node for each symbol
        P.put((value, lab))              # and add it to the priority queue
    while P.qsize() > 1:                 # 2. While there is more than one node
        left, right = P.get(), P.get()   # 2a. remove two highest nodes
        node = HuffmanNode(left, right)  # 2b. create internal node with children
        P.put((left[0]+right[0], node))  # 2c. add new node to queue
    return P.get()                       # 3. tree is complete - return root node


def create_tree2(frequencies, labels=None):
    """By using sorted in stead of priority queue it is possible to handle
    equal probabilities."""
    if labels is None:
        labels = np.arange(0, len(frequencies))
    Q = []
    for value, lab in zip(frequencies, labels):  # 1. Create a leaf node
                                                 # for each symbol
        Q.append((value, lab))               # and add it to the priority queue
    sorted_Q = sorted(Q, key=lambda x: x[0], reverse=True)
    while len(sorted_Q) > 1:             # 2. While there is more than one node
        left, right = sorted_Q.pop(), sorted_Q.pop()  # 2a. remove two highest nodes
        node = HuffmanNode(left, right) # 2b. create internal node with children
        sorted_Q.append((left[0]+right[0], node)) # 2c. add new node to queue
        sorted_Q = sorted(sorted_Q, key=lambda x: x[0], reverse=True)  # Resort the queue
    return sorted_Q.pop()


# Recursively walk the tree down to the leaves,
# assigning a code value to each symbol
def walk_tree(node, prefix="", code={}):
    if isinstance(node[1].left[1], HuffmanNode):
        walk_tree(node[1].left, prefix+"0", code)
    else:
        code[node[1].left[1]] = prefix+"0"
    if isinstance(node[1].right[1], HuffmanNode):
        walk_tree(node[1].right, prefix+"1", code)
    else:
        code[node[1].right[1]] = prefix+"1"
    return(code)


def huffman_coder(frequencies, labels=None, preifx=""):
    root = create_tree(frequencies, labels)
    code = walk_tree(root, code={})
    return code


def huffman2(frequencies, labels=None, preifx=""):
    root = create_tree2(frequencies, labels)
    code = walk_tree(root, code={})
    return code


def get_code_length(code, keep_labels=False):
    if keep_labels:
        length = {}
    else:
        length = np.empty(len(code.values()))

    for key in code.keys():
        length[key] = len(code[key])
    return length


if __name__ == '__main__':
    freq = [
        (8.167, 'a'), (1.492, 'b'), (2.782, 'c'), (4.253, 'd'),
        (12.702, 'e'),(2.228, 'f'), (2.015, 'g'), (6.094, 'h'),
        (6.966, 'i'), (0.153, 'j'), (0.747, 'k'), (4.025, 'l'),
        (2.406, 'm'), (6.749, 'n'), (7.507, 'o'), (1.929, 'p'),
        (0.095, 'q'), (5.987, 'r'), (6.327, 's'), (9.056, 't'),
        (2.758, 'u'), (1.037, 'v'), (2.365, 'w'), (0.150, 'x'),
        (1.974, 'y'), (0.074, 'z') ]

    labels = [x[1] for x in freq]
    probs = [x[0] for x in freq]

    # root_node = create_tree(probs, labels)
    # print(root_node)

    # code = walk_tree(root_node)
    code = huffman_coder(probs, labels)
    root = create_tree2(probs, labels)
    code2 = huffman2(probs, labels)
    len_code = get_code_length(code, keep_labels=True)

    for i in sorted(freq, reverse=True):
        assert(code[i[1]] == code2[i[1]])
        print(i[1], '{:6.2f}'.format(i[0]), code[i[1]])

    pn = np.array([0.012,0.984,0.004])
    # root = create_tree(pn)
    # new_code = walk_tree(root, code={})
    new_code = huffman_coder(pn)
    len_new_code = get_code_length(new_code)
    
    pn = np.array([0.8,0.1,0.1])
    # root = create_tree(pn)
    # new_code = walk_tree(root, code={})
    new_code = huffman2(pn)
    len_new_code = get_code_length(new_code)
