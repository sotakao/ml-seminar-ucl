"""
@author: So Takao
"""
import numpy as np
from models import TreeGraph, FactorGraph
from copy import deepcopy


def _is_leaf(node, root_node, G: TreeGraph):
    """
    Determine whether a given node is a leaf node in a rooted tree graph G.
    ---
    :node: Node to determine whether it is leaf or not
    :root_node: Selected root node of graph
    :G: Tree graph
    """
    return (len(list(G.neighbors(node))) == 1) and (node != root_node)


def BeliefPropagation(tree: TreeGraph, root_node):
    if tree.state_type == 'mode':
        print('Switching state type to "marginal"...')
        tree.state_type = 'marginal'

    # Step 1: Find leaf nodes corresponding to the root
    leaf_nodes = []
    for x in tree.nodes():
        if _is_leaf(x, root_node, tree):
            leaf_nodes.append(x)

    # Step 2: Send messages from leaf nodes to the root node
    G = deepcopy(tree)
    current_nodes = leaf_nodes
    while current_nodes != [root_node]:
        next_nodes = []
        for node in current_nodes:
            if _is_leaf(node, root_node, G):
                # Update message
                parent_node = list(G.neighbors(node))[0] # Note: a leaf has only one parent node
                tree.send_message(node, parent_node)
                next_nodes.append(parent_node)
                G.remove_node(node)
            else:
                next_nodes.append(node)
        current_nodes = list(set(next_nodes)) # Avoid double-counting

    # Step 3: Send messages from root node to leaf nodes
    G = deepcopy(tree)
    while len(current_nodes) != 0:
        next_nodes = []
        for node in current_nodes:
            # Update message
            offsprings = list(G.neighbors(node))
            for child_node in offsprings:
                tree.send_message(node, child_node)
                next_nodes.append(child_node)
            G.remove_node(node)
        current_nodes = next_nodes
            
    # Update states
    for node in tree.nodes():
        tree.update_state(node)


def MaxProduct(tree: TreeGraph, root_node):
    if tree.state_type == 'marginal':
        print('Switching state_type to "mode"...')
        tree.state_type = 'mode'

    # Step 1: Find leaf nodes corresponding to the root
    leaf_nodes = []
    for x in tree.nodes():
        if _is_leaf(x, root_node, tree):
            leaf_nodes.append(x)

    # Step 2: Send messages from leaves to root
    G = deepcopy(tree)
    current_nodes = leaf_nodes
    while current_nodes != [root_node]:
        next_nodes = []
        for node in current_nodes:
            if _is_leaf(node, root_node, G):
                parent_node = list(G.neighbors(node))[0] # Note: there is only one parent node
                tree.send_message(node, parent_node)
                G.remove_node(node)
                next_nodes.append(parent_node)
            else:
                next_nodes.append(node)
        current_nodes = list(set(next_nodes)) # Avoid double-counting

    # Step 2.5: Compute product of all messages going into the root node
    M = tree.nodes[root_node]['potential'].tensor
    for edge in tree.in_edges(root_node):
        M *= tree.edges[edge]['message']

    # Step 3: Back-track most likely states from root to leaves
    G = deepcopy(tree)
    tree.nodes[root_node]['state'] = np.argmax(M)
    while len(current_nodes) != 0:
        next_nodes = []
        for parent_node in current_nodes:
            nodes = list(G.neighbors(parent_node))
            G.remove_node(parent_node)
            for node in nodes:
                offsprings = list(G.neighbors(node))
                tree.update_state(node, parent_node, offsprings)
                next_nodes.append(node)
        current_nodes = next_nodes


def LoopyBP(factorgraph: FactorGraph, num_iter: int=1):
    """Implements Loopy Belief Propagation with flooding schedule"""
    # Iterate LBP message passing steps
    for _ in range(num_iter):
        # Send all variable-to-factor messages
        for variable in factorgraph.variables:
            factorgraph.send_all_messages(variable)
        # Send all factor-to-variable messages
        for factor in factorgraph.factors:
            factorgraph.send_all_messages(factor)
        # Update states
        for variable in factorgraph.variables:
            factorgraph.update_state(variable)


