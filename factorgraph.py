import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union


class Potential:
    def __init__(self, tensor: np.ndarray, variables: Union[List, Tuple]):
        """
        This class represents potential functions, arising in the decomposition of joint distributions in Markov random fields.
        (Can also be used to model conditional distributions in Bayesian networks)
        We only consider the discrete variable setting here and not the continuous setting.
        In this case, a potential function can be characterised by:
        (1) a tensor, that stores the values of the potential function in tensorial form, and
        (2) an ordered list/tuple of variable names, indicating which variable corresponds to which axis of the tensor.
        
        Example:
        ----------
        Consider a potential function in two variables a and b, with K discrete states X = {1, ..., K}.
        Setting
        - tensor = P, and
        - variables = ['a', 'b']
        for some K x K tensor P, yields a potential function f : X x X -> R, such that f(a=i, b=j) = P[i,j] for i,j = 1, ..., K.

        Note: In general, when the potential is a function of N variables, each with K discrete states, then P will be a K x ... x K (N times) tensor.

        """
        assert tensor.ndim == len(variables), f"Dimensions of tensor ({tensor.ndim}) must match the number of variables {len(variables)}."
        self.tensor = tensor
        self.variables = variables
        axes = np.arange(len(variables))
        self.axes = dict(zip(variables, axes))


# TODO: maybe consider a base class for PGM?
class TreeGraph(nx.DiGraph):
    def __init__(self, edge_potentials, node_potentials=None):
        self.num_states = list(edge_potentials.values())[0].tensor.shape[0]

        # assert tree
        edges = []
        for (i,j) in edge_potentials.keys():
            edges.append((i,j))
            edges.append((j,i))

        super().__init__(edges)
        assert nx.is_tree(nx.Graph(self)), "The underlying graph must be a tree"

        # Complete edge potentials
        for edge in edges:
            if edge not in list(edge_potentials.keys()):
                (i,j) = edge
                tensor = edge_potentials[(j,i)].tensor
                edge_potentials[(i,j)] = Potential(tensor.T, (i,j))

        self.edge_potentials = edge_potentials

        # Initialise node potentials
        if node_potentials is None:
            node_potentials = dict([(node, Potential(np.ones(self.num_states), [node])) for node in self.nodes()])
        
        node_attrs = dict([(node, {'state': np.ones(self.num_states), 'potential': potential}) 
                          for node, potential in node_potentials.items()])

        edge_attrs = dict([(edge, {'message': np.ones(self.num_states), 'potential': potential})
                          for edge, potential in edge_potentials.items()])

        nx.set_node_attributes(self, node_attrs)
        nx.set_edge_attributes(self, edge_attrs)

    @property
    def states(self):
        return dict([(x, self.nodes[x]['state']) for x in self.nodes()])


class FactorGraph(nx.DiGraph):
    def __init__(self, potentials):
        """
        Args:
        ----------
        :potentials: List of Potentials
        """
        super().__init__()

        self.num_states = potentials[0].tensor.shape[0]
        self.variables = []
        self.factors = []
        for potential in potentials:
            variables = potential.variables
            factorID = tuple(variables)
            for variableID in variables:
                self.variables.append(variableID)
                self.factors.append(factorID)
                self.add_node(variableID, node_type='variable', state=np.ones(self.num_states))
                self.add_node(factorID, node_type='factor', potential=potential)
                self.add_edge(variableID, factorID, message=np.ones(self.num_states))
                self.add_edge(factorID, variableID, message=np.ones(self.num_states))

        assert nx.is_bipartite(self), "FactorGraph must be bipartite"

    def send_message(self, source_node, target_node):
        if self.nodes[source_node]['node_type'] == 'variable':
            variabletofactormessage = np.ones(self.num_states)
            for node in self.neighbors(source_node):
                if (node != target_node):
                    variabletofactormessage *= self.edges[(node, source_node)]['message']
            variabletofactormessage /= variabletofactormessage.sum() # Normalising message lead to more stable results
            self.edges[(source_node, target_node)]['message'] = variabletofactormessage 

        elif self.nodes[source_node]['node_type'] == 'factor':
            potential = self.nodes[source_node]['potential']
            M = potential.tensor
            for node in self.neighbors(source_node):
                if node != target_node:
                    variabletofactormessage = self.edges[(node, source_node)]['message']
                    axis = potential.axes[node]
                    M = np.tensordot(M, variabletofactormessage, axes=(axis, 0))
                    M = np.expand_dims(M, axis=axis)
            factortovariablemessage = M.squeeze()
            factortovariablemessage /= factortovariablemessage.sum() # Normalising message yields more stable results
            self.edges[(source_node, target_node)]['message'] = factortovariablemessage

    def send_all_messages(self, node):
        for target_node in self.neighbors(node):
            self.send_message(node, target_node)

    def update(self, node):
        assert self.nodes[node]['node_type'] == 'variable', 'State update can only be applied to variable nodes'
        state = np.ones(self.num_states)
        for edge in self.in_edges(node):
            state *= self.edges[edge]['message'] # Multiply factor-to-variable message
        state /= state.sum() # Normalise
        self.nodes[node]['state'] = state

    @property
    def states(self):
        return dict([(x, self.nodes[x]['state']) for x in self.nodes() if self.nodes[x]['node_type'] == 'variable'])


