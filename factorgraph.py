import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict


class FactorGraph:
    def __init__(self, bipartitegraph: nx.Graph, variable_nodes: Dict, factor_nodes: Dict):
        """
        Args:
        ----------
        bipartitegraph: A networkx Graph representing the structure of the factor graph. This should be a bipartite graph
                        (see https://networkx.org/documentation/stable/reference/algorithms/bipartite.html)
                        whith edges represented as (variable, factor).
        variable_nodes: A dictionary whose keys are variable names (as used in bipartitegraph) and values are VariableNode objects.
        factor_nodes: A dictionary whose keys are factor node labels (as used in bipartitegraph) and values are FactorNode objects.
        """
        assert nx.algorithms.bipartite.is_bipartite(bipartitegraph), "Graph must be bipartite"

        for edge in bipartitegraph.edges:
            var_key, fac_key = edge
            assert var_key in variable_nodes.keys(), f"variable {var_key} not found in variable_nodes dictionary"
            assert fac_key in factor_nodes.keys(), f"factor {fac_key} not found in factor_nodes dictionary"

        for var_key in variable_nodes.keys():
            assert var_key in bipartitegraph.nodes, f"variable {var_key} does not correspond to a node in bipartitegraph"

        for fac_key in factor_nodes.keys():
            assert fac_key in bipartitegraph.nodes, f"factor {var_key} does not correspond to a node in bipartitegraph"

        self.graph = bipartitegraph
        self.variables = variable_nodes
        self.factors = factor_nodes
        self._construct_factor_graph()

    def _construct_factor_graph(self):
        for edge in self.graph.edges:
            varID, facID = edge
            variable = self.variables[varID]
            factor = self.factors[facID]
            edge = Edge(num_states=variable.num_states)
            variable.edges[facID] = edge
            factor.edges[varID] = edge

    def reset(self):
        for variable in self.variables.values():
            variable.reset()


class Edge:
    """
    Edge object contain messages sent between factor/varaiable nodes in a factor graph.
    """
    def __init__(self, num_states):
        self.num_states = num_states # Number of possible states a variable can take
        self.variabletofactormessage = np.ones(num_states)
        self.factortovariablemessage = np.ones(num_states)

    def reset(self):
        self.variabletofactormessage = np.ones(self.num_states)
        self.factortovariablemessage = np.ones(self.num_states)


class Node(ABC):
    """
    Base class for a node object in a factor graph.
    """
    def __init__(self):
        self._edges = {}
        super().__init__()

    @property
    def edges(self):
        for edge in self._edges.values():
            assert isinstance(edge, Edge)
        return self._edges

    @abstractmethod
    def send_message(self, outedgeID):
        """
        Abstract method to send message to a neighbouring node 'outedegID'
        """
        pass

    def send_all_messages(self):
        """
        Send message to all neighbours
        """
        for outedgeID in self.edges.keys():
            self.send_message(outedgeID)


class Potential:
    """
    The potential function consists of two objects
    """
    def __init__(self, tensor, variables):
        assert tensor.ndim == len(variables), f"Dimensions of tensor ({tensor.ndim}) must match the number of variables {len(variables)}."
        self.tensor = tensor
        axes = np.arange(len(variables))
        self.axes = dict(zip(variables, axes))


class VariableNode(Node):
    """
    Class representing variable nodes in a factor graph
    """
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self.state = np.ones(num_states)

    def send_message(self, outedgeID):
        """
        Compute variable-to-factor message sent to a factor node with label 'outedgeID'
        """
        variabletofactormessage = np.ones(self.num_states)
        for edgeID, edge in self.edges.items():
            if (edgeID != outedgeID):
                variabletofactormessage *= edge.factortovariablemessage
        variabletofactormessage /= variabletofactormessage.sum() # Normalising message yields more stable results
        self.edges[outedgeID].variabletofactormessage = variabletofactormessage 

    def update(self):
        state = np.ones(self.num_states)
        for edge in self.edges.values():
            state *= edge.factortovariablemessage
        state /= state.sum() # Normalise
        self.state = state

    def reset(self):
        self.state = np.ones(self.num_states)
        for edge in self.edges.values():
            edge.reset()


class FactorNode(Node):
    """
    Class representing factor nodes in a factor graph
    """
    def __init__(self, potential: Potential):
        super().__init__()
        self.potential = potential

    def send_message(self, outedgeID):
        M = self.potential.tensor
        for edgeID in self.edges.keys():
            if edgeID != outedgeID:
                variabletofactormessage = self.edges[edgeID].variabletofactormessage
                axis = self.potential.axes[edgeID]
                M = np.tensordot(M, variabletofactormessage, axes=(axis, 0))
                M = np.expand_dims(M, axis=axis)
        
        factortovariablemessage = M.squeeze()
        factortovariablemessage /= factortovariablemessage.sum() # Normalising message yields more stable results
        self.edges[outedgeID].factortovariablemessage = factortovariablemessage


