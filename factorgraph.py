import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass


class FactorGraph:
    def __init__(self, bipartitegraph: nx.Graph, variable_nodes: Dict, factor_nodes: Dict):
        assert nx.algorithms.bipartite.is_bipartite(bipartitegraph); "Graph must be bipartite"
        # Add assertions for compatibility of bipartite graphs with variable and factor dicts
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
    The Edge object contain messages sent between nodes in a factor graph
    """
    def __init__(self, num_states):
        self.num_states = num_states
        self.variabletofactormessage = np.ones(num_states)
        self.factortovariablemessage = np.ones(num_states)

    def reset(self):
        self.variabletofactormessage = np.ones(self.num_states)
        self.factortovariablemessage = np.ones(self.num_states)


class Node(ABC):
    def __init__(self):
        self.edges = {}
        # TODO: add assert statement
        super().__init__()

    @abstractmethod
    def send_message(self, outedgeID):
        pass

    def send_all_messages(self):
        for outedgeID in self.edges.keys():
            self.send_message(outedgeID)


@dataclass
class Potential:
    tensor: np.ndarray
    variables: List
    def __post_init__(self):
        assert self.tensor.ndim == len(self.variables)
        axes = np.arange(len(self.variables))
        self.axes = dict(zip(self.variables, axes))


class VariableNode(Node):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self.state = np.ones(num_states)

    def send_message(self, outedgeID):
        state = np.ones(self.num_states)
        for edgeID, edge in self.edges.items():
            if (edgeID != outedgeID):
                state *= edge.factortovariablemessage
        self.edges[outedgeID].variabletofactormessage = state

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
        self.edges[outedgeID].factortovariablemessage = M.squeeze()



