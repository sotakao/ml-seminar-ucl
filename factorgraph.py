import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union


class FactorGraph:
    def __init__(self, bipartitegraph: nx.Graph, variable_nodes: Dict, factor_nodes: Dict):
        """
        Args:
        ----------
        bipartitegraph: A networkx Graph representing the structure of the factor graph. This should be a bipartite graph
                        (see https://networkx.org/documentation/stable/reference/algorithms/bipartite.html)
                        whith edges e represented as e = (variable, factor).
        variable_nodes: A dictionary whose keys are variable names (as used in bipartitegraph) and values are VariableNode objects.
        factor_nodes: A dictionary whose keys are factor node labels (as used in bipartitegraph) and values are FactorNode objects.
        """
        assert nx.algorithms.bipartite.is_bipartite(bipartitegraph), "Graph must be bipartite"

        for edge in bipartitegraph.edges:
            var_key, fac_key = edge
            assert var_key in variable_nodes.keys(), f"variable {var_key} not found in variable_nodes"
            assert fac_key in factor_nodes.keys(), f"factor {fac_key} not found in factor_nodes"

        for var_key, variable in variable_nodes.items():
            assert var_key in bipartitegraph.nodes, f"variable {var_key} does not correspond to a node in bipartitegraph"
            assert isinstance(variable, VariableNode), f"variable {var_key} must be a VariableNode object"

        for fac_key, factor in factor_nodes.items():
            assert fac_key in bipartitegraph.nodes, f"factor {fac_key} does not correspond to a node in bipartitegraph"
            assert isinstance(factor, FactorNode), f"factor {fac_key} must be a FactorNode object"

        self.graph = bipartitegraph
        self.variables = variable_nodes
        self.factors = factor_nodes
        self._construct_factor_graph()

    def _construct_factor_graph(self):
        for edge in self.graph.edges:
            varID, facID = edge
            variable = self.variables[varID]
            factor = self.factors[facID]
            # Connect variable and factor nodes by an Edge object
            edge = Edge(num_states=variable.num_states)
            variable.edges[facID] = edge # Connect variable to factor <facID> by an edge
            factor.edges[varID] = edge # Connect factor to variable <varID> by an edge

    def reset(self):
        for variable in self.variables.values():
            variable.reset() # Reset messages and states to 1


class Edge:
    """
    Edge object contains messages that are sent between factor/varaiable nodes in a factor graph.
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
        # edges must be a dictionary whose values are Edge objects
        for edge_key, edge in self._edges.items():
            assert isinstance(edge, Edge), f"edge {edge_key} must be an Edge object"
        return self._edges

    @abstractmethod
    def send_message(self, outedgeID):
        """
        Abstract method to send a message to a neighbouring node <outedgeID>
        """
        pass

    def send_all_messages(self):
        """
        Send messages to all neighbours
        """
        for outedgeID in self.edges.keys():
            self.send_message(outedgeID)


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
        Compute variable-to-factor message sent to a factor node with label <outedgeID>
        by multiplying all incoming factor-to-variable messages, except for that coming from factor <outedgeID>.
        """
        variabletofactormessage = np.ones(self.num_states)
        for edgeID, edge in self.edges.items():
            if (edgeID != outedgeID):
                variabletofactormessage *= edge.factortovariablemessage
        variabletofactormessage /= variabletofactormessage.sum() # Normalising message lead to more stable results
        self.edges[outedgeID].variabletofactormessage = variabletofactormessage 

    def update(self):
        """
        Update state of the variable by multiplying together all incoming factor-to-variable messages
        """
        state = np.ones(self.num_states)
        for edge in self.edges.values():
            state *= edge.factortovariablemessage
        state /= state.sum() # Normalise
        self.state = state

    def reset(self):
        self.state = np.ones(self.num_states) # Reset state to a uniform distribution of ones
        for edge in self.edges.values():
            edge.reset() # Reset all messages to ones


class FactorNode(Node):
    """
    Class representing factor nodes in a factor graph
    """
    def __init__(self, potential: Potential):
        super().__init__()
        assert isinstance(potential, Potential)
        self.potential = potential

    def send_message(self, outedgeID):
        """
        Compute factor-to-variable message sent to a variable node with label <outedgeID>
        by filtering all incoming variable-to-factor messages (except for that coming from variable <outedgeID>)
        through the potential function.
        """
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


