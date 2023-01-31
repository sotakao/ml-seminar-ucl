"""
@author: So Takao
"""
import numpy as np
import networkx as nx
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union, Hashable
from copy import deepcopy


class Potential:
    def __init__(self, tensor: np.ndarray, variables: Union[List, Hashable]):
        """
        This class represents potential functions, arising in the decomposition of joint distributions in Markov random fields.
        (Can also be used to model conditional distributions in Bayesian networks)
        We only consider the discrete variable setting here and not the continuous setting.
        In this case, a potential function can be characterised by:
        (1) a tensor, that stores the values of the potential function in tensorial form, and
        (2) an ordered list of variable names, indicating which variable corresponds to which axis of the tensor.
        
        Example:
        ----------
        Consider a potential function in two variables a and b, with K discrete states X = {1, ..., K}.
        Setting
        - tensor = P, and
        - variables = ['a', 'b']
        for some K x K tensor P, yields a potential function f : X x X -> R, such that f(a=i, b=j) = P[i,j] for i,j = 1, ..., K.

        Note: In general, when the potential is a function of N variables, each with K discrete states, then 'tensor' will be a K x ... x K (N times) tensor.
        Note: If the tensor is one-dimensional, we accept any hashable type (e.g. int, str, etc...) for 'variables', indicating the label for the unique variable.
        """
        if isinstance(variables, Hashable):
            variables = [variables]
        assert tensor.ndim == len(variables), f"Dimensions of tensor ({tensor.ndim}) must match the number of variables {len(variables)}."
        self.tensor = tensor
        self.variables = variables
        axes = np.arange(len(variables))
        self.axes = dict(zip(variables, axes))


class ProbabilisticGraphicalModel(nx.DiGraph, metaclass=ABCMeta):
    """
    Base class for probabilistic graphical models.
    This is defined as a directed graph (networkx DiGraph), whose node attributes include the states of the model
    and directed edge attributes include the messages sent between the nodes.
    """
    def __init__(self, *args, **kwargs):
        self._variables = None

        super().__init__(*args, **kwargs)

        # Build graph with node (state) and edge (message) attributes
        # ...

    @property
    def variables(self):
        """
        A list of names of all variables nodes in the graph
        """
        if self._variables is None:
            raise NotImplementedError
        else:
            return list(set(self._variables)) # Convert to set first to remove duplicates

    @property
    def states(self):
        """
        Each variable node has the attribute "state"
        """
        return dict([(x, self.nodes[x]['state']) for x in self.variables])

    @property
    def messages(self):
        """
        Each directed edge has the attribute "message"
        """
        return dict([(e, self.edges[e]['message']) for e in self.edges()])

    @abstractmethod
    def send_message(self, source_node, target_node, *args, **kwargs):
        """
        Send message from source_node to target_node
        """
        pass
    
    @abstractmethod
    def update_state(self, node, *args, **kwargs):
        """
        Update state at node
        """
        pass

    def send_all_messages(self, node):
        """
        Send messages to all neighbours of a specified node
        """
        for target_node in self.neighbors(node):
            self.send_message(node, target_node)


class TreeGraph(ProbabilisticGraphicalModel):
    """
    Class defining a tree-structured graphical model.
    Note that by the Hammersley-Clifford theorem, this can be defined completely by specifying the
    pairwise potentials and node potentials.
    """
    def __init__(self, edge_potentials: Union[Dict, List],
                       node_potentials: Union[Dict, List, None]=None,
                       state_type: str='marginal'):
        """
        Args
        ----------
        :edge_potentials: List or Dict of all pairwise potentials
        :node_potentials: List or Dict of nodewise potentials. If None, set all nodewise potentials to 1.
        :state_type: Choose either "marginal" or "mode". Specifies whether the states define the marginals of the graphical model or the mode.
                     This is dictated by whether we are running belief propagation to compute marginals, or max-product to find the mode.
        """
        super().__init__()

        assert state_type in ['marginal', 'mode'], 'Only accept state_type = "marginal" or "mode"'
        self.state_type = state_type

        # Convert edge_potentials to dict if it is a list
        if isinstance(edge_potentials, list):
            edge_potentials_ = {}
            for potential in edge_potentials:
                variables = potential.variables
                factorID = tuple(variables)
                edge_potentials_[factorID] = potential
        elif isinstance(edge_potentials, dict):
            edge_potentials_ = edge_potentials
        else:
            raise TypeError

        # Build graph from edge potentials
        self.num_states = list(edge_potentials_.values())[0].tensor.shape[0]
        self._variables = []
        for factorID, potential in edge_potentials_.items():
            (i,j) = potential.variables
            self._variables.append(i)
            self._variables.append(j)
            self.add_node(i, state=self._initialise_state())
            self.add_node(j, state=self._initialise_state())
            self.add_edge(i, j, message=np.ones(self.num_states),
                                ID=factorID,
                                potential=potential)
            self.add_edge(j, i, message=np.ones(self.num_states),
                                ID=factorID,
                                potential=potential)

        # assert tree structure of graph
        assert nx.is_tree(nx.Graph(self)), "The underlying graph must be a tree"

        # Add node potentials to every node
        node_potentials_ = dict([(node, Potential(np.ones(self.num_states), node)) for node in self.nodes()])
        if node_potentials is None:
            pass
        elif isinstance(node_potentials, list):
            for potential in node_potentials:
                variables = potential.variables
                node = variables[0]
                node_potentials_[node] = potential
        elif isinstance(node_potentials, dict):
            for potential in node_potentials.values():
                variables = potential.variables
                node = variables[0]
                node_potentials_[node] = potential
        else:
            raise TypeError

        node_attrs = dict([(node, {'potential': potential}) for node, potential in node_potentials_.items()])
        nx.set_node_attributes(self, node_attrs)

    def _initialise_state(self):
        if self.state_type == 'marginal':
            return np.ones(self.num_states)
        else:
            return None

    def send_message(self, source_node, target_node):
        edge_potential = self.edges[(target_node, source_node)]['potential']
        source_axis = edge_potential.axes[source_node] # Axis corresponding to the source node (0 or 1)
        target_axis = edge_potential.axes[target_node] # Axis corresponding to the target node (0 or 1)
        node_potential = self.nodes[source_node]['potential']
        prod_msgs = deepcopy(node_potential.tensor)
        for e in self.in_edges(source_node):
            if e != (target_node, source_node):
                prod_msgs *= self.edges[e]['message']
        if self.state_type == 'marginal': # For marginal computation (used in belief propagation)
            tensor = edge_potential.tensor
            message = np.tensordot(tensor, prod_msgs, axes=(source_axis, 0))
        elif self.state_type == 'mode': # For mode computation (used in max-product)
            tensor = edge_potential.tensor
            prod_msgs = np.expand_dims(prod_msgs, axis=target_axis)
            message = np.max(tensor * prod_msgs, axis=source_axis)
        self.edges[(source_node, target_node)]['message'] = message

    def update_state(self, node, parent_node=None, offsprings=None):
        if self.state_type == 'marginal': # Marginal updates (used in belief propagation)
            node_potential = self.nodes[node]['potential']
            state = deepcopy(node_potential.tensor)
            for e in self.in_edges(node):
                state *= self.edges[e]['message']
            state /= state.sum()
            self.nodes[node]['state'] = state

        elif self.state_type == 'mode': # Mode updates (used in max-product)
            x = self.nodes[parent_node]['state']
            edge_potential = self.edges[(parent_node, node)]['potential']
            node_potential = self.nodes[node]['potential']
            prod_msgs = deepcopy(node_potential.tensor)
            for child_node in offsprings:
                prod_msgs *= self.edges[(child_node, node)]['message']
            parent_axis = edge_potential.axes[parent_node] # Axis corresponding to the parent node (0 or 1)
            if parent_axis == 0:
                vector = edge_potential.tensor[x,:]
            elif parent_axis == 1:
                vector = edge_potential.tensor[:,x]
            self.nodes[node]['state'] = np.argmax(vector * prod_msgs) # TODO: change


class GaussianTreeGraph(TreeGraph):
    def __init__(self, edge_potentials: Union[Dict, List], node_potentials: Union[Dict, List, None]=None):
        super().__init__(edge_potentials, node_potentials)

    def send_message(self, source_node, target_node):
        ...

    def update_state(self, node):
        ...


class FactorGraph(ProbabilisticGraphicalModel):
    """
    Class defining a factor graph.
    This is characterised completely by the potentials in the factorisation given by the Hammersley-Clifford theorem.
    """
    def __init__(self, potentials: Union[Dict, List]):
        """
        Args
        ----------
        :potentials: List or Dict of Potentials
        """
        super().__init__()

        if isinstance(potentials, list):
            potentials_ = {}
            for potential in potentials:
                variables = potential.variables
                factorID = tuple(variables)
                potentials_[factorID] = potential
        elif isinstance(potentials, dict):
            potentials_ = potentials
        else:
            raise TypeError

        self.num_states = list(potentials_.values())[0].tensor.shape[0]
        self._variables = []
        self.factors = []
        for factorID, potential in potentials_.items():
            variables = potential.variables
            for variableID in variables:
                self._variables.append(variableID)
                self.factors.append(factorID)
                self.add_node(variableID, node_type='variable', state=np.ones(self.num_states))
                self.add_node(factorID, node_type='factor', potential=potential)
                self.add_edge(variableID, factorID, message=np.ones(self.num_states))
                self.add_edge(factorID, variableID, message=np.ones(self.num_states))

        assert nx.is_bipartite(self), "FactorGraph must be bipartite"

    def send_message(self, source_node, target_node):
        if self.nodes[source_node]['node_type'] == 'variable': # Send message from variable node to factor node
            variabletofactormessage = np.ones(self.num_states)
            for node in self.neighbors(source_node):
                if (node != target_node):
                    variabletofactormessage *= self.edges[(node, source_node)]['message']
            variabletofactormessage /= variabletofactormessage.sum() # Normalising message lead to more stable results
            self.edges[(source_node, target_node)]['message'] = variabletofactormessage 

        elif self.nodes[source_node]['node_type'] == 'factor': # Send message from factor node to variable node
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

    def update_state(self, node):
        assert self.nodes[node]['node_type'] == 'variable', 'State update can only be applied to variable nodes'
        state = np.ones(self.num_states)
        for edge in self.in_edges(node):
            state *= self.edges[edge]['message'] # Multiply factor-to-variable message
        state /= state.sum() # Normalise
        self.nodes[node]['state'] = state


