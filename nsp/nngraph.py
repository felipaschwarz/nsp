import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import copy

from .transformer import Transformer

class NNGraph(nx.DiGraph):
    """Directed graph representing in the computational graph structure of a Pytorch neural network.

    Parameters
    ----------

    activations : Activations
        Any sample activation pattern generated by the network under observation.
        The resulting `NNGraph` is independent of the activation values contained
        in the specific activation pattern, i.e. `activation.layeractivations`.
    INCL : boolean, default True
        If `True`, it includes edges in the NNGraph even if the weight of the edge
        in the neural network is 0.
        If `False`, only edges with weight unequal to 0 are added to the graph.
        Especially, `INCL = False` will only have one incoming edge per activation
        in  a `torch.nn.MaxPool` layer.

    Attributes
    ----------
    inv_F : scipy.sparse.tril(A, format='csr'), default None
        Inverse Fourier Transform matrix w.r.t. https://acl.inf.ethz.ch/research/ASP/.
        Is `None` until the first invocation of `NNGraph.transform(activations, type='standard')`
        or can be explicitly assigned by calling `NNGraph.compute_transformer(type='standard')`.
    lu_piv : numpy.ndarray, numpy.ndarray
        LU-decomposition of the inverse Fourier Transform matrix w.r.t. https://arxiv.org/pdf/1211.0053.pdf.
        Is `None` until the first invocation of `NNGraph.transform(activations, type='laplacian')`
        or can be explicitly assigned by calling `NNGraph.compute_transformer(type='laplacian')`.
    """
    # INCL set to True: includes edge even if weight = 0.
    # Otherwise only edges with weight !=0 are added.
    # Especially, INCL = False will only have one incoming edge per MaxPool Activation
    def __init__(self, activations, INCL=True):
        super().__init__()
        self.INCL = INCL
        self._add_nodes_from_activations(activations)
        self._add_edges_from_jacobian(activations)
        self.inv_F = None
        self.lu_piv = None

    def _add_nodes_from_activations(self, activations):
        for index_layer, layeractivation in enumerate(activations.layeractivations):
            for index_generator in np.ndindex(layeractivation.shape):
                self.add_node((index_layer, *index_generator))

    def _add_edges_from_jacobian(self, activations):
    # only accept nn.Linear layers if input is flattened
        for index_previous_layer, (layeractivation, layerdescription) in enumerate(zip(activations.layeractivations[1:], activations.layerdescriptions[1:])):
            previous_layeractivation = activations.layeractivations[index_previous_layer]

            if (isinstance(layerdescription, nn.Linear)):
                # assume only flattened inputs x are passed to Linear layers (x = x.view(x.size(0), -1))
                for index_generator_from_node in np.ndindex(previous_layeractivation.shape):
                        for index_generator_to_node in np.ndindex(layeractivation.shape):
                            from_node = (index_previous_layer, *index_generator_from_node)
                            to_node = (index_previous_layer + 1, *index_generator_to_node)
                            assert self.has_node(from_node), f'from_node does not exist: {from_node}'
                            assert self.has_node(to_node), f'to_node does not exist: {to_node}'
                            self.add_edge(from_node, to_node)

            else:
                if (self.INCL): # INCLUDE zero-weight edges
                    if (isinstance(layerdescription, nn.MaxPool2d)):
                        assert (layerdescription.ceil_mode == False)
                        n_channels = previous_layeractivation.shape[0]
                        layerdescription_for_jacobian = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                                                    kernel_size=layerdescription.kernel_size,
                                                                    stride=layerdescription.stride,
                                                                    padding=layerdescription.padding,
                                                                    dilation=layerdescription.dilation,
                                                                    groups=n_channels) # important
                    else:
                      layerdescription_for_jacobian = copy.deepcopy(layerdescription)
                    # draw edge in covergraph even if weight = 0
                    # i.e. ensure jacobian is either 1 (is input) or 0 (is not input) regardless of weight of this input
                    if not (isinstance(layerdescription, nn.AvgPool2d)): # change to if instance has weight attribute then fill with 1
                        layerdescription_for_jacobian.weight.data.fill_(1)
                    # previous_layeractivation.unsqueeze(0) adds the batch dimension
                    jacobian = torch.autograd.functional.jacobian(layerdescription_for_jacobian, previous_layeractivation.clone().detach().unsqueeze(0))
                else: # EXCLUDE zero-weight edges
                    jacobian = torch.autograd.functional.jacobian(layerdescription, previous_layeractivation)
                for index_generator in np.ndindex(jacobian.size()):
                    to_batch, to_channel, to_i, to_j, from_batch, from_channel, from_i, from_j = index_generator
                    if jacobian[index_generator]:
                        from_node = (index_previous_layer, from_channel, from_i, from_j)
                        to_node = (index_previous_layer + 1, to_channel, to_i, to_j)
                        assert self.has_node(from_node), f'from_node does not exist: {from_node}'
                        assert self.has_node(to_node), f'to_node does not exist: {to_node}'
                        self.add_edge(from_node, to_node)

    def get_transformer(self, type='standard'):
        """
        Compute the inverse Fourier transformations for the graph.

        Parameters
        ----------
        type : {'standard', 'laplacian'}, default 'standard'
            - 'standard' : computes the inverse Fourier Transform matrix w.r.t.
                            https://acl.inf.ethz.ch/research/ASP/ and puts it in `self.inv_F.
            - 'laplacian' : computes the LU-Decomposition of the inverse Fourier Transform matrix w.r.t.
                            https://arxiv.org/pdf/1211.0053.pdf and puts it in `self.lu_piv`.

        Returns
        -------
        scipy.sparse.tril(A, format='csr')
            if `type='standard'`.
        numpy.ndarray, numpy.ndarray
            if `type='laplacian'`.
        """
        if type =='standard':
            self.inv_F = Transformer.get_inv_fourier_matrix(self, type=type)
            return self.inv_F
        elif type =='laplacian':
            self.lu_piv = Transformer.get_inv_fourier_matrix(self, type=type)
            return self.lu_piv
        else:
            raise NotImplementedError

    def transform(self, activations, type='standard'):
        """
        Compute the Fourier coefficients, or spectrum, of `Activations`.

        It creates a copy of `activations` with the Fourier coefficients `activations.layeractivations`.

        Parameters
        ----------
        activations : Activation
            Activation to transfrom.
        type : {'standard', 'laplacian'}, default 'standard'
            - 'standard' : computes the inverse Fourier Transform matrix w.r.t.
                            https://acl.inf.ethz.ch/research/ASP/ and puts it in `self.inv_F.
            - 'laplacian' : computes the LU-Decomposition of the inverse Fourier Transform matrix w.r.t.
                            https://arxiv.org/pdf/1211.0053.pdf and puts it in `self.lu_piv`

        Returns
        -------
        Activations
        """
        if type =='standard':
            if self.inv_F is None:
                self.get_transformer(type=type)
            return Transformer.get_fourier_coefficients(activations=activations, inv_F=self.inv_F, type=type, lu_piv=None)
        elif type =='laplacian':
            if self.lu_piv is None:
                self.lu_piv = Transformer.get_inv_fourier_matrix(self, type=type)
            return Transformer.get_fourier_coefficients(activations=activations, inv_F=None, type=type, lu_piv=self.lu_piv)
        else:
            raise NotImplementedError

    def node_id(self, node):
        """
        Get the index of a node in the total order of all nodes.

        Returns
        -------
        int
        """
        nodes = list(self.nodes)
        return nodes.index(node)
