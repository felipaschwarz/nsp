import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import copy

from nsp.transformer import Transformer

class NNGraph(nx.DiGraph):
    # INCL set to True: includes edge even if weight = 0.
    # Otherwise only edges with weight !=0 are added.
    # Especially, INCL = False will only have one incoming edge per MaxPool Activation
    INCL = True

    def __init__(self, activations):
        super().__init__()
        self.add_nodes_from_activations(activations)
        self.add_edges_from_jacobian(activations)
        self.inv_F = None

    def add_nodes_from_activations(self, activations):
        for index_layer, layeractivation in enumerate(activations.layeractivations):
            for index_generator in np.ndindex(layeractivation.shape):
                self.add_node((index_layer, *index_generator))

    def add_edges_from_jacobian(self, activations):
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

    def compute_transformer(self, type='standard'):
        self.inv_F = Transformer.compute_inv_fourier_matrix(self, type=type)

    def transform(self, activations, type='standard'):
        if self.inv_F is None:
            self.compute_transformer()
        return Transformer.compute_fourier_activations(activations=activations, inv_F=self.inv_F, type=type, lu_piv=None)

    def node_id(self, node):
        nodes = list(self.nodes)
        return nodes.index(node)
