import numpy as np
import itertools
import torch
import copy

class Activations():
    """
    Holds the activation pattern generated in a neural network.

    Parameters
    ----------
    network : torch.nn.Module
        PyTorch neural network.
    input : torch.tensor
        A valid single input to the network.
        For later computations make sure to pass a single input, i.e. `batch_size = 1`.

    Attributes
    ----------
    layernames : list of str
        Names of the children modules, i.e. layer names, of the network.
    layerdescriptions : list of torch.nn
        Children modules, i.e. layer types and specifications, of the network.
    layeractivations : list of torch.tensor
        Activations of every child module, i.e. values of the activation pattern.
    """
    def __init__(self, network, input):
        self.layernames = ['input']
        self.layerdescriptions = [f'Input{list(input.shape)}']
        self.layeractivations = [input]

        handles = []
        for layername, layerdescription in network.named_children():
            handles.append(layerdescription.register_forward_hook(self._get_activation(layername, layerdescription)))
        y = network(input)
        for handle in handles:
            handle.remove()

        self._postprocess()

    def _get_activation(self, layername, layerdescription):
        def hook(model, input, output):
            self.layernames.append(layername)
            self.layerdescriptions.append(layerdescription)
            self.layeractivations.append(output.detach())
        return hook

    def _postprocess(self):
        for index_layer, layeractivation in enumerate(self.layeractivations):
            self.layeractivations[index_layer] = self.layeractivations[index_layer].squeeze(0)

    def to_vector(self):
        """
        Get and reshape `layeractivations` into a vector.

        Returns
        -------
        list
        """
        vector = [layer.view(1,-1).squeeze(0).tolist() for layer in self.layeractivations]
        vector = list(itertools.chain(*vector))
        return vector

    def to_activations(self, signal):
        """
        Create object like self but with `layeractivations` from values in signal.

        Parameters
        ----------
        signal : list, ndarray
            `layeractivations`, i.e. activation pattern, as vector.

        Returns
        -------
        Activations
        """
        activations = copy.deepcopy(self)
        for index_layer, (layeractivation) in enumerate(activations.layeractivations):
            size = layeractivation.size()
            n = torch.numel(layeractivation)
            activations.layeractivations[index_layer] = torch.tensor(signal[0:n]).view(size)
            signal = signal[n:]
        return activations
