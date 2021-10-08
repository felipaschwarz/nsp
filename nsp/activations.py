import numpy as np
import itertools
import torch
import copy

class Activations():
    def __init__(self, network, input):
        self.layernames = ['input']
        self.layerdescriptions = [f'Input{list(input.shape)}']
        self.layeractivations = [input]

        handles = []
        for layername, layerdescription in network.named_children():
            handles.append(layerdescription.register_forward_hook(self.get_activation(layername, layerdescription)))
        y = network(input)
        for handle in handles:
            handle.remove()

        self.postprocess()

    def get_activation(self, layername, layerdescription):
        def hook(model, input, output):
            self.layernames.append(layername)
            self.layerdescriptions.append(layerdescription)
            self.layeractivations.append(output.detach())
        return hook

    def postprocess(self):
        for index_layer, layeractivation in enumerate(self.layeractivations):
            self.layeractivations[index_layer] = self.layeractivations[index_layer].squeeze(0)

    def to_vector(self):
        vector = [layer.view(1,-1).squeeze(0).tolist() for layer in self.layeractivations]
        vector = list(itertools.chain(*vector))
        return vector

    def to_activations(self, signal):
        activations = copy.deepcopy(self)
        for index_layer, (layeractivation) in enumerate(activations.layeractivations):
            size = layeractivation.size()
            n = torch.numel(layeractivation)
            activations.layeractivations[index_layer] = torch.tensor(signal[0:n]).view(size)
            signal = signal[n:]
        return activations
