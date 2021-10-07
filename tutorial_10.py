import torch
import nsp

from networks.network_1 import Network_1

image = torch.tensor([[0, 1, 2, 3],
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [3, 4, 5, 6]], dtype = torch.float).unsqueeze(0).unsqueeze(0) # batch & channel dimension

network = Network_1()
activations = nsp.Activations(network, image)
graph = nsp.NNGraph(activations)
spectrum = graph.transform(activations)

nsp.Visualizer.visualize_activations(activations,
                                pdf_filepath='output/visual/tutorial_10/activations.pdf',
                                style='layernorm',
                                cmap_style='viridis')
nsp.Visualizer.visualize_activations(spectrum,
                                pdf_filepath='output/visual/tutorial_10/spectrum.pdf',
                                style='layernorm',
                                cmap_style='viridis')
