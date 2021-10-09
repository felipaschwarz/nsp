import nsp
import torch
import torchvision

# import the network class
from networks.network_mnist_1 import Network_mnist_1

# load the pretrained network
network = torch.load('networks/network_mnist_1.pth')

# get the input images
testset = torchvision.datasets.MNIST(root='./datasets',
                                    train=False,
                                    download=True,
                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

# get one input image
image, label = testset[0]
# add the batch dimension from the image
image = image.unsqueeze(0)

# feed single input into the network and retrieve the activations
activations = nsp.Activations(network, image)

# get the directed graph of the network
graph = nsp.NNGraph(activations)

# precompute the Fourier transform matrix, optional
graph.get_transformer(type='standard')
# precompute the Fourier transform matrix, optional
graph.get_transformer(type='laplacian')

# compute the Fourier coefficients of the activations, default type='standard'
spectrum = graph.transform(activations)

# compute the Fourier coefficients of the activations, type='laplacian'
spectrum_lap = graph.transform(activations, type='laplacian')

# visualize the activations
nsp.Visualizer.visualize_activations(activations,
                                pdf_filepath='outputs/visual/tutorial_2/activations2.pdf',
                                style='layernorm',
                                cmap_style='viridis')

# visualize the Fourier coefficients, type='standard'
nsp.Visualizer.visualize_activations(spectrum,
                                pdf_filepath='outputs/visual/tutorial_2/spectrum2.pdf',
                                style='layernorm',
                                cmap_style='viridis')

# visualize the Fourier coefficients, type='laplacian'
nsp.Visualizer.visualize_activations(spectrum_lap,
                                pdf_filepath='outputs/visual/tutorial_2/spectrum_lap2.pdf',
                                style='layernorm',
                                cmap_style='viridis')

# save the graph with its transform for later, optional
nsp.OutputLoader.save(graph, 'outputs/data/graph.obj')
# load a graph that you previously computed, optional
graph = nsp.OutputLoader.load('outputs/data/graph.obj')
# save the activations for later use, optional
nsp.OutputLoader.save(activations, 'outputs/data/activations.obj')
# save the Fourier coefficients for later use, optional
nsp.OutputLoader.save(activations, 'outputs/data/activations.obj')
