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
# add the batch dimension to the input image
image = image.unsqueeze(0)

# feed single input into the network and retrieve the activations
activations = nsp.Activations(network, image)

# get the directed graph of the network
graph = nsp.NNGraph(activations)

# -------- OPTIONAL --------
# precompute the standard Fourier transform matrix
graph.get_transformer(type='standard')

# precompute the laplacian Fourier transform matrix and its LU-decomposition
graph.get_transformer(type='laplacian')

# save the graph with its precomputed transforms for later use
nsp.OutputLoader.save(graph, 'tutorial_2/data/graph.obj')

# load a graph that you previously computed
graph = nsp.OutputLoader.load('tutorial_2/data/graph.obj')
# -------- OPTIONAL --------

# compute the Fourier coefficients of the activations, default type='standard'
spectrum = graph.transform(activations)

# compute the Fourier coefficients of the activations, type='laplacian'
spectrum_lap = graph.transform(activations, type='laplacian')

# visualize the activation pattern
nsp.Visualizer.visualize_pattern(activations,
                                pdf_filepath='tutorial_2/activations.pdf',
                                scale='layernorm',
                                cmap_style='viridis')

# visualize the Fourier coefficients, type='standard'
nsp.Visualizer.visualize_pattern(spectrum,
                                pdf_filepath='tutorial_2/spectrum.pdf',
                                scale='layernorm',
                                cmap_style='viridis')

# visualize the Fourier coefficients, type='laplacian'
nsp.Visualizer.visualize_pattern(spectrum_lap,
                                pdf_filepath='tutorial_2/spectrum_lap.pdf',
                                scale='layernorm',
                                cmap_style='viridis')

# -------- OPTIONAL --------
# save the activations for later use
nsp.OutputLoader.save(activations, 'tutorial_2/activations.obj')
# save the Fourier coefficients, type='standard', for later use
nsp.OutputLoader.save(activations, 'tutorial_2/spectrum.obj')
# save the Fourier coefficients, type='laplacian', for later use
nsp.OutputLoader.save(activations, 'tutorial_2/spectrum_lap.obj')
# -------- OPTIONAL --------
