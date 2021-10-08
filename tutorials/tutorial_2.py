import nsp
import torch
import torchvision

from networks.network_mnist_00 import Network_mnist_00
from networks.network_mnist_0 import Network_mnist_0
from networks.network_mnist_1 import Network_mnist_1
from networks.network_mnist_2 import Network_mnist_2
from networks.network_cifar_1 import Network_cifar_1
from networks.network_fashionmnist_0 import Network_fashionmnist_0
from networks.network_fashionmnist_1 import Network_fashionmnist_1

network = torch.load('networks/network_mnist_1.pth')
testset = torchvision.datasets.MNIST(root='./datasets',
                                    train=False,
                                    download=True,
                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
image, label = testset[2]
image = image.unsqueeze(0) # remove batch dimension

activations = nsp.Activations(network, image)
#graph = nsp.NNGraph(activations)
#graph.compute_transformer()
#nsp.OutputLoader.save(graph, 'outputs/data/graph.obj')
graph = nsp.OutputLoader.load('outputs/data/graph.obj')


spectrum = graph.transform(activations)

nsp.Visualizer.visualize_activations(activations,
                                pdf_filepath='outputs/visual/tutorial_2/activations2.pdf',
                                style='layernorm',
                                cmap_style='viridis')
nsp.Visualizer.visualize_activations(spectrum,
                                pdf_filepath='outputs/visual/tutorial_2/spectrum2.pdf',
                                style='layernorm',
                                cmap_style='viridis')


nsp.OutputLoader.save(activations, 'outputs/data/activations.obj')
nsp.OutputLoader.save(spectrum, 'outputs/data/spectrum.obj')
