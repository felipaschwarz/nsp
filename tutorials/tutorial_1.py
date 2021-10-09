import nsp
import torch
import torch.nn as nn

# define a network, this one is just a dummy network and not trained
class Network_1(nn.Module):
    def __init__(self):
        super(Network_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=2)
        self.fc1 = nn.Linear(1*4*8, 8)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = x.view(-1, 1*4*8)
        x = self.fc1(x)
        return x
network = Network_1()

# define an input for the network
image = torch.tensor([[[[-6, -1, -2,  5],
                        [-3, -6,  5,  4],
                        [ 2,  5, -6,  3],
                        [ 5,  0,  1, -6]]]], dtype = torch.float)

# feed single input into the network and retrieve the activations
activations = nsp.Activations(network, image)

# get the directed graph of the network
graph = nsp.NNGraph(activations)

# compute the Fourier coefficients of the activations
spectrum = graph.transform(activations)

# visualize the activation pattern
nsp.Visualizer.visualize_pattern(activations,
                                pdf_filepath='tutorial_1/activations.pdf',
                                scale='layernorm',
                                cmap_style='viridis')

# visualize the Fourier coefficients, type='standard'
nsp.Visualizer.visualize_pattern(spectrum,
                                pdf_filepath='tutorial_1/spectrum.pdf',
                                scale='layernorm',
                                cmap_style='viridis')
