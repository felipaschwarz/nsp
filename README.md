# Neural Network Signal Processing

nsp is a Python package for extracting and visualizing activation patterns of PyTorch neural networks. It can

- extract the computational graph of a neural network as a directed graph.
- extract the activation pattern generated by an input of a neural network.
- compute the Fourier transform of a causal signal (activation pattern) on a directed graph (neural network) based on [[1]](https://arxiv.org/pdf/2012.04358.pdf).
- visualize activation patterns and their spectrum.

### Visualization of the activation pattern
![activation](img/activation_mnist.png)
### Visualization of the Fourier coefficients
![spectrum](img/spectrum_mnist.png)

## Dependencies

Installation requires [pytorch](https://pytorch.org/get-started/locally/), [networkx](https://networkx.org/), and [numpy](https://numpy.org/). Some functions will use [scipy](https://www.scipy.org/) and/or [matplotlib](https://matplotlib.org/).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install nsp.

```bash
pip install nsp
```

## Usage

Get your network as `torch.nn.Module` and input image.

```python
import torch
import torch.nn as nn

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

network = Network()
image = torch.tensor([[[[-6, -1, -2,  5],
                        [-3, -6,  5,  4],
                        [ 2,  5, -6,  3],
                        [ 5,  0,  1, -6]]]], dtype = torch.float)

```

Extract the activation pattern.

```python
activations = nsp.Activations(network, image)
```

Extract the graph of your neural network. `NNGraph` extends [`networkx.DiGraph`](https://networkx.org/documentation/stable/reference/classes/digraph.html).

```python
graph = nsp.NNGraph(activations)
```

Transform the activation pattern into its spectrum.

```python
spectrum = graph.transform(activations)
```

Visualize the activation pattern and the spectrum. Pick your favorite `cmap_style` from [matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html).

```python
nsp.Visualizer.visualize_pattern(activations, pdf_filepath='activations.pdf', scale='layernorm', cmap_style='viridis')

nsp.Visualizer.visualize_pattern(spectrum, pdf_filepath='spectrum.pdf', scale='layernorm', cmap_style='viridis')
```

#### Visualization of the activation pattern
![activation](img/activation.png)
#### Visualization of the spectrum
![spectrum](img/spectrum.png)

For more details check out the [**tutorials**](https://github.com/felipaschwarz/nsp/tree/main/tutorials) and read the [**documentation**](https://github.com/felipaschwarz/nsp/tree/main/documentation).

## License
[MIT](https://choosealicense.com/licenses/mit/)

Developed by Felipa Schwarz (c) 2021

## References
[1]
Markus Püschel, Bastian Seifert, and Chris Wendler. Discrete signal processing on meet/join lattices. IEEE Transactions on Signal Processing, 2021.
