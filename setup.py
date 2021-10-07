from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.5'
DESCRIPTION = 'Neural Network Signal Processing'
LONG_DESCRIPTION = 'Analyze activation patterns of neural networks as causal signals on directed acyclic graphs (DAGs).'

# Setting up
setup(
    name="nsp",
    version=VERSION,
    author="Felipa Schwarz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'torch', 'networkx', 'matplotlib', 'torch'],
    keywords=['python', 'neural network', 'analysis', 'activations', 'activation pattern', 'fourier analysis'],
)
