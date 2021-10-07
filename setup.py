from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Neural Network Signal Processing'
LONG_DESCRIPTION = 'Analyse activation patterns of neural networks as causal signals on directed acyclic graphs (DAGs).'

# Setting up
setup(
    name="nsp",
    version=VERSION,
    author="Felipa Schwarz",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'torch', 'itertools', 'networkx', 'copy', 'pickle', 'matplotlib', 'torch'],
    keywords=['python', 'neural network', 'analysis', 'activations', 'activation pattern', 'fourier analysis'],
)
