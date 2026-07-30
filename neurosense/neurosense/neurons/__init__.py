"""Neurons — from-scratch neural computation. No frameworks, only numpy."""

from .network import NeuralNetwork
from .layers import Dense, Activation
from .activations import relu, sigmoid, tanh, softmax, ACTIVATIONS
from .optimizers import SGD, Momentum, Adam
from .hebbian import HebbianLayer
from .spiking import SpikingNeuron, SpikingNetwork

__all__ = [
    "NeuralNetwork",
    "Dense",
    "Activation",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "ACTIVATIONS",
    "SGD",
    "Momentum",
    "Adam",
    "HebbianLayer",
    "SpikingNeuron",
    "SpikingNetwork",
]
