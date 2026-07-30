"""Neural primitives."""
from .activations import Activation, get_activation, leaky_relu, linear, relu, sigmoid, softmax, tanh
from .layers import DenseLayer, DropoutLayer, LayerNorm
from .network import NeuralNetwork, cross_entropy_loss, mse_loss
from .optimizers import Adam, SGD
from .spiking import LIFNeuron, SpikingNetwork, poisson_encode
from .hebbian import HebbianNetwork, HebbianSynapse, hebbian_update, oja_update

__all__ = [
    "Activation",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "linear",
    "softmax",
    "get_activation",
    "DenseLayer",
    "DropoutLayer",
    "LayerNorm",
    "NeuralNetwork",
    "mse_loss",
    "cross_entropy_loss",
    "SGD",
    "Adam",
    "LIFNeuron",
    "SpikingNetwork",
    "poisson_encode",
    "HebbianSynapse",
    "HebbianNetwork",
    "hebbian_update",
    "oja_update",
]
