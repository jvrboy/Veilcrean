"""NeuralNetwork — a complete trainable feed-forward network from scratch.

Backpropagation, mini-batches, multiple losses and optimizers,
save/load to JSON. No frameworks, only numpy.
"""

from __future__ import annotations

import json

import numpy as np

from .layers import Dense, Activation, Dropout, Layer
from .activations import softmax
from .optimizers import Adam, Optimizer


# ---------------------------- losses ---------------------------------- #

def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    diff = pred - target
    return float(np.mean(diff**2)), 2 * diff / diff.size


def cross_entropy_loss(logits: np.ndarray,
                       target: np.ndarray) -> tuple[float, np.ndarray]:
    """Softmax cross-entropy. target is one-hot (batch x classes)."""
    probs = softmax(logits)
    eps = 1e-12
    loss = -float(np.mean(np.sum(target * np.log(probs + eps), axis=1)))
    grad = (probs - target) / len(logits)
    return loss, grad


LOSSES = {"mse": mse_loss, "cross_entropy": cross_entropy_loss}


# ---------------------------- network --------------------------------- #

class NeuralNetwork:
    """Build, train, and use a feed-forward neural network.

    >>> net = NeuralNetwork([4, 16, 3], activation="relu", output="softmax")
    >>> net.train(X, Y_onehot, epochs=200)
    >>> net.predict_class(x)
    """

    def __init__(self, layer_sizes: list[int] | None = None,
                 activation: str = "relu", output: str = "softmax",
                 seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.layers: list[Layer] = []
        self.output_kind = output
        self.loss_name = "cross_entropy" if output == "softmax" else "mse"
        self._sizes = layer_sizes or []
        self._activation = activation
        if layer_sizes:
            for i in range(len(layer_sizes) - 1):
                self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1],
                                         rng=self.rng))
                if i < len(layer_sizes) - 2:
                    self.layers.append(Activation(activation))
                elif output not in ("softmax", "linear"):
                    self.layers.append(Activation(output))

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(x)
        if self.output_kind == "softmax":
            out = softmax(out)
        return out

    def predict_class(self, x: np.ndarray) -> int:
        return int(np.argmax(self.predict(x), axis=1)[0])

    # ------------------------------------------------------------------ #
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100,
              batch_size: int = 32, optimizer: Optimizer | None = None,
              loss: str | None = None, verbose: bool = False,
              shuffle: bool = True) -> list[float]:
        """Train with mini-batch gradient descent. Returns loss history."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        opt = optimizer or Adam(lr=0.005)
        loss_fn = LOSSES[loss or self.loss_name]
        history = []
        n = len(X)
        for epoch in range(epochs):
            if shuffle:
                order = self.rng.permutation(n)
                X, Y = X[order], Y[order]
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                xb, yb = X[start:start + batch_size], Y[start:start + batch_size]
                pred = self.forward(xb)
                loss_value, grad = loss_fn(pred, yb)
                epoch_loss += loss_value
                batches += 1
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                for layer in self.layers:
                    if layer.params:
                        opt.step(layer.params, layer.grads)
            history.append(epoch_loss / max(batches, 1))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss {history[-1]:.6f}")
        return history

    def set_training(self, training: bool) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        state = {
            "sizes": self._sizes,
            "activation": self._activation,
            "output": self.output_kind,
            "weights": [[p.tolist() for p in layer.params]
                        for layer in self.layers if layer.params],
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        with open(path) as f:
            state = json.load(f)
        net = cls(state["sizes"], activation=state["activation"],
                  output=state["output"])
        dense_layers = [l for l in net.layers if l.params]
        for layer, saved in zip(dense_layers, state["weights"]):
            layer.W[...] = np.array(saved[0])
            layer.b[...] = np.array(saved[1])
        return net
