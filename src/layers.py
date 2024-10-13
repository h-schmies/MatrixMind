from tensor import Tensor
import numpy as np

from typing import SupportsInt


class Linear:
    """A simple linear fully-connected-layer"""

    def __init__(self, input_size: SupportsInt, output_size: SupportsInt):
        """
        Constructs a new Linear objet

        Args:
            input_size (SupportsInt): The input size of the layer
            output_size (SupportsInt): The output size of the layer / the amount of neurons
        """
        self.weights = Tensor(
            np.random.randn(input_size, output_size), requires_grad=True
        )
        self.biases = Tensor(np.random.randn(output_size), requires_grad=True)

    def __call__(self, x: Tensor):
        """
        Calculates linear output of the layer

        Args:
            x (Tensor): The input data, has to match input size
        """
        return x @ self.weights + self.biases
