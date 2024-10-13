import numpy as np

from tensor import Tensor
from layers import Linear
from activation_funcs import relu
from loss import mse

from typing import SupportsInt, SupportsFloat, Callable


class SimpleNN:
    """Simple neural network with one hidden layer"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation_function: Callable = relu,
    ):
        """
        Construct a new SimpleNN object

        Args:
            input_size (int): The input size of the network, has to align with training data
            hidden_size (int): The size of the hidden layer, depends on problem complexity
            output_size (int): The output size of the network, has to aligh with training data
        """
        self.l1 = Linear(input_size, hidden_size)
        self.l2 = Linear(hidden_size, output_size)

        self.activation_function = activation_function

    def __call__(self, x: Tensor) -> Tensor:
        """
        Performs forward propagation on network

        Args:
            x (Tensor): The input data to perform forward propagation with
        """
        x = self.l1(x)
        x = self.activation_function(x)
        x = self.l2(x)
        return x

    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        epochs: int = 100,
        learning_rate: SupportsFloat = 0.01,
    ):
        """
        Performs training on network

        Args:
            x_train (Tensor): Tensor containing the training data
            y_train (Tensor): Tensor containing the training lables
            epochs (int): The number of epochs to train for
            learning_rate (SupportsFloat): The learning rate to train with, should be a small value for good results
        """
        for epoch in range(epochs):
            pred = self(x_train)
            loss = mse(pred, y_train)

            loss.backward()

            for param in [
                self.l1.weights,
                self.l1.biases,
                self.l2.weights,
                self.l2.biases,
            ]:
                param.data -= learning_rate * param.grad
                param.grad = np.zeros_like(param.data)
            print(f"Epoch {epoch}, Loss: {loss}")
