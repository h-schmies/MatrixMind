import numpy as np

from tensor import Tensor
from layers import Linear
from activation_funcs import relu
from loss import mse

from typing import SupportsInt, SupportsFloat


class SimpleNN:
    """Simple neural network with one hidden layer"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Construct a new SimpleNN object

        Args:
            input_size (int): The input size of the network, has to align with training data
            hidden_size (int): The size of the hidden layer, depends on problem complexity
            output_size (int): The output size of the network, has to aligh with training data
        """
        self.l1 = Linear(input_size, hidden_size)
        self.l2 = Linear(hidden_size, output_size)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Performs forward propagation on network

        Args:
            x (Tensor): The input data to perform forward propagation with
        """
        x = self.l1(x)
        x = relu(x)
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


if __name__ == "__main__":
    # Create some dummy data for training
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [0]]
    x_train = Tensor(
        np.array(x_train, dtype=np.float64), requires_grad=True
    )  # 10 samples, 3 features
    y_train = Tensor(
        np.array(y_train, dtype=np.float64), requires_grad=True
    )  # 10 samples, 2 output values

    # Initialize the model
    model = SimpleNN(input_size=2, hidden_size=5, output_size=1)

    # Train the model
    model.train(x_train, y_train, epochs=10000)

    print(model(Tensor(np.array([0, 0], dtype=np.float64))))
    print(model(Tensor(np.array([1, 0], dtype=np.float64))))
    print(model(Tensor(np.array([0, 1], dtype=np.float64))))
    print(model(Tensor(np.array([1, 1], dtype=np.float64))))
