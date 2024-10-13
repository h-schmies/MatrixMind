import numpy as np

from typing import Callable, List, SupportsFloat, SupportsInt


class Tensor:
    """Class representing a tensor, supports basic operations like addition, element wise addition and matmul"""

    def __init__(self, data: List | np.ndarray, requires_grad=False):
        """
        Creates a new Tensor from the given data

        Args:
            data (List | np.ndarray): The data of the tensor
            requires_grad (bool): Whether the tensor needs gradient tracking, defaults to False
        """
        self.data = np.array(data) if isinstance(data, list) else data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward: Callable | None = None
        self._children: List[Tensor] = []

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other: "Tensor"):
        """
        Adds two tensors together, tracking gradient function if necesarry

        Args:
            other (Tensor): The tensor to add
        """
        result = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * result.grad
            if other.requires_grad:
                if other.grad.shape != result.grad.shape:
                    other.grad += np.sum(result.grad, axis=0)
                else:
                    other.grad += np.ones_like(other.data) * result.grad

        result._backward = _backward
        result._children = [self, other]

        return result

    def __mul__(self, other: "Tensor"):
        """
        Performs element-wise multiplication with another tensor

        Args:
            other (Tensor): The tensor to multiply with
        """
        result = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += other.data * result.grad
            if other.requires_grad:
                other.grad += self.data * result.grad

        result._backward = _backward
        result._children = [self, other]

        return result

    def __matmul__(self, other: "Tensor"):
        """
        Performs matrix multiplication on two tensors

        Args:
            other (Tensor): The tensor to perform matmul with
        """
        result = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                # Transpose other.data to match dimensions for matmul
                self.grad += np.matmul(result.grad, other.data.T)
            if other.requires_grad:
                # Transpose self.data to match dimensions for matmul
                other.grad += np.matmul(self.data.T, result.grad)

        result._backward = _backward
        result._children = [self, other]

        return result

    def backward(self):
        """Performs gradient calculation with respect to self for all dependencies if requires_grad=True"""
        if not self.requires_grad:
            raise RuntimeError("Cannot compute gradient with requires_grad=False")

        if self.requires_grad and np.all(self.grad == 0):
            self.grad = np.ones_like(self.data)

        if self._backward:
            self._backward()

        for child in self._children:
            child.backward()


if __name__ == "__main__":
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)

    c = a * b
    d = b * c

    print(f"Forward pass result: {d}")

    d.backward()

    print(f"Gradient of a (dC/da): {a.grad}")
    print(f"Gradient of b (dC/db): {b.grad}")
