import numpy as np

from tensor import Tensor

import math


def relu(x: Tensor) -> Tensor:
    """
    ReLU function implementation with appropriate gradient calculation
    Note: Derivative is defined to fill patch differentiability at 0

    x (Tensor): The input data
    """
    result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    def _backward():
        x.grad = x.grad + (x.data > 0) * result.grad

    result._backward = _backward
    result._children = [x]

    return result


def sigmoid(x: Tensor) -> Tensor:
    result = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)

    def _backward():
        sigmoid_grad = result.data * (1 - result.data)
        x.grad += sigmoid_grad * result.grad


    result._backward = _backward
    result._children = [x]
    
    return result
