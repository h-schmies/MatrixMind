import numpy as np

from tensor import Tensor


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
