import numpy as np

from tensor import Tensor

def mse(pred: Tensor, target: Tensor): 
    """
    Mean squared error function with appropriate gradient calculation

    Args: 
        pred (Tensor): The predictions from the model
        target (Tensor): The actual target values
    """
    loss = Tensor([np.mean((pred.data - target.data)**2)], requires_grad=True)

    def _backward(): 
        pred.grad += (2 * (pred.data - target.data)) / target.data.size

    loss._backward = _backward
    loss._children = [pred]
    return loss
