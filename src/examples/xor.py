import numpy as np


if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        from tensor import Tensor
        from nn import SimpleNN
    else:
        from ..tensor import Tensor
        from ..nn import SimpleNN

    # Create XOR data from training
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [0]]
    x_train = Tensor(
        np.array(x_train, dtype=np.float64), requires_grad=True
    )  
    y_train = Tensor(
        np.array(y_train, dtype=np.float64), requires_grad=True
    )  

    # Initialize the model
    model = SimpleNN(input_size=2, hidden_size=5, output_size=1)

    # Train the model
    model.train(x_train, y_train, epochs=10000)

    print(model(Tensor(np.array([0, 0], dtype=np.float64)))) # Should be 0
    print(model(Tensor(np.array([1, 0], dtype=np.float64)))) # Should be 1
    print(model(Tensor(np.array([0, 1], dtype=np.float64)))) # Should be 1
    print(model(Tensor(np.array([1, 1], dtype=np.float64)))) # Should be 0