# Custom Tensor Library with Neural Network Implementation

This project implements a custom tensor library with automatic differentiation (autograd) capabilities, along with a simple neural network framework built on top of it.

## Features

- Custom Tensor implementation
- Automatic differentiation (autograd)
- Simple neural network layers (Linear)
- Activation functions (ReLU implemented, others can be added)
- Basic training loop for neural networks

## Project Structure

- `tensor.py`: Core Tensor class implementation
- `layers.py`: Neural network layers (e.g., Linear)
- `activation_funcs.py`: Activation functions
- `loss.py`: Loss functions for training
- `nn.py`: Neural network model implementation (SimpleNN class)

## Usage

Here's a basic example of how to use the SimpleNN class to create and train a neural network:

```python
from nn import SimpleNN
from tensor import Tensor
from activation_funcs import relu

# Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
model = SimpleNN(input_size=2, hidden_size=3, output_size=1, activation_function=relu)

# Example training data
X_train = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = Tensor([[0], [1], [1], [0]])

# Train the model
model.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = model(X_train)
print("Predictions:", predictions)
```

## Implementation Details

- The `SimpleNN` class implements a basic feedforward neural network with one hidden layer.
- The `__call__` method performs forward propagation through the network.
- The `train` method implements a simple training loop using stochastic gradient descent.

## Future Improvements

- Implement more layer types (e.g., Convolutional, Recurrent)
- Add more activation functions and loss functions
- Implement batch processing for improved training efficiency
- Add regularization techniques
- Implement model saving and loading functionality
- Add GPU support for faster training and running

## Contributing

Contributions to improve and expand this tensor library and neural network implementation are welcome. Please feel free to submit pull requests or open issues to discuss potential improvements.

## License

[MIT License](https://opensource.org/licenses/MIT)