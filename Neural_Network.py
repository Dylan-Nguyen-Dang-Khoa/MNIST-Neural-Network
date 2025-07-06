import numpy as np


class Layer:
    def __init__(self, weights, bias):
        self.weights = weights  # The weight matrix is of the order n_out * n_in
        self.bias = bias  # The bias vector is a column vector with n_out biases

    def forward_pass(self, input):  # Input is a column vector
        output = np.dot(self.weights, input) + self.bias  # Equivalent to z = Wx + b
        return max(
            0, output
        )  # Implemented ReLu activation for the pre-activation output

