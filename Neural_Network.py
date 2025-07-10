import numpy as np


class Layer:
    def __init__(self, weights, bias, activation):
        self.weights = weights  # The weight matrix is of the order n_out * n_in
        self.bias = bias  # The bias vector is a column vector with n_out biases
        self.activation = activation

    def forward_pass(self, input):  # Input is a column vector
        pre_activation_output = (
            np.dot(self.weights, input) + self.bias
        )  # Equivalent to z = Wx + b
        return (
            self.ReLu(pre_activation_output)
            if self.activation == "relu"
            else self.softmax(pre_activation_output)
        )  # Return activated output with ReLu if hidden layer else use softmax for output layer

    def ReLu(self, pre_activation_output):
        return np.maximum(0, pre_activation_output)

    def softmax(self, pre_activation_output):
        z_shifted = pre_activation_output - np.max(
            pre_activation_output, axis=0, keepdims=True
        )
        z_exp = np.exp(z_shifted)
        return z_exp / np.sum(z_exp, axis=0, keepdims=True)


def he_normal(fan_in, fan_out):
    return np.random.normal(loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_out, fan_in))


def load_data():
    data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
    X_data, Y_data = shuffle_data(data[:, 1:], data[:, 0].astype(int))
    return (
        X_data[:50000, :] / 255,
        Y_data[:50000],
        X_data[50000:, :] / 255,
        Y_data[50000:],
    )


def shuffle_data(x_data, y_data):
    assert x_data.shape[0] == y_data.shape[0]
    permuted_indicies = np.random.permutation(x_data.shape[0])
    return x_data[permuted_indicies], y_data[permuted_indicies]


def forward_propagation(X_train, Y_train):
    hidden_layer_1 = Layer(he_normal(784, 512), np.zeros((512,)), "relu")
    hidden_layer_2 = Layer(he_normal(512, 256), np.zeros((256,)), "relu")
    hidden_layer_3 = Layer(he_normal(256, 128), np.zeros((128,)), "relu")
    output_layer = Layer(he_normal(128, 10), np.zeros((10,)), "softmax")
    max_epochs = 80
    correct_outputs = 0
    total_outputs = 0
    for epoch in range(max_epochs):
        for row in range(len(X_train)):
            layer_1_output = hidden_layer_1.forward_pass(X_train[row, :])
            layer_2_output = hidden_layer_2.forward_pass(layer_1_output)
            layer_3_output = hidden_layer_3.forward_pass(layer_2_output)
            final_output = output_layer.forward_pass(layer_3_output)
            print(final_output)


def main():
    X_train, Y_train, X_validation, Y_validation = load_data()
    forward_propagation(X_train, Y_train)


main()
