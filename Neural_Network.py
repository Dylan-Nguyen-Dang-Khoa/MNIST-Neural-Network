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
        return np.maximum(0, pre_activation_output), pre_activation_output

    def softmax(self, pre_activation_output):
        z_shifted = pre_activation_output - np.max(
            pre_activation_output, axis=0, keepdims=True
        )
        z_exp = np.exp(z_shifted)
        return z_exp / np.sum(z_exp, axis=0, keepdims=True), pre_activation_output


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


def dropout_mask(activated_output, keep_prob):
    mask = np.random.rand(*activated_output.shape) < keep_prob
    dropped_output = activated_output * mask
    return dropped_output


def forward_propagation(data, l1, l2, l3, l4):
    a1, z1 = l1.forward_pass(data)
    d1 = dropout_mask(a1, 0.7)
    a2, z2 = l2.forward_pass(d1)
    d2 = dropout_mask(a2, 0.7)
    a3, z3 = l3.forward_pass(d2)
    d3 = dropout_mask(a3, 0.7)
    a4, z4 = l4.forward_pass(d3)
    return a4, (z1, a1, z2, a2, z3, a3, z4)


def cross_entropy_loss(number_probabilities, label):
    return -np.log10(number_probabilities[label])

def calculate_gradients(delta, a_previous):
    return delta * a_previous, delta


def train():
    X_train, Y_train, X_validation, Y_validation = load_data()
    l1 = Layer(he_normal(784, 512), np.zeros((512,)), "relu")
    l2 = Layer(he_normal(512, 256), np.zeros((256,)), "relu")
    l3 = Layer(he_normal(256, 128), np.zeros((128,)), "relu")
    l4 = Layer(he_normal(128, 10), np.zeros((10,)), "softmax")
    max_epochs = 30
    correct_outputs = 0
    total_outputs = 0
    for epoch in range(max_epochs):
        X_train, Y_train = shuffle_data(X_train, Y_train)
        for row in range(len(X_train)):
            final_output, intermediate_outputs = forward_propagation(
                X_train[row, :], l1, l2, l3, l4
            )
            loss = cross_entropy_loss(final_output, Y_train[row])
            dW, db = calculate_gradients()


train()
