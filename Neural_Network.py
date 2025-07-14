import numpy as np


class LoadData:
    def __init__(self):
        training_data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
        X_train, Y_train = self.shuffle_data(
            training_data[:, 1:], training_data[:, 0].astype(int)
        )
        test_data = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
        X_test, Y_test = self.shuffle_data(
            test_data[:, 1:], test_data[:, 0].astype(int)
        )

    def shuffle_data(self):
        assert self.X_data.shape[0] == self.Y_data.shape[0]
        permuted_indicies = np.random.permutation(self.X_data.shape[0])
        return self.X_data[permuted_indicies], self.Y_data[permuted_indicies]

    def load_training_data(self):
        return (
            self.X_data[:50000, :] / 255,
            self.Y_data[:50000],
        )

    def load_validation_data(self):
        return self.X_data[50000:, :] / 255, self.Y_data[50000:]

    def load_test_data(self):
        return self.X_test, self.Y_test


class Layer:
    def __init__(self, inputs, outputs):
        self.weights = self.he_normal(
            inputs, outputs
        )  # The weight matrix is of the order n_out * n_in
        self.bias = np.zeros(
            outputs
        )  # The bias vector is a column vector with n_out biases

    def he_normal(self, fan_in, fan_out):
        return np.random.normal(
            loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_out, fan_in)
        )


class Network:
    def __init__(self, layers):
        self.l1 = Layer(784, 512)
        self.l2 = Layer(512, 256)
        self.l3 = Layer(256, 128)
        self.l4 = Layer(128, 10)

    def dropout_mask(self, activated_output, keep_prob):
        mask = np.random.rand(*activated_output.shape) < keep_prob
        dropped_output = activated_output * mask
        return dropped_output

    def forward_propagation(self, data):
        self.z1 = self.l1.weights @ data + self.l1.bias
        self.a1 = self.ReLu(self.z1)
        self.d1 = self.dropout_mask(self.a1, 0.7)

    def ReLu(self, pre_activation_output):
        return np.maximum(0, pre_activation_output), pre_activation_output

    def softmax(self, pre_activation_output):
        z_shifted = pre_activation_output - np.max(
            pre_activation_output, axis=0, keepdims=True
        )
        z_exp = np.exp(z_shifted)
        return z_exp / np.sum(z_exp, axis=0, keepdims=True), pre_activation_output


def delta_calculation(intermediate_outputs, layers): ...


def train():
    big_data = LoadData()
    layers = [
        Layer(he_normal(784, 512), np.zeros((512,)), "relu"),
        Layer(he_normal(512, 256), np.zeros((256,)), "relu"),
        Layer(he_normal(256, 128), np.zeros((128,)), "relu"),
        Layer(he_normal(128, 10), np.zeros((10,)), "softmax"),
    ]
    max_epochs = 30
    correct_outputs = 0
    total_outputs = 0
    calculate_dW = lambda delta, a_previous: delta * a_previous
    for epoch in range(max_epochs):
        X_train, Y_train = shuffle_data(X_train, Y_train)
        for row in range(len(X_train)):
            correct_answer = [Y_train[row]]
            final_output, intermediate_outputs = forward_propagation(
                X_train[row, :], *layers
            )
            correct_outputs += 1 if np.argmax(final_output) == correct_answer else 0
            total_outputs += 1
            delta = delta_calculation(intermediate_outputs, layers)
            for layer_i in range(3, 0, -1):
                layer = layers[layer_i]
                layer_delta = delta[layer_i]
                a_previous = intermediate_outputs[layer_i - 1][0].reshape(
                    1, intermediate_outputs[layer_i - 1][0].shape[0]
                )
                dW, db = calculate_dW(delta, a_previous), delta


train()
