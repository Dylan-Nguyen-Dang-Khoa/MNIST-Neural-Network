import numpy as np


class LoadData:
    def __init__(self):
        training_data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
        self.X_train, self.Y_train = training_data[:, 1:], training_data[:, 0].astype(
            int
        )
        test_data = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
        self.X_test, self.Y_test = test_data[:, 1:], test_data[:, 0].astype(int)

    def shuffle_data(self, x_data, y_data):
        assert x_data.shape[0] == x_data.shape[0]
        permuted_indicies = np.random.permutation(x_data.shape[0])
        return x_data[permuted_indicies], y_data[permuted_indicies]

    def load_training_data(self):
        return self.shuffle_data(self.X_train[:50000, :] / 255, self.Y_train[:50000])

    def load_validation_data(self):
        return self.shuffle_data(self.X_train[50000:, :] / 255, self.Y_train[50000:])

    def load_test_data(self):
        return self.shuffle_data(self.X_test, self.Y_test)


class Layer:
    def __init__(self, fan_in, fan_out):
        self.weights = self.he_normal(
            fan_in, fan_out
        )  # The weight matrix is of the order n_out * n_in
        self.bias = np.zeros(
            fan_out
        )  # The bias vector is a column vector with n_out biases

    def he_normal(self, fan_in, fan_out):
        return np.random.normal(
            loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_out, fan_in)
        )


class Network:
    def __init__(self):
        self.l1 = Layer(784, 512)
        self.l2 = Layer(512, 256)
        self.l3 = Layer(256, 128)
        self.l4 = Layer(128, 10)
        self.lr = 0.01
        self.weight_decay = 0.0001

    def dropout_mask(self, activated_output, keep_prob):
        mask = np.random.rand(*activated_output.shape) < keep_prob
        dropped_output = activated_output * mask
        return dropped_output

    def forward_propagation(self, data):
        self.z1 = self.l1.weights @ data + self.l1.bias
        self.a1 = self.ReLu(self.z1)
        self.d1 = self.dropout_mask(self.a1, 0.7)
        self.z2 = self.l2.weights @ self.d1 + self.l2.bias
        self.a2 = self.ReLu(self.z2)
        self.d2 = self.dropout_mask(self.a2, 0.7)
        self.z3 = self.l3.weights @ self.d2 + self.l3.bias
        self.a3 = self.ReLu(self.z3)
        self.d3 = self.dropout_mask(self.a3, 0.7)
        self.z4 = self.l4.weights @ self.d3 + self.l4.bias
        self.a4 = self.softmax(self.z4)

    def hidden_layer_grad_calculation(
        self,
        next_layer_weights,
        next_layer_delta,
        this_layer_weights,
        this_layer_z,
        this_layer_a,
    ):
        delta = (
            next_layer_weights
            @ next_layer_delta
            * self.ReLu_differentiation(this_layer_z)
        )
        return (
            delta * this_layer_a.reshape(1, this_layer_a.shape[0])
            + self.weight_decay * this_layer_weights,
            delta,
        )

    def update_params(self, dW, db, weights, bias):
        weights -= self.lr * dW
        bias -= self.lr * db

    def backward_propagation(self, correct_answer, y=np.zeros(10)):
        y[correct_answer] = 1
        delta_l4 = (self.a4 - y).reshape(10, 1)
        dW_4 = delta_l4 * self.a3.reshape(1, 128) + self.weight_decay * self.l4.weights
        db_4 = delta_l4
        dW_3, db_3 = self.hidden_layer_grad_calculation(
            self.l4.weights, delta_l4, self.l3.weights, self.z3, self.a3
        )
        self.update_params(dW_3, db_3, self.l3.weights, self.l3.bias)
        dW_2, db_2 = self.hidden_layer_grad_calculation(
            self.l3.weights, db_3, self.l2.weights, self.z2, self.a2
        )
        self.update_params(dW_2, db_2, self.l2.weights, self.l2.bias)
        dW_1, db_1 = self.hidden_layer_grad_calculation(
            self.l2.weights, db_2, self.l1.weights, self.z1, self.a1
        )
        self.update_params(dW_1, db_1, self.l1.weights, self.l1.bias)

    def ReLu_differentiation(array):
        return (array > 0).astype(float)

    def ReLu(self, pre_activation_output):
        return np.maximum(0, pre_activation_output)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


def train():
    big_data = LoadData()
    nn = Network()
    max_epochs = 30
    correct_outputs = 0
    total_outputs = 0
    for epoch in range(max_epochs):
        X_train, Y_train = big_data.load_training_data()
        for row in range(len(X_train)):
            small_data = X_train[row]
            nn.forward_propagation(small_data)
            nn.backward_propagation(Y_train[row])


train()
