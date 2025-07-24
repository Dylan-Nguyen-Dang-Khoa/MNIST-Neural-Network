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
        )  # The weight matrix is of the order n_in * n_out
        self.bias = np.zeros(
            fan_out
        )  # The bias vector is a column vector with n_out biases

    def he_normal(self, fan_in, fan_out):
        return np.random.normal(
            loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_in, fan_out)
        )


class Network:
    def __init__(self):
        self.l1 = Layer(784, 512)
        self.l2 = Layer(512, 256)
        self.l3 = Layer(256, 128)
        self.l4 = Layer(128, 10)
        self.lr = 0.01
        self.weight_decay = 0.0001
        self.batch_size = 64

    def dropout_mask(self, activated_output, keep_prob):
        mask = np.random.rand(*activated_output.shape) < keep_prob
        dropped_output = activated_output * mask
        return dropped_output

    def forward_propagation(self, a0):
        self.z1 = a0 @ self.l1.weights + self.l1.bias
        self.a1 = self.ReLu(self.z1)
        self.d1 = self.dropout_mask(self.a1, 0.7)
        self.z2 = self.d1 @ self.l2.weights + self.l2.bias
        self.a2 = self.ReLu(self.z2)
        self.d2 = self.dropout_mask(self.a2, 0.7)
        self.z3 = self.d2 @ self.l3.weights + self.l3.bias
        self.a3 = self.ReLu(self.z3)
        self.d3 = self.dropout_mask(self.a3, 0.7)
        self.z4 = self.d3 @ self.l4.weights + self.l4.bias
        self.a4 = self.softmax(self.z4)

    def update_params(self, dW, db, weights, bias):
        weights -= (self.lr * dW) / self.batch_size
        bias -= (self.lr * db) / self.batch_size

    def backward_propagation(self, a0, correct_answer):
        y_true = self.one_hot_generation(correct_answer)
        delta4 = self.a4 - y_true
        dW4 = self.a3.T @ delta4
        db4 = np.sum(delta4, axis=0)
        delta3 = (delta4 @ self.l4.weights.T) * self.ReLu_differentiation(self.z3)
        dW3 = self.a2.T @ delta3
        db3 = np.sum(delta3, axis=0)
        delta2 = (delta3 @ self.l3.weights.T) * self.ReLu_differentiation(self.z2)
        dW2 = self.a1.T @ delta2
        db2 = np.sum(delta2, axis=0)
        delta1 = (delta2 @ self.l2.weights.T) * self.ReLu_differentiation(self.z1)
        dW1 = a0.T @ delta1
        db1 = np.sum(delta1, axis=0)
        self.update_params(dW4, db4, self.l4.weights, self.l4.bias)
        self.update_params(dW3, db3, self.l3.weights, self.l3.bias)
        self.update_params(dW2, db2, self.l2.weights, self.l2.bias)
        self.update_params(dW1, db1, self.l1.weights, self.l1.bias)

    def one_hot_generation(self, correct_answers):
        y_true = np.zeros((correct_answers.shape[0], 10))
        for row in range(len(y_true)):
            y_true[row][correct_answers[row]] = 1.0
        return y_true

    def ReLu_differentiation(self, array):
        return (array > 0).astype(float)

    def ReLu(self, pre_activation_output):
        return np.maximum(0, pre_activation_output)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def cross_entropy_loss(self, correct_answers):
        y_pred = np.clip(self.a4, 1e-12, 1.0)
        correct_probs = y_pred[np.arange(len(y_pred)), correct_answers]
        batch_total_loss = -np.sum(np.log10(correct_probs))
        return batch_total_loss

    def epoch_details(self, epoch_num, loss, epoch_accuracy):
        print(f"Epoch {epoch_num}")
        print()
        print(f"Average training loss: {loss}")
        print(f"Epoch Accuracy: {epoch_accuracy}")

    def is_correct_output(self, correct_answers):
        highest_probs_classes = np.argmax(self.a4, axis=1)
        return np.sum(np.equal(highest_probs_classes, correct_answers))


def train():
    big_data = LoadData()
    nn = Network()
    max_epochs = 30

    for epoch in range(max_epochs):
        X_train, Y_train = big_data.load_training_data()
        total_epoch_loss = 0
        correct_outputs = 0
        for row in range(0, len(X_train), nn.batch_size):
            small_data = X_train[row : row + nn.batch_size]
            correct_answer = Y_train[row : row + nn.batch_size]
            nn.forward_propagation(small_data)
            total_epoch_loss += nn.cross_entropy_loss(correct_answer)
            nn.backward_propagation(small_data, correct_answer)
            correct_outputs += nn.is_correct_output(Y_train[row : row + nn.batch_size])
        average_loss = total_epoch_loss / len(X_train)
        epoch_accuracy = correct_outputs / len(X_train)
        nn.epoch_details(epoch, average_loss, epoch_accuracy)


train()
