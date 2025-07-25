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
        self.l1 = Layer(784, 256)
        self.l2 = Layer(256, 128)
        self.l3 = Layer(128, 10)
        self.lr = 0.0001
        self.weight_decay = 0.0005
        self.batch_size = 64
        self.dropout_prob = 0.3

    def dropout_mask(self, activated_output):
        mask = (np.random.rand(*activated_output.shape) > self.dropout_prob).astype(
            float
        )
        dropped_output = activated_output * mask
        return dropped_output

    def forward_propagation(self, a0):
        self.z1 = a0 @ self.l1.weights + self.l1.bias
        self.a1 = self.ReLu(self.z1)
        self.d1 = self.dropout_mask(self.a1)
        self.z2 = self.d1 @ self.l2.weights + self.l2.bias
        self.a2 = self.ReLu(self.z2)
        self.d2 = self.dropout_mask(self.a2)
        self.z3 = self.d2 @ self.l3.weights + self.l3.bias
        self.a3 = self.stable_softmax(self.z3)

    def stable_softmax(self, z):
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward_propagation(self, a0, correct_answer):
        y_true = self.one_hot_generation(correct_answer)
        delta3 = self.a3 - y_true
        dW3 = (
            self.a2.T @ delta3 + self.weight_decay * self.l3.weights
        ) / self.batch_size
        db3 = np.sum(delta3, axis=0) / self.batch_size
        delta2 = (delta3 @ self.l3.weights.T) * self.ReLu_differentiation(self.z2)
        dW2 = (
            self.a1.T @ delta2 + self.weight_decay * self.l2.weights
        ) / self.batch_size
        db2 = np.sum(delta2, axis=0) / self.batch_size
        delta1 = (delta2 @ self.l2.weights.T) * self.ReLu_differentiation(self.z1)
        dW1 = a0.T @ delta1 + self.weight_decay * self.l1.weights / self.batch_size
        db1 = np.sum(delta1, axis=0) / self.batch_size
        self.l3.weights -= self.lr * dW3
        self.l3.bias -= self.lr * db3
        self.l2.weights -= self.lr * dW2
        self.l2.bias -= self.lr * db2
        self.l1.weights -= self.lr * dW1
        self.l1.bias -= self.lr * db1

    def one_hot_generation(self, correct_answers):
        y_true = np.zeros((correct_answers.shape[0], 10))
        for row in range(len(y_true)):
            y_true[row][correct_answers[row]] = 1.0
        return y_true

    def ReLu_differentiation(self, array):
        return (array > 0).astype(float)

    def ReLu(self, pre_activation_output):
        return np.maximum(0, pre_activation_output)

    def cross_entropy_loss(self, correct_answers):
        y_pred = self.a3
        correct_probs = y_pred[np.arange(len(y_pred)), correct_answers]
        batch_total_loss = -np.sum(np.log(correct_probs))
        return batch_total_loss

    def epoch_details(self, epoch_num, loss, epoch_accuracy):
        print("-" * 50)
        print(f"Epoch {epoch_num+1}")
        print()
        print(f"Average training loss: {loss}")
        print(f"Epoch Accuracy: {epoch_accuracy}")
        print("-" * 50)

    def is_correct_output(self, correct_answers):
        highest_probs_classes = np.argmax(self.a3, axis=1)
        return np.sum(np.equal(highest_probs_classes, correct_answers))


def train():
    big_data = LoadData()
    nn = Network()
    max_epochs = 67
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
