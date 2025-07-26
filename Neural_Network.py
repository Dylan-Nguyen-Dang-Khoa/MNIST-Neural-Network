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
        return self.shuffle_data(self.X_test / 255, self.Y_test)


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
        self.lr = 0.1
        self.weight_decay = 0.0001
        self.batch_size = 32
        self.dropout_prob = 0.0

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
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward_propagation(self, a0, correct_answer):
        clip_value = float("inf")
        y_true = self.one_hot_generation(correct_answer)
        delta3 = self.a3 - y_true
        dW3 = np.clip(
            (self.a2.T @ delta3) / self.batch_size
            + self.weight_decay * self.l3.weights,
            -clip_value,
            clip_value,
        )
        db3 = np.clip(np.sum(delta3, axis=0) / self.batch_size, -clip_value, clip_value)
        delta2 = (delta3 @ self.l3.weights.T) * self.ReLu_differentiation(self.z2)
        dW2 = np.clip(
            (self.a1.T @ delta2) / self.batch_size
            + self.weight_decay * self.l2.weights,
            -clip_value,
            clip_value,
        )
        db2 = np.clip(np.sum(delta2, axis=0) / self.batch_size, -clip_value, clip_value)
        delta1 = (delta2 @ self.l2.weights.T) * self.ReLu_differentiation(self.z1)
        dW1 = np.clip(
            a0.T @ delta1 / self.batch_size + self.weight_decay * self.l1.weights,
            -clip_value,
            clip_value,
        )
        db1 = np.clip(np.sum(delta1, axis=0) / self.batch_size, -clip_value, clip_value)
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
        y_pred = np.clip(self.a3, 1e-10, 1.0)
        correct_probs = y_pred[np.arange(len(y_pred)), correct_answers]
        return -np.sum(np.log(correct_probs))

    def epoch_details(
        self,
        epoch_num,
        training_loss,
        training_accuracy,
        validation_loss,
        validation_accuracy,
    ):
        longest_data_value = max(
            len(f"Epoch {epoch_num+1}"),
            len(f"Average training loss: {training_loss}"),
            len(f"Training accuracy: {training_accuracy}"),
            len(f"Average validation loss: {validation_loss}"),
            len(f"Validation accuracy: {validation_accuracy}"),
        )
        print("-" * longest_data_value)
        print(f"Epoch {epoch_num+1}")
        print()
        print(f"Average training loss: {training_loss}")
        print(f"Training accuracy: {training_accuracy}")
        print(f"Average validation loss: {validation_loss}")
        print(f"Validation accuracy: {validation_accuracy}")
        print("-" * longest_data_value)

    def is_correct_output(self, correct_answers):
        highest_probs_classes = np.argmax(self.a3, axis=1)
        return np.sum(np.equal(highest_probs_classes, correct_answers))


def train():
    big_data = LoadData()
    nn = Network()
    max_epochs = 100
    for epoch in range(max_epochs):
        X_train, Y_train = big_data.load_training_data()
        total_epoch_loss = 0.0
        correct_outputs = 0.0
        for row in range(0, len(X_train), nn.batch_size):
            small_data = X_train[row : row + nn.batch_size]
            correct_answer = Y_train[row : row + nn.batch_size]
            nn.forward_propagation(small_data)
            total_epoch_loss += nn.cross_entropy_loss(correct_answer)
            nn.backward_propagation(small_data, correct_answer)
            correct_outputs += nn.is_correct_output(correct_answer)
        average_training_loss = total_epoch_loss / len(X_train)
        training_accuracy = correct_outputs / len(X_train)
        average_validation_loss, validation_accuracy = validate(
            *big_data.load_validation_data(), nn
        )
        nn.epoch_details(
            epoch,
            average_training_loss,
            training_accuracy,
            average_validation_loss,
            validation_accuracy,
        )


def validate(X_validation, Y_validation, nn):
    correct_outputs = 0.0
    total_validation_loss = 0.0
    for row in range(0, len(X_validation), nn.batch_size):
        small_data = X_validation[row : row + nn.batch_size]
        correct_answer = Y_validation[row : row + nn.batch_size]
        nn.forward_propagation(small_data)
        total_validation_loss += nn.cross_entropy_loss(correct_answer)
        correct_outputs += nn.is_correct_output(correct_answer)
    return total_validation_loss / len(X_validation), correct_outputs / len(
        X_validation
    )


train()
