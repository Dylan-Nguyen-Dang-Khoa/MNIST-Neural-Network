from numpy.typing import NDArray
import numpy as np


class LoadData:
    def __init__(self):
        self.training_data = np.loadtxt(
            "./Training Data/Self Training/mnist_test.csv",
            delimiter=",",
            skiprows=1,
        )
        self.X_train, self.Y_train = self.training_data[:, 1:], self.training_data[
            :, 0
        ].astype(int)
        self.test_data = np.loadtxt(
            "./Training Data/Kaggle Competition Data/test.csv",
            delimiter=",",
            skiprows=1,
        )
        self.X_test = self.test_data[:, :]

    def shuffle_data(
        self, x_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert x_data.shape[0] == x_data.shape[0]
        permuted_indicies = np.random.permutation(x_data.shape[0])
        return x_data[permuted_indicies], y_data[permuted_indicies]

    def load_training_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.shuffle_data(
            self.X_train[: int(0.9 * len(self.X_train)), :] / 255,
            self.Y_train[: int(0.9 * len(self.X_train))],
        )

    def load_validation_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.shuffle_data(
            self.X_train[int(0.9 * len(self.X_train)) :, :] / 255,
            self.Y_train[int(0.9 * len(self.X_train)) :],
        )

    def load_test_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.X_test / 255


class Layer:
    def __init__(self, fan_in: int, fan_out: int):
        self.weights = self.he_normal(
            fan_in, fan_out
        )  # The weight matrix is of the order n_in * n_out
        self.bias = np.zeros(
            fan_out
        )  # The bias vector is a column vector with n_out biases

    def he_normal(self, fan_in: int, fan_out: int) -> NDArray[np.float64]:
        return np.random.normal(
            loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_in, fan_out)
        )


class Network:
    def __init__(self):
        self.l1 = Layer(784, 1024)
        self.l2 = Layer(1024, 512)
        self.l3 = Layer(512, 10)
        self.lr = 0.08
        self.weight_decay = 0.001
        self.dropout_prob = 0.0
        self.batch_size = 64

    def dropout_mask(
        self, activated_output: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        mask = (np.random.rand(*activated_output.shape) > self.dropout_prob).astype(
            float
        )
        dropped_output = activated_output * mask
        return dropped_output

    def forward_propagation(self, a0: NDArray[np.float64], dropout: float) -> None:
        self.z1 = a0 @ self.l1.weights + self.l1.bias
        self.a1 = self.ReLu(self.z1)
        self.d1 = self.dropout_mask(self.a1) if dropout else self.a1
        self.z2 = self.d1 @ self.l2.weights + self.l2.bias
        self.a2 = self.ReLu(self.z2)
        self.d2 = self.dropout_mask(self.a2) if dropout else self.a2
        self.z3 = self.d2 @ self.l3.weights + self.l3.bias
        self.a3 = self.stable_softmax(self.z3)

    def stable_softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward_propagation(
        self, a0: NDArray[np.float64], correct_answer: NDArray[np.float64]
    ) -> None:
        clip_value = 1.0
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

    def one_hot_generation(
        self, correct_answers: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        y_true = np.zeros((correct_answers.shape[0], 10))
        for row in range(len(y_true)):
            y_true[row][correct_answers[row]] = 1.0
        return y_true

    def ReLu_differentiation(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        return (array > 0).astype(float)

    def ReLu(self, pre_activation_output: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, pre_activation_output)

    def cross_entropy_loss(self, correct_answers: NDArray[np.float64]) -> float:
        y_pred = np.clip(self.a3, 1e-10, 1.0)
        correct_probs = y_pred[np.arange(len(y_pred)), correct_answers]
        return -np.sum(np.log(correct_probs))

    def epoch_details(
        self,
        epoch_num: int,
        training_loss: float,
        training_accuracy: float,
        validation_loss: float,
        validation_accuracy: float,
        validation_wrong_predictions: dict[int, int],
    ) -> None:
        print("-" * 50)
        print(f"Epoch {epoch_num+1}")
        print()
        print(f"Average training loss: {training_loss}")
        print(f"Training accuracy: {training_accuracy}")
        print(f"Average validation loss: {validation_loss}")
        print(f"Validation accuracy: {validation_accuracy}")
        print(
            f"The following are the number of times the wrong class was predicted for each class:"
        )
        print()
        for num_class, wrong_predictions in validation_wrong_predictions.items():
            print(f"{num_class}: {wrong_predictions}")
        print("-" * 50)

    def test_details(
        self, testing_accuracy: float, test_wrong_predictions: dict[int, int]
    ) -> None:
        print("-" * 50)
        print(f"Testing accuracy: {testing_accuracy}")
        print()
        print(
            f"The following are the number of times the wrong class was predicted for each class:"
        )
        print()
        for num_class, wrong_predictions in test_wrong_predictions.items():
            print(f"{num_class}: {wrong_predictions}")
        print("-" * 50)

    def is_correct_output(self, correct_answers: NDArray[np.float64]) -> int:
        predicted_classes = np.argmax(self.a3, axis=1)
        correct_mask = predicted_classes == correct_answers
        return np.sum(correct_mask)

    def wrong_counter(
        self, correct_answers: NDArray[np.float64], wrong_predictions: dict[int, int]
    ) -> dict[int, int]:
        predicted_classes = np.argmax(self.a3, axis=1)
        correct_mask = predicted_classes == correct_answers
        for true_class in correct_answers[~correct_mask]:
            wrong_predictions[true_class] += 1
        return wrong_predictions

    def save_parameters(
        self, filepath="./Training Results/Default/model_kaggle_parameters.npz"
    ) -> None:
        parameters = {
            "layer 1 weights": self.l1.weights,
            "layer 1 bias": self.l1.bias,
            "layer 2 weights": self.l2.weights,
            "layer 2 bias": self.l2.bias,
            "layer 3 weights": self.l3.weights,
            "layer 3 bias": self.l3.bias,
        }
        np.savez(filepath, **parameters)

    def save_hyperparameters(
        self, filepath="./Training Results/Default/model_kaggle_hyperparameters.npz"
    ) -> None:
        hyperparameters = {
            "learning rate": self.lr,
            "weight decay": self.weight_decay,
            "batch size": self.batch_size,
            "dropout prob": self.dropout_prob,
        }
        np.savez(filepath, **hyperparameters)

    def load_parameters(
        self, filepath="./Training Results/Default/model_kaggle_parameters.npz"
    ) -> None:
        try:
            load_parameters = np.load(filepath, allow_pickle=True)
            self.l1.weights = load_parameters["layer 1 weights"]
            self.l1.bias = load_parameters["layer 1 bias"]
            self.l2.weights = load_parameters["layer 2 weights"]
            self.l2.bias = load_parameters["layer 2 bias"]
            self.l3.weights = load_parameters["layer 3 weights"]
            self.l3.bias = load_parameters["layer 3 bias"]
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except KeyError as e:
            print(f"Error: Missing parameter {e} in the file.")

    def load_hyperparameters(
        self, filepath="./Training Results/Default/model_kaggle_hyperparameters.npz"
    ) -> None:
        try:
            load_hyperparameters = np.load(filepath, allow_pickle=True)
            self.lr = load_hyperparameters["learning rate"]
            self.weight_decay = load_hyperparameters["weight decay"]
            self.batch_size = load_hyperparameters["batch size"]
            self.dropout_prob = load_hyperparameters["dropout prob"]
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except KeyError as e:
            print(f"Error: Missing hyperparameter {e} in the file.")


class Corrupt:
    @staticmethod
    def corrupt_labels(
        y: NDArray[np.float64], corruption_rate=1.0
    ) -> NDArray[np.float64]:
        np.random.seed(42)
        mask = np.random.rand(len(y)) < corruption_rate
        y_corrupted = y.copy()
        y_corrupted[mask] = np.random.randint(0, 10, size=np.sum(mask))
        return y_corrupted

    @staticmethod
    def corrupt_features_gaussian(
        x: NDArray[np.float64], corruption_rate=0.67
    ) -> NDArray[np.float64]:
        np.random.seed(42)
        mask = np.random.rand(len(x)) < corruption_rate
        x_corrupted = x.copy()
        x_corrupted[mask] = x[mask] + np.random.normal(0, 0.5, x[mask].shape)
        x_corrupted = np.clip(x_corrupted, 0, 1)
        return x_corrupted

    @staticmethod
    def corrupt_features_uniform(
        x: NDArray[np.float64], corruption_rate=1.0
    ) -> NDArray[np.float64]:
        np.random.seed(42)
        mask = np.random.rand(len(x)) < corruption_rate
        x_corrupted = x.copy()
        x_corrupted[mask] = np.random.rand(*x_corrupted[mask].shape)
        return x_corrupted


def train() -> None:
    save_weights = (
        input("Do you wish to save the weights of the training? (y, n): ")
        .strip()
        .lower()
    )
    big_data = LoadData()
    nn = Network()
    max_epochs = 2**63 - 1
    best_validation_accuracy = 0.0
    best_epoch = 0
    try:
        for epoch in range(max_epochs):
            X_train, Y_train = big_data.load_training_data()
            total_epoch_loss = 0.0
            correct_outputs = 0
            for row in range(0, len(X_train), nn.batch_size):
                small_data = X_train[row : row + nn.batch_size]
                correct_answers = Y_train[row : row + nn.batch_size]
                nn.forward_propagation(small_data, True)
                total_epoch_loss += nn.cross_entropy_loss(correct_answers)
                nn.backward_propagation(small_data, correct_answers)
                correct_outputs += nn.is_correct_output(correct_answers)
            average_training_loss = total_epoch_loss / len(X_train)
            training_accuracy = correct_outputs / len(X_train)
            (
                average_validation_loss,
                validation_accuracy,
                validation_wrong_predictions,
            ) = validate(*big_data.load_validation_data(), nn)
            nn.epoch_details(
                epoch,
                average_training_loss,
                training_accuracy,
                average_validation_loss,
                validation_accuracy,
                validation_wrong_predictions,
            )
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
                if save_weights == "y":
                    nn.save_parameters(
                        filepath="./Training Results/Untested/model_kaggle_parameters.npz"
                    )
                    print("Parameters saved")
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
        print(f"Best epoch: {best_epoch}")
        print(f"Epoch {best_epoch} validation accuracy: {best_validation_accuracy}")


def validate(
    X_validation: NDArray[np.float64], Y_validation: NDArray[np.float64], nn: Network
) -> tuple[float, float, dict[int, int]]:
    validation_wrong_predictions = {num_class: 0 for num_class in range(10)}
    correct_outputs = 0
    total_validation_loss = 0.0
    for row in range(0, len(X_validation), nn.batch_size):
        small_data = X_validation[row : row + nn.batch_size]
        correct_answers = Y_validation[row : row + nn.batch_size]
        nn.forward_propagation(small_data, False)
        total_validation_loss += nn.cross_entropy_loss(correct_answers)
        correct_outputs += nn.is_correct_output(correct_answers)
        validation_wrong_predictions = nn.wrong_counter(
            correct_answers, validation_wrong_predictions
        )
    return (
        total_validation_loss / len(X_validation),
        correct_outputs / len(X_validation),
        validation_wrong_predictions,
    )


def test() -> None:
    big_data = LoadData()
    nn = Network()
    nn.batch_size = 512
    nn.load_parameters(
        filepath="./Training Results/0.97814/model_kaggle_parameters.npz"
    )
    X_test = big_data.load_test_data()
    index = 1
    with open("./Training Results/0.97814/submissions.csv", "w") as f:
        f.write("ImageId,Label\n")
    for row in range(0, len(X_test), nn.batch_size):
        small_data = X_test[row : row + nn.batch_size]
        nn.forward_propagation(small_data, False)
        with open("./Training Results/0.97814/submissions.csv", "a") as f:
            for predictions in nn.a3:
                np.savetxt(
                    f,
                    np.array([[index, np.argmax(predictions)]]),
                    fmt="%d",
                    delimiter=",",
                )
                index += 1


def main():
    action = input("Train or test: ").strip().lower()
    if action == "train":
        train()
        print("Training complete")
    elif action == "test":
        test()
        print("Testing complete")


if __name__ == "__main__":
    main()
