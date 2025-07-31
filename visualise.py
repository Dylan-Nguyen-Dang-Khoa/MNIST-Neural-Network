import matplotlib.pyplot as plt
import numpy as np
from Neural_Network import LoadData 
from random import randint

big_data = LoadData(test_label=False)
X_test = big_data.load_test_data()
Y_test = np.loadtxt("./Models/0.97814/Model Results/MNIST Kaggle Test Set/answers.csv", delimiter=",", skiprows=1, dtype=int)
predictions = np.loadtxt("./Models/0.97814/Model Results/MNIST Kaggle Test Set/submissions.csv", delimiter=",", skiprows=1)
correct_mask = predictions[:, 1] == Y_test[:, 1]
wrong_images = X_test[~correct_mask, 1:]
print(Y_test[0, 1])
image_index = randint(0, len(wrong_images)-1)
if len(wrong_images):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(wrong_images[image_index + i].reshape((28, 28)), cmap="binary")
        plt.title(f"Label: {Y_test[image_index + i, 1]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


