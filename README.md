# MNIST Handwritten Digit Classifier
*A fully connected neural network achieving 97.8% accuracy with L2 regularisation and gradient descent*

## Overview
This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset.

**Key Features:**
- 97.8% accuracy on Kaggle's test set
- L2 regularization
- Standard gradient descent optimisation
- Tracks the best validation accuracy during training and saves the weights for that epoch.

## Architecture
**Network Structure:**
- Input layer: 784 neurons (28x28 pixels)
- Hidden layers: 1024 -> 512
- Output layer: 10 neurons (digits 0-9)

**Hyperparameters:**
- Learning rate: 0.08
- L2 lambda: 0.001
- Dropout: 0.0
- Batch size: 64

## Dependencies
- Ensure that you have downloaded the latest version of Python
- Download the file requirements.txt to find the entire list of dependencies

## Usage
- I recommend using a virtual environment when running the program
- In the src folder, open your virtual environment, and run
pip install -r requirements.txt
- This will download all the required libraries 
- Run the file using python3 Neural_Network.py training_file testing_file test_label [train|test]
- training_file is the filepath to your training data.
- testing_file is the filepath to your testing data
- type "true" for test_label if your test data has labels. Training data naturally has labels. Type "false" if your test data does not have labels
- the last argument [train|test] is required for the code to run. If it does not exist the code will not do anything. Entering "train" will cause the model to start training based on data from the training_file path. Entering "test" will cause the model to test the model against the data in the test_file

## Future Improvements
- Add batch norm to push model accuracy


