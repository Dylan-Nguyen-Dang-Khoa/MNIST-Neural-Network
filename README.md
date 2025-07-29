# MNIST Handwritten Digit Classifier
*A fully connected neural network achieving 97.8% accuracy with L2 regularization and gradient descent*

## Overview
This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset.

**Key Features:**
- 97.8% accuracy on Kaggle's test set
- L2 regularization
- Standard gradient descent optimization
- Weight saving system based on validation accuracy during training

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
- Python 3+
- Numpy

## Usage
### Training a Model
- Change the file that the parameters will be saved to in the train() function
- Feel free to play around with hyperparameters in the __init__ dunder method.
- Run the program and when prompted, type in "train" and "y" if you wish to save the weights and "n" if you do not
### Testing the Model itself
- Specify the model file to load for testing in the test() function
- Specify where you wish to save the results of the test. It will come out as a pure CSV file (No labels, if you wish to you can modify the code)
- Run the program and when prompted, type in "test". Your model should output the CSV of the predictions in your specified location to save the results

## Future Improvements
- Data visualisation in the form of plots and graphs
- Add batch norm to push model accuracy


