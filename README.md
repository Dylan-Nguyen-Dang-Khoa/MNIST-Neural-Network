# MNIST Handwritten Digit Classifier
*A fully connected neural network achieving 97.8% accuracy with L2 regularisation and gradient descent*

## Overview
This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset.

**Key Features:**
- 97.8% accuracy on Kaggle's test set
- L2 regularization
- Standard gradient descent optimization
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
- Python 3+
- Numpy

## File Structure
.
├── Training Results/
│   ├── 0.97814/               # Best model (97.8% accuracy)
│   ├── Untested/              # New trainings
│   └── Default/               # Misc runs
├── Neural_Network.py          # Main script
└── README.md
└── visualise.py               # Data plotting and graphing


## Usage
### Training a Model
- Change the file that the parameters will be saved to in the train() function. You would have to change it in the codebase itself.
- Feel free to play around with hyperparameters in the __init__ dunder method.
- Run the program. The prompt is as follows:
Train or test:
- Put "train" to train the model
- Afterwards, you will be prompted:
Do you wish to save the weights of the training? (y, n):
- Put "y" if you wish to save to the model parameters and "n" if you do not wish to
  
### Testing the Model 
- Specify the model file to load for testing in the test() function
- Specify where you wish to save the results of the test. It will come out as a pure CSV file (No labels, if you wish to you can modify the code)
- Run the program. The prompt is as follows:
Train or test:
- Put "test" to run the model parameters against the test data

## Future Improvements
- Data visualisation in the form of plots and graphs
- Add batch norm to push model accuracy


