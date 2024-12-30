# First-Image-Classifier

## Overview
Used a pre-trained model (VGG16) to create an image classifier that identifies and classifies different types of flowers and plants.

## Detailed steps
- Loaded a pre-trained network: VGG16
- Defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
- Trained the classifier layers using backpropagation using the pre-trained network to get the features
- Tracked the loss and accuracy on the validation set to determine the best hyperparameters

## Results
- Final Test loss: 0.437  Test accuracy: 0.882
