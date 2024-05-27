<img src="https://github.com/trtrgfh/Handwritten-Digit-Recognition/assets/73056232/bcca0d8f-4664-4306-9f09-cd371c1e9124" width="500"/>

# Handwritten-Digit-Recognition

# Project Overview
This project serves as a valuable practice in applying machine learning algorithms to image recognition. It involves implementing two key algorithms: the K-Nearest Neighbors (KNN) Algorithm and the Neural Network Algorithm. Additionally, the project explores various techniques to evaluate and enhance the performance of these machine learning models.

# Installation and Setup
## Python Packages Used
- **Data Manipulation:** numpy, pandas
- **Data Visualization:** matplotlib
- **Machine Learning:** tensorflow, scikit-learn

# Data
Dataset can be found at https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer.
- 42000 examples of 28 pixels by 28 pixels grayscale images of handwritten digits 0-9

# Results and evaluation
## Neural Network Reconizer 
- Number of units in each layer: 784 -> 25 -> 15 -> 10
- Two hidden dense layers with ReLU activations
- One output dense layer with a linear activation (softmax is grouped with the loss function for numerical stability)
- train_acc: 0.9638, val_acc: 0.9538, test_acc: 0.9472

## K-Means Reconizer
- 60 centroids, 30 iterations
- train_acc: 0.8348, val_acc: 0.8384, test_acc: 0.8369
