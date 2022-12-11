# Handwritten-Digit-Recognition
<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="Python" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />  <img alt="Numpy" 
src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img alt="Scikit-Learn" 
src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="TensorFlow" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="Keras" 
src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=00b35a&labelColor=00b35a" /> <img alt="Pycharm" 
src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /> <img alt="Pandas" 
src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" /> 

## Description
This project is a good practice of machine learning algorithm on image recognition.\
The project includes two algorithms: the K-Nearest Neighbors Algorithm and the Neural Network Algorithm, and it also tries some techniques to evaluate and improve the Machine Learning Models.

## MNIST Dataset
https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer
- 42000 examples of 28 pixels by 28 pixels grayscale images of handwritten digits 0-9

## Neural Network Reconizer 
- Number of units in each layer: 784 -> 25 -> 15 -> 10
- Two hidden dense layers with ReLU activations
- One output dense layer with a linear activation (softmax is grouped with the loss function for numerical stability)
- train_acc: 0.9638, val_acc: 0.9538, test_acc: 0.9472

## K-Means Reconizer
- 60 centroids, 30 iterations
- train_acc: 0.8348, val_acc: 0.8384, test_acc: 0.8369
