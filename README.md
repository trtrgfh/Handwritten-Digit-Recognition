# Handwritten-Digits-Recognizer
<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="Python" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />  <img alt="Numpy" 
src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img alt="Scikit-Learn" 
src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="TensorFlow" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="Keras" 
src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=00b35a&labelColor=00b35a" /> <img alt="Pycharm" 
src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /> 
 


## Dataset
- 5000 examples of 20 pixels by 20 pixels grayscale images of handwritten digits 0-9
- 3500 training examples, 750 validation examples, 750 test examples

## Neural Network Reconizer 
- Two dense layers with ReLU activations
- One output layer with a linear activation (softmax is grouped with the loss function for numerical stability)
- train_acc: 0.9980, val_acc: 0.9347, test_acc: 0.9227

## k-nearest neighbors Reconizer
- 60 centroids, 30 iterations
- train_acc: 0.8214, val_acc: 0.8440, test_acc: 0.8347
