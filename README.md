# Handwritten-Digits-Recognizer
<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="NumPy" src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" />  <img alt="Jupyter" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" /> <img alt="Keras" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="Keras"
src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" /> 
                                                                                                                  
## Dataset
- The dataset contains 5000 examples of 20 pixels by 20 pixels grayscale images of handwritten digits 0-9

## Neural Network Reconizer 
- Two dense layers with ReLU activations
- One output layer with a linear activation (softmax is grouped with the loss function for numerical stability)
- 3500 training examples, 750 validation examples, 750 test examples
- train_acc: 0.9980, val_acc: 0.9347, test_acc: 0.9227

## k-nearest neighbors Reconizer
- best_k: 150, best_max_iter: 50
- training set accuracy: 0.8728571428571429
- validation set accuracy: 0.8906666666666667
- test set accuracy: 0.8586666666666667
