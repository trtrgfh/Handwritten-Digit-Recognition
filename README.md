# Handwritten-Digits-Recognizer

## Dataset
- The dataset contains 5000 examples of 20 pixels by 20 pixels images of handwritten digits 0-9

## Neural Network Reconizer
- Three layers neural network
- Two dense layers with ReLU activations
- One output layer with a linear activation (softmax is grouped with the loss function for numerical stability)
- 3500 training examples, 750 validation examples, 750 test examples
- train_acc: 0.9980, val_acc: 0.9347, test_acc: 0.9227
