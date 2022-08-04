import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from helper import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid


def load_data():
    X = np.load("/Users/yehao/Desktop/Projects/Machine Learning Practice/Handwritten Digits Recognizer/data/X.npy")
    y = np.load("/Users/yehao/Desktop/Projects/Machine Learning Practice/Handwritten Digits Recognizer/data/y.npy")
    return split_data(X, y)

def split_data(X, y, train_size = 0.7):
    """
    Return a dictionary containing the traning set, validation set and the test set
    """
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5)
    data = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    return data

def create_nn():
    # Since each image is 20 by 20, the input is 400
    tf.random.set_seed(1234)
    model = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(25, activation = 'relu', name = 'layer1'),
            Dense(15, activation = 'relu', name = 'layer2'),
            Dense(10, activation = 'linear', name = 'layer3')
        ], name = "nn_model")
    return model

def select_model(data, learning_rate, epochs):
    best_acc = 0
    best_epoch = 0
    best_lr = 0
    best_model = None

    for epoch in epochs:
        for l in learning_rate:
            model = create_nn()
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=l),
            )
            model.fit(
                data['train'][0], data['train'][1],
                epochs=epoch
            )
            train_prediction, train_acc = predict(model, data['train'][0], data['train'][1])
            # print("epochs: {}, learning rate: {}, train_acc: {}".format(epoch, l, train_acc))
            train_prediction, val_acc = predict(model, data['val'][0], data['val'][1])
            # print("epochs: {}, learning rate: {}, val_acc: {}".format(epoch, l, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_lr = l
                best_model = model

    print("best epochs: {}, best learning rate: {}".format(best_epoch, best_lr))

    train_prediction, train_acc = predict(best_model, data['train'][0], data['train'][1])
    print("train_acc: {:.4f}".format(train_acc))

    train_prediction, val_acc = predict(best_model, data['val'][0], data['val'][1])
    print("val_acc: {:.4f}".format(val_acc))

    test_prediction, test_acc = predict(best_model, data['test'][0], data['test'][1])
    print("test_acc: {:.4f}".format(test_acc))

    return best_model



def predict(model, X, y):
    m, n = X.shape
    prediction = model.predict(X.reshape(-1, 400))
    prediction = np.argmax(prediction, axis = 1)
    prediction = prediction.reshape(m, 1)
    accuracy = np.sum((prediction == y) / m)

    return prediction, accuracy

def main():
    # The data contains 5000 examples of handwritten digits. Each example is a 20 pixel by 20 pixel grayscale image
    # Each 20 by 20 image is unrolled into a 400 dimensional vector.
    data = load_data()
    print("shape of X_train is {}".format(data['train'][0].shape))
    print("shape of y_train is {}".format(data['train'][1].shape))
    print("shape of X_val is {}".format(data['val'][0].shape))
    print("shape of y_val is {}".format(data['val'][1].shape))
    print("shape of X_test is {}".format(data['test'][0].shape))
    print("shape of y_test is {}".format(data['test'][1].shape))

    # learning_rate = [0.001, 0.01, 0.05]
    # epochs = [10, 25, 40, 60]
    learning_rate = [0.001]
    epochs = [40]
    best_model = select_model(data, learning_rate, epochs)
    print(f"{display_errors(best_model, data['test'][0], data['test'][1])} errors out of {len(data['test'][0])} images in test set")
    plt.show()


if __name__ == '__main__':
    main()
