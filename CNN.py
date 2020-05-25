"""
-- *********************************************
-- Author       :	Fawaz Qutami
-- Create date  :   10th May 2020
-- Description  :   Convolution Neural Network Functions
-- File Name    :   CNN.py
-- *********************************************
"""

# load Packages
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD, RMSprop, Adam

from eHandler import PrintException as EH
from plotting import plotModel

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def CNN():
    """
    Convolution Neural Network
    :return: Model obj
    """
    try:
        """:
        There are two ways to build Keras models: sequential and functional.

        The sequential API allows you to create models layer-by-layer for most problems. 
        It is limited in that it does not allow you to create models that share 
        layers or have multiple inputs or outputs.
        """

        # Build the Model
        # ------------------------------------------------------------
        model = Sequential()
        # Convolution layer 1
        model.add(Conv2D(filters=16
                         , kernel_size=2
                         , input_shape=(100, 100, 3)
                         , activation='relu'
                         , padding='same'))
        # Pooling layer 1 - reduce the size of the input layer
        model.add(MaxPooling2D(pool_size=2))
        # Convolution layer 2
        model.add(Conv2D(filters=32
                         , kernel_size=2
                         , activation='relu'
                         , padding='same'))
        # Pooling layer 2 - reduce the size of the input layer
        model.add(MaxPooling2D(pool_size=2))
        # Convolution layer 3
        model.add(Conv2D(filters=64
                         , kernel_size=2
                         , activation='relu'
                         , padding='same'))
        # Pooling layer 3 - reduce the size of the input layer
        model.add(MaxPooling2D(pool_size=2))
        # Convolution layer 4
        model.add(Conv2D(filters=128
                         , kernel_size=2
                         , activation='relu'
                         , padding='same'))
        # Pooling layer 4 - reduce the size of the input layer
        model.add(MaxPooling2D(pool_size=2))
        # Dropout - regularization technique
        """
        Dropout is a technique where randomly selected neurons are ignored during training. 
        They are “dropped-out” randomly. This means that their contribution to the 
        activation of downstream neurons is temporally removed on the forward pass 
        and any weight updates are not applied to the neuron on the backward pass.
        """
        model.add(Dropout(0.3))
        # Flatten - converting Matrix to single array - remove all of the dimensions
        # except for one
        model.add(Flatten())
        # Fully connected layer1 -Dense  (262 neurons in the FIRST hidden layer)
        model.add(Dense(260, activation='relu'))
        # Dropout - regularization technique
        model.add(Dropout(0.4))
        # Fully connected layer 2 -Dense ((131 neurons in the SECOND hidden layer))
        model.add(Dense(131, activation='softmax'))  # softmax, sigmoid
        # ------------------------------------------------------------
        # Model Summary
        print("\n --- Model Summary:")
        print(model.summary())

        # Plot the model graph
        plotModel(model)

        # Compile the model
        compileModel(model)

        return model

    except :
        EH()


def compileModel(model):
    """
    Compile Model
    :param model: Model obj
    :return: None
    """
    try:
        # Compile the model
        print("\n --- Compiling the model ...")

        # The purpose of loss functions is to compute the quantity
        # that a model should seek to minimize during training:
        # 1. Gradient descent (with momentum) optimizer
        sgd = SGD(learning_rate=0.01)   #, momentum=0.8)
        # 2. RMSprop algorithm
        rmsprop = RMSprop(learning_rate=0.01)
        # 3. Adam optimization is a stochastic gradient descent method that is based
        # on adaptive estimation of first-order and second-order moments
        adam = Adam(learning_rate=0.001)

        # categorical_crossentropy: Computes the cross-entropy loss between true
        # labels and predicted labels.
        model.compile(loss='categorical_crossentropy',
                      optimizer= adam,  # sgd , rmsprop, adam
                      metrics=['accuracy'])
        print('Model Compiled!')

    except:
        EH()