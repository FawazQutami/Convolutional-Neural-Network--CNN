"""
-- *********************************************
-- Author       :	Fawaz Qutami
-- Create date  :   10th May 2020
-- Description  :   Plot Functions
-- File Name    :   plotting.py
-- *********************************************
"""

# load Packages
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

from eHandler import PrintException as EH


def plotPredictions(model, x_test, y_test, target_labels):
    """
    Plot predicted images
    :param model: Model obj
    :param x_test: list
    :param y_test: list
    :param target_labels: list
    :return: None
    """
    try:
        # Let's visualize test prediction.
        y_prediction = model.predict(x_test)

        # Plot a random sample of test images, predicted labels, and true labels
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Prediction of 25 images - chosen randomly)")

        # Create a sample of 20 images
        sample = np.random.choice(x_test.shape[0], size=12, replace=False)
        # Loop over the sample images
        for i, item in enumerate(sample):
            ax = fig.add_subplot(3, 4, i + 1
                                 , xticks=[] # get rid of ticks
                                 , yticks=[])
            ax.imshow(np.squeeze(x_test[item]))
            prediction_index = np.argmax(y_prediction[item])
            true_index = np.argmax(y_test[item])
            ax.set_title(f"{target_labels[true_index]} || "
                         f"Predicted:{target_labels[prediction_index]}"
                         , color=("green" if prediction_index == true_index else "red")
                         ,fontsize=9)
        plt.show()

    except:
        EH()


def plotModel(model):
    """
    Plot the model diagram
    :param model: Model obj
    :return: None
    """
    try:
        # Plot the model graph
        print("\n --- Store the model graph in the 'data' folder...")
        plot_model(model
                   , to_file="data/model.png"
                   , show_shapes=False
                   , show_layer_names=True
                   , rankdir="TB"
                   , expand_nested=False
                   , dpi=96, )
        print('Done!')

    except:
        EH()


def showImage(X_train, target_labels):
    """
    Plot a Sample images
    :param X_train: list
    :return: None
    """
    try:
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle("Sample of 16 image - chosen randomly")
        # Create a sample of 16 images
        sample = np.random.choice(X_train.shape[0], size=12, replace=False)
        for i, item in enumerate(sample):
            ax = fig.add_subplot(3, 4, i + 1
                                 , xticks=[]
                                 , yticks=[])
            ax.imshow(np.squeeze(X_train[i]))
            index = np.argmax(X_train[item])
            ax.set_title("{}".format(target_labels[index]), color="green")
        plt.show()

    except:
        EH()


def visualizeLoss_Acc(modelFitting):
    """
    Plot loss and accuracy vs epochs
    :param modelFitting: Fit obj
    :return: None
    """
    try:
        # Visualize the loss and accuracy vs epochs

        #fig = plt.figure(figsize=(7, 5))
        fig = plt.figure(figsize=(14, 5), constrained_layout=True)

        ax1, ax2 = fig.subplots(1, 2)
        fig.suptitle("Performance of the Train and Test sets\n Accuracy and Loss vs Epochs")
        # Plot accuracy vs epochs
        plt.setp(ax1.set_title("Model Accuracy"), color='b')
        ax1.plot(modelFitting.history['accuracy'])
        ax1.plot(modelFitting.history['val_accuracy'])
        ax1.set_ylabel("Accuracy")
        ax1.yaxis.label.set_color('red')
        ax1.set_xlabel("Epochs")
        ax1.xaxis.label.set_color('red')
        ax1.legend(['Train', 'Test'])
        # Plot loss vs epochs
        plt.setp(ax2.set_title("Model Loss"), color='b')
        ax2.plot(modelFitting.history['loss'])
        ax2.plot(modelFitting.history['val_loss'])
        ax2.set_ylabel("Loss")
        ax2.yaxis.label.set_color('red')
        ax2.set_xlabel("Epochs")
        ax2.xaxis.label.set_color('red')
        ax2.legend(['Train', 'Test'])

        plt.show()

    except:
        EH()
