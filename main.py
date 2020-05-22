"""
-- *********************************************
-- Author       :	Fawaz Qutami
-- Create date  :   10th May 2020
-- Description  :   Main Functions
-- File Name    :   main.py
-- *********************************************
"""

# load Packages
import datetime as dt
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

from eHandler import PrintException as EH
from data import dataPreparation
from CNN import CNN
from setup import install_required_Packages
from plotting import plotPredictions, visualizeLoss_Acc


def checkPoints(model, X_train, y_train, X_validation, y_validation):
    """
    Model Fit and Check Points
    :param model: obj
    :param X_train: list
    :param y_train: list
    :param X_validation: list
    :param y_validation: list
    :return: Fit obj
    """
    try:
        print("\n --- Check Points ... ")

        # Create and save check points
        checks = ModelCheckpoint(filepath='data/cnn_fruits.hdf5'
                                       , verbose=1
                                       , save_best_only=True)

        # Fit the model with batch size =32 and epoch 25
        modelFitting = model.fit(X_train, y_train
                            , batch_size=32
                            , epochs=15
                            , validation_data=(X_validation, y_validation)
                            , callbacks=[checks]
                            , verbose=2
                            , shuffle=True)

        return modelFitting

    except:
        EH()


def modelScores(model, X_test, y_test):
    """
    Model Accuracy Score
    :param model: obj
    :param X_test: list
    :param y_test: list
    :return: None
    """
    try:
        print("\n --- Accuracy Score ... ")
        # First load pre-trained weights
        model.load_weights('data/cnn_fruits.hdf5')

        # Evaluate and print test accuracy
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Accuracy Score: {score[1]*100:.2f}%')

    except:
        EH()


def main():
    """
    Main Function
    :return: None
    """
    try:
        # Execution Time Start
        start = dt.datetime.now()

        # Data Preparation
        X_train, y_train, X_validation, y_validation, X_test, y_test, target_labels \
                = dataPreparation()
        
        # Build the model
        model = CNN()

        # Model Check Points
        modelFitting = checkPoints(model, X_train, y_train, X_validation, y_validation)

        # Accuracy Score
        modelScores(model, X_test, y_test)

        # Plot the Predictions
        plotPredictions(model, X_test, y_test, target_labels)

        # Plot the loss and accuracy vs epochs
        visualizeLoss_Acc(modelFitting)

        # Execution Time End
        end = dt.datetime.now()  # time.time()
        executionTime = (end - start).seconds
        print(f'\n --- Total Execution Time - in minutes: {executionTime/60:.3f}')

    except :
        EH()


if __name__ == "__main__":
    try:
        # Install Required packages
        install_required_Packages()
        main()

    except :
        EH()