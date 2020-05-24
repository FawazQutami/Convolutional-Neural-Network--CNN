"""
-- *********************************************
-- Author       :	Fawaz Qutami
-- Create date  :   10th May 2020
-- Description  :   Data Preparation Functions
-- File Name    :   data.py
-- *********************************************
"""

# load Packages
from sklearn.datasets import load_files
import numpy as np
#from os import path
import pickle
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img

from eHandler import PrintException as EH
from plotting import showImage


def loadData(path):
    """
    Load Data
    :param path: string
    :return: img, list, list[string]
    """
    try:
        # load files from a path
        imageData = load_files(path)
        # Set the image file name
        files = np.array(imageData['filenames'])
        # print(f"\n Images files: ------------ \n {files}")
        # # Set the target array
        targets = np.array(imageData['target'])
        # print(f"\n Targets: ------------ \n {targets}")
        # Set the target labels
        targetLabels = np.array(imageData['target_names'])
        # print(f"\n Targets labels: ------------ \n {targetLabels}")

        return files, targets, targetLabels

    except:
        EH()


def convertImageToArray(files):
    """
    Convert Image to Array
    :param files: list[string]
    :return: array
    """
    try:

        images_as_array = []
        for file in files:
            # Convert to Numpy Array
            images_as_array.append(img_to_array(load_img(file)))

        return np.array(images_as_array)

    except:
        EH()


def scaling(item):
    """
    Data Scaling
    :param item: list
    :return: list
    """
    try:
        # Re-scale - pixel values between (0 and 1)
        item = item.astype('float64') / 255

        return item

    except:
        EH()


def xData(X_train, X_validation, X_test):
    """
    Prepare X data (X_train, X_validation, X_test)
    :param X_train: list
    :param X_validation: list
    :param X_test: list
    :return: lists
    """
    try:
        # Convert X data into pixel matrix
        print("\n --- Convert X data into pixel matrix ...")
        X_train = convertImageToArray(X_train)
        X_validation = convertImageToArray(X_validation)
        X_test = convertImageToArray(X_test)
        # Scale X data
        print("\n --- Scale X data - normalizing ...")
        X_train = scaling(X_train)
        X_validation = scaling(X_validation)
        X_test = scaling(X_test)

        return X_train, X_validation, X_test

    except:
        EH()


def savePickle(path, dicts):
    """
    Save Pickle File
    :param path: string
    :param dicts: dict
    :return: None
    """
    try:
        # Save a dictionary into a pickle file
        with open (path, "wb") as file:
            pickle.dump(dicts, file)

    except:
        EH()


def loadPickle(path):
    """
    Load Pickle File
    :param path: string
    :return: dict
    """
    try:
        # Save a dictionary into a pickle file
        with open ( path, "rb" ) as file:
            dicts = pickle.load(file)

        return dicts

    except:
        EH()


def dataPreparation():
    """
    Data Processing
    :return: arrays
    """
    try:
        # Set the training and test directories
        train_dir = 'data/fruits/Training'
        test_dir = 'data/fruits/Test'
        # Load and Split the training set to X_train, y_train, target_labels
        X_train, y_train, target_labels = loadData(train_dir)
        # Load and Split the test set to X_train, y_train, target_labels
        X_test, y_test, _ = loadData(test_dir)

        print('\n --- Loading training and testing data has been completed!')
        print('Training set size : ', X_train.shape[0])
        print('Testing set size  : ', X_test.shape[0])

        # number of classes
        classes =  len(np.unique(y_train))
        print(f"\n --- No of Classes: {classes}")

        # Convert to a categorical
        y_train = np_utils.to_categorical(y_train, classes)
        y_test = np_utils.to_categorical(y_test, classes)

        # Divide the test set into test set and validation set
        testLength = int(len(X_test) / 2)
        X_test, X_validation = X_test[testLength:], X_test[:testLength]
        y_test, y_validation = y_test[testLength:], y_test[:testLength]
        print("\n --- Divide the test set:")
        print('Validation X : ', X_validation.shape)
        print('Validation y :', y_validation.shape)
        print('Test X : ', X_test.shape)
        print('Test y : ', y_test.shape)

        # X data prepossessing
        X_train, X_validation, X_test = xData(X_train, X_validation, X_test)

        showImage(X_train, target_labels)

        return X_train, y_train, X_validation, y_validation, X_test, y_test, target_labels

    except:
        EH()