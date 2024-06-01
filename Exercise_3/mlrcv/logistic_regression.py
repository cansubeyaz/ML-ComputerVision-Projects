import numpy as np
import matplotlib.pyplot as plt
from mlrcv.core import *
from mlrcv.utils import *
import typing

class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
        y_pred = None

        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der (np.ndarray): first derivative value
        """
        der = None

        return der

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta parameters that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """
        
        acc = None

        return acc

class MultiClassLogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_class = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
        y_pred = None

        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y,
        for each possible class.

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der: first derivative value
        """
        der = None

        return der

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta_class parameters (multiclass) that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """

        acc = None

        return acc