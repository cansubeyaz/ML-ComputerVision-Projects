import numpy as np
from typing import Optional
from mlrcv.utils import *

class LinearRegression:
    def __init__(self):
        self.theta_0 = None
        self.theta_1 = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray):
        """
        This function should calculate the parameters theta0 and theta1 for the regression line

        Args:
            - x (np.array): input data
            - y (np.array): target data

        """
        xm,ym,xym,xxm = np.mean(x),np.mean(y),np.mean(x*y),np.mean(x**2)
        self.theta_0 = (ym-((xym-xm*ym)/(xxm-(xm**2)))*xm)
        self.theta_1 = ((xym-(xm*ym))/(xxm-(xm**2)))
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta0 and theta1 to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y_pred: y computed w.r.t. to input x and model theta0 and theta1

        """
        y_pred = x * self.theta_1 + self.theta_0 #linear function
        return y_pred

class NonLinearRegression:
    def __init__(self):
        self.theta = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray, degree: Optional[int] = 2):
        """
        This function should calculate the parameters theta for the regression curve.
        In this case there should be a vector with the theta parameters (len(parameters)=degree + 1).

        Args:
            - x: input data
            - y: target data
            - degree (int): degree of the polynomial curve

        Returns:

        """
        matrix_vander = np.vander(x,degree + 1,increasing="increasing") #Generate Vandermonde matrix
        self.theta = np.linalg.lstsq(matrix_vander,y,rcond=None)[0] #To fıt a lınear model to the data
        self.matrix_vander = np.vander(x,degree + 1,increasing= "increasing")
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y: y computed w.r.t. to input x and model theta parameters
        """
        return np.dot(self.matrix_vander, self.theta)
