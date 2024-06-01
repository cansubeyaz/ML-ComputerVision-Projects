import numpy as np
from typing import Optional

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function should calculate the root mean squared error given target y and prediction y_pred

    Args:
        - y(np.array): target data
        - y_pred(np.array): predicted data

    Returns:
        - err (float): root mean squared error between y and y_pred

    """
    return np.sqrt(np.mean((y-y_pred)**2))

def split_data(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training, validation

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation

    """
    dataset = int(len(x)*0.80)
    #x_train = x[:dataset] #%80 of all dataset for training
    #y_train = y[:dataset] #%80 of all dataset for training
    #x_val = x[dataset:] #%20 of all dataset for training
    #y_val = y[dataset:] #%20 of all dataset for training
    x_train,y_train,x_val,y_val = x[:dataset],y[:dataset],x[dataset:],y[dataset:]
    return x_train, y_train, x_val, y_val