import numpy as np
import pandas as pd
from mlrcv.core import *
from typing import Optional

class KMeans:
    def __init__(self, k_clusters: int, max_iter: Optional[int] = 300, init_method: Optional[str] = 'rand'):
        """
        This function initializes the KMeans method:

        Args:
            - k_clusters (int): number of cluster to divide your data
            - max_iter (int): max number of iterations to define the clusters
            - init_method (str): method to initialize the cluster centers (rand or k++)

        Returns:
        """
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.k_centers = None
        self.init_method = init_method
        self.init_dict = {'rand': self.init_random_centers, 'k++': self.init_centers_kpp, 'fixed': self.init_fixed_centers}

    def euclidean_distance(self, x: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        This function computes the euclidean distance between every point in x to the center:

        Args:
            - x (np.ndarray): input points x
            - center (np.ndarray): cluster center to compute the euclidean distance

        Returns:
           - dist (np.ndarray): the euclidean distance of each x point to the center 
        """

        dist = None

        return dist

    def init_fixed_centers(self, _):
        """
        No need to implement anything, this just initialize the centers with an example of failing case when initializing with random seeds
        """
        self.k_centers = np.array([[0.12334165, -0.04550881], [0.13236559, -1.42220759], [0.12894003, -1.42228261]])

    def init_centers_kpp(self, x: np.ndarray):
        """
        This function initialize the clusters centers using the KMeans++ method (farthest point sampling). Instead of 
        randomly initialize the centers this method gets the first center as a random sample of x then computes the
        of every other x sample to the already defined centers, then use the calculated distance as a probability
        distribution to pick other sample as the next center (repeat until define the K initial centers):

        Args:
            - x (np.ndarray): input points x

        Returns:
        """

        # implement here your function
        
        """
        plot the initial centers, should be at the end of your function so you can see where are the initial centers
        """
        print('Initial centers from k++ initialization')
        plot_clusters(x, self)

    def init_random_centers(self, x: np.ndarray):
        """
        This function randomly initializes the K initial centers

        Args:
            - x (np.ndarray): input points x

        Returns:
        """

        # implement here your function
        
        """
        plot the initial centers, should be at the end of your function so you can see where are the initial centers
        """
        print('Initial centers from random initialization')
        plot_clusters(x, self)

    def fit(self, x: np.ndarray):
        """
        This function uses the input x data to define the K clusters centers. Iterate over x, compute the points
        distances to the center, assign the clusters points, and update the clusters centers (repeat until convergence
        or reach the max_iter):

        Args:
            - x (np.ndarray): input points x

        Returns:
        """

        # implement here your function

        """
        you can use the plot_clusters(x, self) function call inside your training loop to check how the 
        clusters behave while updating
        """
        # plot_clusters(x, self)

    def predict_c(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts to which cluster each point in x belongs

        Args:
            - x (np.ndarray): input points x

        Returns:
            - c_pred (np.ndarray): assigned clusters to each point in x
        """

        c_pred = None

        return c_pred
