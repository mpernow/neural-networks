"""
nn.py

Contains class definition of neural network
and some useful functions
"""

import numpy as np

class NN:
    """
    Neural network class definition
    """
    def set_layers(self, layers):
        """
        Takes list of integers [a, b, c, ...]
        and sets the layers of the network to be of size a, b, c, ...
        """
        self.num_layers = len(layers)
        self.layer_sizes = layers

    def init_weights_and_biases(self):
        """
        Initialises the weights and biases of the network
        using standard normal distribution
        """
        # biases not set for first layer
        self.biases = [np.random.randn(l, 1) for l in self.layer_sizes[1:]]
        self.weights = [np.random.randn(l2, l1) for l1, l2 in
                zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def feed_forward(self):
        """
        Computes the output of the network given an input a
        a is a numpy array of size n where n is dimension of input space
        Can also take numpy array of size (n, m) where m is the number of
        data points in a batch.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


def sigmoid(z):
    """
    Sigmoid function of variable z
    """
    return 1./(1. + np.exp(-z))
