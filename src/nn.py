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
    def __init__(self):
        """
        Initialise default cost function and activation
        """
        self.cost_func = QuadraticCost

    def set_cost(self, cost_func):
        """
        Sets the cost function to something other than default.
        cost_func should be a class containing 'fn' and 'delta' static methods
        """
        self.cost_func = cost_func

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

    def feed_forward(self, a):
        """
        Computes the output of the network given an input a
        a is a numpy array of size n where n is dimension of input space
        Can also take numpy array of size (n, m) where m is the number of
        data points in a batch.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """
        Implements back propagation
        Returns a tuple of gradients of cost function w.r.t each parameter,
        (nabla_b, nabla_w), which are of the same shape as self.biases,
        self.weights.
        Vectorised so that x and y are matrices with N column vectors for the N
        samples in the batch.
        Input: x is input to the network, y is correct output.
        """
        # lists to store derivatives of cost function w.r.t each weight and bias
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feed forward and store all activations:
        activation = x
        activations = [x] # store the sigmoid(z) variables
        zs = [] # store the w.x+b variables
        for b, w in list(zip(self.biases, self.weights)):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # go backward and store results in nabla_{b,w}
        delta = self.cost_func.delta(zs[-1], activations[-1], y)
        # derivative w.r.t last layer params
        nabla_b[-1] = np.sum(delta, axis=1).reshape((self.layer_sizes[-1],1))
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # loop through the rest of the network
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) *\
            sigmoid_prime(z)
            nabla_b[-l] = np.sum(delta, axis=1).reshape((self.layer_sizes[-l],1))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Number of test inputs for which the network produces the correct
        output.
        Assume output is the index with the highest activation
        """
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in
                test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, mini_batch_size, epochs, eta, test_data=None):
        """
        Trains neural network using stochastic gradient descent by calling
        update_mini_batch() function.
        training_data is tuple (input, desired_output),
        epochs is number of training epochs,
        eta is learning rate.
        If test_data supplied, will return list of progress.
        """
        if test_data:
            progress = []
            n_test = len(test_data)
        n = len(list(training_data))
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                    range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print('Epoch ', j)
            if test_data:
                progress.append(self.evaluate(test_data)/n_test)
        if test_data:
            return progress

    def update_mini_batch(self, mini_batch, eta):
        """
        Applies gradient descent for a mini_batch.
        Meant to be called by SGD() function.
        Vectorised, so that it passes the whole batch to backprop at once.
        """
        X = np.array([sample[0].reshape(self.layer_sizes[0]) for sample in
            mini_batch]).transpose()
        Y = np.array([sample[1].reshape(self.layer_sizes[-1]) for sample in
            mini_batch]).transpose()
        nabla_b, nabla_w = self.backprop(X, Y)
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in
                zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in
                zip(self.biases, nabla_b)]

class QuadraticCost:
    """
    Class to contain the cost function and derivative for quadratic cost.
    """
    @staticmethod
    def fn(a, y):
        """
        Cost function between output a and desired output y.
        """
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """
        Error derivative from output layer
        """
        return (a - y) * sigmoid_prime(z)

class CrossEntropyCost:
    """
    Class to contain the cost function and derivative for cross-entropy.
    """
    @staticmethod
    def fn(a, y):
        """
        Cost function between output a and desired output y.
        Note that since y is either 0 or 1, only one of the terms is active.
        The np.nan_to_num corrects for the nan that occurs in log(0).
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Error derivative.
        z is passed only for compatibility with quadratic cost.
        """
        return (a - y)

def sigmoid(z):
    """
    Sigmoid function of variable z
    """
    return 1./(1. + np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of sigmoid
    """
    return sigmoid(z) * (1. - sigmoid(z))
