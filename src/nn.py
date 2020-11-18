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
        delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])
        # derivative w.r.t last layer params
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # loop through the rest of the network
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) *\
            sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Number of test inputs for which the network produces the correct
        output.
        Assume output is the index with the highest activation
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in
                test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_prime(self, pred, true):
        """
        Derivative of cost function
        Assumes distance squared between predicted and true values as the cost
        function.
        """
        return (pred - true)

    def SGD(self, training_data, mini_batch_size, epochs, eta):
        """
        Trains neural network using stochastic gradient descent by calling
        update_mini_batch() function.
        training_data is tuple (input, desired_output),
        epochs is number of training epochs,
        eta is learning rate.
        """
        n = len(list(training_data))
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                    range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        """
        Applies gradient descent for a mini_batch.
        Meant to be called by SGD() function.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in
                zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in
                zip(self.biases, nabla_b)]


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
