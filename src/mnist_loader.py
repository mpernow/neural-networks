"""
mnist_loader

Library to load the MNIST data in the format made available bi Michael Nielsen
at https://github.com/mnielsen/neural-networks-and-deep-learning.
This code is based on his.
"""

import pickle
import gzip
import numpy as np

def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the training images as a numpy ndarray 
    with 50,000 entries.  Each entry is a numpy ndarray with 784 values, 
    representing the 28 * 28 = 784 pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries. Each entry is the digit value (0...9) 
    for the corresponding image contained in the first entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    For ease of use, this should be called by load_data_wrapper() below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Calls the load_data() above and returns it in a more useful
    format.

    All the outputs variables are lists containing a 2-tuple for each sample.
    The first ontry in the 2-tuple is a 784-dimensional numpy ndarray
    representing the pixel values. The second entry is a 10-dimensional vector
    corresponding to the result for the training_data. For the validation_data
    and test_data, the second entry is instead an integer corresponding to the
    correct result.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
