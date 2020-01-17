import _pickle
import gzip
import numpy as np
import random

class neural_network(object):
    def __init__(self, layers):
        self.num_of_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, x):
        for hidden_layer in range(0, self.num_of_layers - 1):
            w = self.weights[hidden_layer]
            x = np.dot(w, x)
            b = self.biases[hidden_layer]
            z = np.add(x, b)

        a = sigmoid(z)

        return a

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for x, y in test_data
        ]

        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, a, y):
        # Return the vector of partial derivatives \partial C_x / \partial a 
        # for the output activations a.
        return (a - y)

    def backpropagation(self, x, y):
        # Returns the appropriate gradient for the cost 
        # associated to the training example x

        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        a = x
        a_layers = [a] # Store all the a, layer by layer
        z_layers = [] # store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.add(np.dot(w, a), b)
            z_layers.append(z)
            a = sigmoid(z)
            a_layers.append(a)

        # Backward
        delta_cost = self.cost_derivative(a_layers[-1], y) * sigmoid_derivative(z_layers[-1])
        delta_biases[-1] = delta_cost
        delta_weights[-1] = np.dot(delta_cost, a_layers[-2].transpose())

        for layer in range(2, self.num_of_layers):
            z = z_layers[-layer]
            a = sigmoid_derivative(z)

            delta_cost = np.dot(self.weights[-layer + 1].transpose(), delta_cost) * a
            delta_biases[-layer] = delta_cost
            delta_weights[-layer] = np.dot(delta_cost, a_layers[-layer - 1].transpose())

        return (delta_biases, delta_weights)

    def update_mini_batch(self, mini_batch, eta):
        # Update the network's weights and biases by applying
        # gradient descent using backpropagation to a single mini batch.

        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_biases, delta_weights = self.backpropagation(x, y)

            new_biases = [(new_b + delta_b) for new_b, delta_b in zip(new_biases, delta_biases)]
            new_weights = [(new_w + delta_w) for new_w, delta_w in zip(new_weights, delta_weights)]

        self.biases = [
            old_bias - (eta / len(mini_batch) * new_bias)
            for old_bias, new_bias in zip(self.biases, new_biases)
        ]

        self.weights = [
            old_weight - (eta / len(mini_batch) * new_weight)
            for old_weight, new_weight in zip(self.weights, new_weights)
        ]

        return None

    def sgd_training(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        # Train the neural network using mini-batch
        # stochastic gradient descent.
        
        if test_data:
            test_data_size = len(test_data)

        training_data_size = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[it : (it + mini_batch_size)]
                for it in range(0, training_data_size, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluate(test_data), test_data_size
                ))
            else:
                print("Epoch {0} completed!".format(epoch))

        return None

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def vectorized_result(y):
    vect = np.zeros((10, 1))
    vect[y] = 1.0

    return vect

def load_data():
    training_data = validation_data = test_data = None

    with gzip.open('mnist.pkl.gz', 'rb') as handler:
        training_set, validation_set, test_set = _pickle.load(handler, encoding = 'latin1')

        training_inputs = [np.reshape(x, (784, 1)) for x in training_set[0]]
        training_results = [vectorized_result(y) for y in training_set[1]]
        training_data = list(zip(training_inputs, training_results))

        validation_inputs = [np.reshape(x, (784, 1)) for x in validation_set[0]]
        validation_data = list(zip(validation_inputs, validation_set[1]))

        test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
        test_data = list(zip(test_inputs, test_set[1]))

        handler.close()

    return (training_data, validation_data, test_data)

def main():
    training_data, validation_data, test_data = load_data()

    if (training_data != None) and (validation_data != None) and (test_data != None):
        neural_network_obj = neural_network([784, 30, 10])
        neural_network_obj.sgd_training(training_data, 10, 10, 0.5, test_data = test_data)
    else:
        print('Err @ Invalid Data!')

    return None

main()