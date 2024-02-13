import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util

activation_function = np.tanh
output_layer_function = util.soft_max_loss


class ff_residual_neural_network:

    def __init__(self, input_dimension, hidden_layers_dimensions, output_layer_dimension) -> None:
        self.input_dimension = input_dimension
        self.hidden_layers_dimensions = hidden_layers_dimensions
        self.output_layer_dimension = output_layer_dimension
        self.num_of_layers = len(hidden_layers_dimensions) + 2
        self.activations = []
        if len(hidden_layers_dimensions) > 0:
            self.W1s = [np.random.rand(hidden_layers_dimensions[i], input_dimension)
                        for i in range(len(hidden_layers_dimensions))]

            self.bs = [np.random.rand(hidden_layers_dimensions[i], 1)
                       for i in range(len(hidden_layers_dimensions))]

            self.W2s = [np.random.rand(input_dimension, hidden_layers_dimensions[i])
                        for i in range(len(hidden_layers_dimensions))]

            self.output_layer_W = np.random.rand(
                output_layer_dimension, hidden_layers_dimensions[-1])

            self.output_layer_b = np.random.rand(output_layer_dimension, 1)

        else:
            self.output_layer_W = np.random.rand(
                output_layer_dimension, input_dimension)

            self.output_layer_b = np.random.rand(output_layer_dimension, 1)

    def feed_forward(self, x, y):

        for i in range(len(self.hidden_layers_dimensions)):
            x = x + \
                self.W2s[i] @ activation_function(self.W1s[i] @ x + self.bs[i])

        return output_layer_function(self.output_layer_W @ x + self.output_layer_b, y)

    # TODO:

    def fit(self, x_train, y_train, epochs=10, mini_batch_size=10):

        x_batches, y_batches = util.split_into_batches(
            x_train, y_train, mini_batch_size)

        for i in range(epochs):

            for x_batch, y_batch in zip(x_batches, y_batches):

                # feed foraward on all x_batch
                # compute loss
                # compute gradients (backpropagation)
                # update weights
                pass
