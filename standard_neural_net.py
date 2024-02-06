import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_utils as du

import torch
import torch.nn.functional as F

activation_function = util.relu
output_layer_function = util.soft_max_loss
activation_function_derivative = util.relu_derivative
loss_function_grad_theta = util.soft_max_regression_grad_by_theta
loss_function_grad_x = util.soft_max_regression_grad_by_x

class ff_standard_neural_network:
    def __init__(self, input_dimension, hidden_layers_dimensions, output_layer_dimension) -> None:
        self.input_dimension = input_dimension
        self.hidden_layers_dimensions = hidden_layers_dimensions
        self.output_layer_dimension = output_layer_dimension
        self.num_of_layers = len(hidden_layers_dimensions) + 2

        self.weights = [np.random.rand(hidden_layers_dimensions[0], input_dimension + 1)] + \
                [np.random.rand(hidden_layers_dimensions[i], hidden_layers_dimensions[i-1] + 1) for i in range(1, len(hidden_layers_dimensions))] + \
                [np.random.rand(output_layer_dimension, hidden_layers_dimensions[-1] + 1)]

        # self.derivatives = [np.zeros((hidden_layers_dimensions[i], hidden_layers_dimensions[i-1] + 1)) for i in range(self.num_of_layers-1)]
        self.activations = []
        



    def feed_forward(self, X):
        self.activations = [X]
        #x is a batch of inputs
        # concatenate 1 at the bottom of x for bias
        for layer in range(self.num_of_layers - 1):
            X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
            X = self.weights[layer] @ X
            if layer < self.num_of_layers - 2:
                X = activation_function(X)
            else:
                X = output_layer_function(X,self.weights[-1])
            self.activations.append(X)
        # activations[i] is the output of layer i
        return X


    def fit(self,x_train,y_train, epochs=10, mini_batch_size=10):

        x_batches, y_batches = util.split_into_batches(x_train, y_train, mini_batch_size)
        
        for i in range(epochs):


            for x_batch, y_batch in zip(x_batches, y_batches):
                outputs = self.feed_forward(x_batch)
                # compute loss
                # compute gradients (backpropagation)
                # update weights
                pass



    

    # def Jac_f_by_x(W, z, vec):
    #     diag = activation_function_derivative(z)
    #     jac = diag[:, np.newaxis] * W
    #     return jac.T @ vec

    # def Jac_f_by_theta(n1, n2, x, z, activation_function_derivative, vec):
    #     grad_theta = np.array([])
    #     for i in range(n1):
    #         for j in range(n2):
    #             row = np.zeros((1, n2))
    #             row[i] = activation_function_derivative(z[j]) * x[i]
    #             np.concatenate(grad_theta, np.array([np.dot(row, vec)]))
    #     for i in range(n2):
    #         row = np.zeros((1, n2))
    #         row[i] = activation_function_derivative(z[j])
    #         np.concatenate(grad_theta, np.array([np.dot(row, vec)]))
    #     return grad_theta
    

    def Grad_F_by_Theta(self,C):
        grad = np.array([loss_function_grad_theta(self.activations[-2], self.weights[-1], C)])
        grads_Ws= [grad]
        back_prop_grad = loss_function_grad_x(self.activations[-2], self.weights[-1], C)

        for X, weights in reversed(zip(self.activations[:-1], self.weights)):
            grad = util.Jac_f_by_theta(X, weights, back_prop_grad)
            grads_Ws.insert(0, grad)
            back_prop_grad = util.Jac_f_by_x(X, weights, back_prop_grad)
        
        return grads_Ws

# Sample test input to initialize ff_standard_neural_network class

input_dimension = 5  # For example, a neural network with 5 input features
hidden_layers_dimensions = [4, 3]  # Two hidden layers, first with 4 neurons and second with 3 neurons
output_layer_dimension = 2  # Assuming a binary classification task, so 2 output neurons

# Initialize the ff_standard_neural_network instance
network = ff_standard_neural_network(input_dimension, hidden_layers_dimensions, output_layer_dimension)
# print([network.weights[i].shape for i in range(len(network.weights))])

network = ff_standard_neural_network(3, [2], 3)

# Test input: batch of 2 vectors, each with 3 features
input_batch = np.array([
    [0.5, -0.5, 0.3],
    [0.1, 0.2, -0.1]
])

# Execute feed_forward
network.feed_forward(input_batch.T)
activations = network.activations
# Display the output
for i, activation in enumerate(activations):
    print(f"Layer {i+1} activations:\n{activation}\n")
