import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_utils as du

import torch
import torch.nn.functional as F


class ff_standart_neural_network:

    def __init__(self, activation_function, output_layer_function, Loss_function, layers_dimensions: list) -> None:
        self.activation_function = activation_function
        self.output_layer_function = output_layer_function
        self.Loss_function = Loss_function
        self.classes = layers_dimensions[-1]
        self.num_of_activation_layers = len(layers_dimensions) - 1
        self.weights = [np.ones((layers_dimensions[i+1], layers_dimensions[i]+1))  # biases are concatenated as a column to each weights matrix
                        for i in range(len(layers_dimensions) - 1)]
        # TODO: check if last layer weight matrix needed
        # self.biases = [np.zeros(d) for d in layers_dimensions[1:]] TODO delete

    def feed_forward(self, x):
        X = np.array([])
        # x has '1' concatenated at the bottom for bias
        x = np.concatenate(x, np.array[1])
        for layer in range(len(self.num_of_activation_layers)):
            x = self.activation_function(
                self.weights[layer] @ x)
            X = np.concatenate(X, np.array([x]), axis=1)
            x = np.concatenate(x, np.array[1])

        x = self.output_layer_function(self.weights[-1] @ x)
        X.np.concatenate(X, np.array([x]), axis=1)
        return X


    def Jac_f_by_x(W, z, activation_function_derivative, vec):
        diag = activation_function_derivative(z)
        jac = diag[:, np.newaxis] * W
        return jac.T @ vec

    def Jac_f_by_theta(n1, n2, x, z, activation_function_derivative, vec):
        grad_theta = np.array([])
        for i in range(n1):
            for j in range(n2):
                row = np.zeros((1, n2))
                row[i] = activation_function_derivative(z[j]) * x[i]
                np.concatenate(grad_theta, np.array([np.dot(row, vec)]))
        for i in range(n2):
            row = np.zeros((1, n2))
            row[i] = activation_function_derivative(z[j])
            np.concatenate(grad_theta, np.array([np.dot(row, vec)]))

        return grad_theta

    def Grad_F_by_Theta(self, X, loss_function_grad_theta, loss_function_grad_x, activation_function_derivative):
        grad = np.array([loss_function_grad_theta(X[:, -1])])
        back_prop_grad = loss_function_grad_x(self.weights[-1])
        for i, x in enumerate(1, X.reversed):  # TODO: fix loop condition
            z = self.weights[i] @ np.concatenate(x, np.array[1])
            z_next = self.weights[i+1] @ np.concatenate(X[i+1], np.array[1])
            back_prop_grad = ff_standart_neural_network.J
            (
                self.weights, z, activation_function_derivative, back_prop_grad)
            grad_theta_i = ff_standart_neural_network.Jac_f_by_theta(
                self.weights[i+1].shape[1] - 1, self.weights[i+1].shape[0], X[i+1], z_next, activation_function_derivative, back_prop_grad)
            grad = np.concatenate(grad, grad_theta_i)



    def fit(self,x_train,y_train, epochs=10, mini_batch_size=10):

        x_batches, y_batches = util.split_into_batches(x_train, y_train, mini_batch_size)
    
        for i in range(epochs):


            for x_batch, y_batch in zip(x_batches, y_batches):
                # feed foraward on all x_batch
                # compute loss
                # compute gradients (backpropagation)
                # update weights
                pass

                


    



