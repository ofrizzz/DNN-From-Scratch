import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util

class ff_residual_neural_network:

    def __init__(self, activation_function, output_layer_function, Loss_function,  layers_dimensions: list) -> None:
        self.activation_function = activation_function
        self.output_layer_function = output_layer_function
        self.Loss_function = Loss_function
        self.classes = layers_dimensions[-1]
        self.num_of_activation_layers = len(layers_dimensions) - 1
        self.W1s = [np.ones((layers_dimensions[i+1], layers_dimensions[i]+1))  # biases are concatenated as a column to each weights matrix
                        for i in range(len(layers_dimensions) - 1)]
        self.W2s = [np.ones((layers_dimensions[i], layers_dimensions[i+1]+1))  # biases are concatenated as a column to each weights matrix
                        for i in range(len(layers_dimensions) - 1)]
        # TODO: check if last layer weight matrix needed
        # self.biases = [np.zeros(d) for d in layers_dimensions[1:]] TODO delete

    def feed_forward(self, x):
        X = np.array([])
        # x has '1' concatenated at the bottom for bias
        # x = np.concatenate(x, np.array[1])
        for layer in range(len(self.num_of_activation_layers)):

            W1 = self.W1s[layer]
            W2 = self.W2s[layer]
            x1 = np.concatenate(x, np.array[1])
            z1 = self.activation_function(W1 @ x1)
            z1 = np.concatenate(z1, np.array[1])
            x = self.activation_function(x+ W2 @ z1)
            X = np.concatenate(X, np.array([x]), axis=1)
        
        # W1 = self.W1s[-1]
        # W2 = self.W2s[-1]
        # x1 = np.concatenate(x, np.array[1])
        # z1 = self.activation_function(W1 @ x)
        # z1 = np.concatenate(z1, np.array[1])
        # x = self.output_layer_function(x + W2 @ z1)
        # X.np.concatenate(X, np.array([x]), axis=1)
            
        x = x.concatenate(x, np.array[1])
        x = self.output_layer_function(self.W1s[-1] @ x)
        X.np.concatenate(X, np.array([x]), axis=1)  
        return X
    
    #TODO: optional changa diag to something more efiicient
    def Jac_f_by_x(W1, W2, x, activation_function_derivative, vec):
        n1 = W1.shape[1]
        n2 = W1.shape[0]
        x_t = np.concatenate(x, np.array([1]))
        sig_tag = np.diag(activation_function_derivative(W1 @ x_t))
        
        jac = np.eye(n1) + W2 @ sig_tag @ W1
        return jac @ vec

    def Jac_f_by_W1(n1, n2, x, activation_function_derivative, vec):
        pass

    def Jac_f_by_W2(W1, W2, x, activation_function, activation_function_derivative, vec):
        
        z1 = activation_function(W1 @ x)
        pass
    

    def fit(self,x_train,y_train, epochs=10, mini_batch_size=10):

        x_batches, y_batches = util.split_into_batches(x_train, y_train, mini_batch_size)
        
        for i in range(epochs):


            for x_batch, y_batch in zip(x_batches, y_batches):
                # feed foraward on all x_batch
                # compute loss
                # compute gradients (backpropagation)
                # update weights
                pass
            