import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util
import resNet_util
import data_utils as du

activation_function = np.tanh
output_layer_function = resNet_util.soft_max_loss


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
                output_layer_dimension, input_dimension)

            self.output_layer_b = np.random.rand(output_layer_dimension, 1)

        else:
            self.output_layer_W = np.random.rand(
                output_layer_dimension, input_dimension)

            self.output_layer_b = np.random.rand(output_layer_dimension, 1)

    def feed_forward(self, x, y, W1s=None, W2s=None, bs=None, output_layer_W=None, output_layer_b=None, use_stored_weights=True):
        if use_stored_weights:
            W1s = self.W1s
            W2s = self.W2s
            bs = self.bs
            output_layer_W = self.output_layer_W
            output_layer_b = self.output_layer_b

        self.activations = [x]
        for i in range(len(self.hidden_layers_dimensions)):

            x = x + \
                W2s[i] @ activation_function(W1s[i] @ x + bs[i])
            self.activations.append(x)
        x = output_layer_function(x, output_layer_W, output_layer_b, y)
        return x

    def Grad_F_by_Theta(self, y):
        grad_Loss_by_W = resNet_util.soft_max_regression_grad_by_W(self.activations[-1],
                                                                   self.output_layer_W,
                                                                   self.output_layer_b, y)
        grads_Loss_by_b = resNet_util.soft_max_regression_grad_by_b(self.activations[-1],
                                                                    self.output_layer_W,
                                                                    self.output_layer_b, y)
        grads_by_W1 = []
        grads_by_W2 = []
        grads_by_b = []
        back_prop_grad = resNet_util.soft_max_regression_grad_by_x(self.activations[-1],
                                                                   self.output_layer_W,
                                                                   self.output_layer_b, y)

        for x, W1, W2, b in reversed(list(zip(self.activations[:-1], self.W1s, self.W2s, self.bs))):
            grads_by_b.insert(0, resNet_util.Jac_f_by_b(
                x, W1, W2, b).T @ back_prop_grad)
            grads_by_W1.insert(0, np.reshape(resNet_util.Jac_f_by_W1(
                x, W1, W2, b).T @ back_prop_grad, W1.shape, order='F'))
            grads_by_W2.insert(0, np.reshape(resNet_util.Jac_f_by_W2(
                x, W1, W2, b).T @ back_prop_grad, W2.shape, order='F'))
            back_prop_grad = resNet_util.Jac_f_by_x(
                x, W1, W2, b).T @ back_prop_grad
        return (grads_by_W1, grads_by_W2, grads_by_b, grad_Loss_by_W, grads_Loss_by_b)

    def gradient_test_resnet(self, iterations=20):
        x = np.random.rand(self.input_dimension, 1)
        y = np.zeros((self.output_layer_dimension, 1))
        y[np.random.randint(0, self.output_layer_dimension)][0] = 1

        d0_W1 = [np.random.rand(W.shape[0], W.shape[1]) for W in self.W1s]
        d0_W2 = [np.random.rand(W.shape[0], W.shape[1]) for W in self.W2s]
        d0_b = [np.random.rand(W.shape[0], W.shape[1]) for W in self.bs]
        d0_out_layer_W = np.random.rand(
            self.output_layer_W.shape[0], self.output_layer_W.shape[1])
        d0_out_layer_b = np.random.rand(
            self.output_layer_b.shape[0], self.output_layer_b.shape[1])

        eps0 = 0.5
        F0 = self.feed_forward(x, y)
        g0_W1, g0_W2, g0_b, g0_out_layer_W, g0_out_layer_b = self.Grad_F_by_Theta(
            y)

        y1 = []
        y2 = []
        for i in range(iterations):
            epsilon = eps0 ** i
            d_W1 = [epsilon * W_i for W_i in d0_W1]
            d_W2 = [epsilon * W_i for W_i in d0_W2]
            d_b = [epsilon * W_i for W_i in d0_b]
            d_out_layer_W = epsilon * d0_out_layer_W
            d_out_layer_b = epsilon * d0_out_layer_b

            W1_plus_d = [self.W1s[i] + d_W1[i] for i in range(len(self.W1s))]
            W2_plus_d = [self.W2s[i] + d_W2[i] for i in range(len(self.W2s))]
            b_plus_d = [self.bs[i] + d_b[i] for i in range(len(self.bs))]
            out_layer_W_plus_d = self.output_layer_W + d_out_layer_W
            out_layer_b_plus_d = self.output_layer_b + d_out_layer_b

            F1 = self.feed_forward(x, y, W1_plus_d, W2_plus_d, b_plus_d,
                                   out_layer_W_plus_d, out_layer_b_plus_d, use_stored_weights=False)
            F2 = np.sum([np.dot((g0_W1[j]).flatten(), (d_W1[j]).flatten()) for j in range(len(d_W1))]) + \
                np.sum([np.dot((g0_W2[j]).flatten(), (d_W2[j]).flatten()) for j in range(len(d_W2))]) + \
                np.sum([np.dot((g0_b[j]).flatten(), (d_b[j]).flatten()) for j in range(len(d_b))]) + \
                np.dot(g0_out_layer_W.flatten(), d_out_layer_W.flatten()) + \
                np.dot(g0_out_layer_b.flatten(), d_out_layer_b.flatten())
            y1.append(np.abs(F1 - F0))
            y2.append(np.abs(F1 - F0 - F2))
        xs = np.arange(iterations)
        plt.plot(xs, y1, label="first order approximation")
        plt.plot(xs, y2, label="second order approxination")
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('approximation')
        plt.title('Full ResNet Gradient Test')
        plt.legend()
        plt.show()

    def fit(self, x_train, y_train, learning_rate=0.1, epochs=10, mini_batch_size=10):

        x_batches, y_batches = util.split_into_batches_T(
            x_train, y_train, mini_batch_size)

        for i in range(epochs):

            for x_batch, y_batch in zip(x_batches, y_batches):
                m = x_batch.shape[1]
                batch_grads_W1 = [np.zeros(W.shape) for W in self.W1s]
                batch_grads_W2 = [np.zeros(W.shape) for W in self.W2s]
                batch_grads_b = [np.zeros(b.shape) for b in self.bs]
                batch_grads_out_layer_W = np.zeros(self.output_layer_W.shape)
                batch_grads_out_layer_b = np.zeros(self.output_layer_b.shape)

                for i in range(x_batch.shape[1]):
                    x = x_batch[:, i]
                    y = y_batch[:, i]
                    loss = self.feed_forward(x, y)
                    grads_by_W1, grads_by_W2, grads_by_b, grad_Loss_by_W, grads_Loss_by_b = self.Grad_F_by_Theta(
                        y)
                    # sum batch gradients:
                    batch_grads_W1 = [grads_by_W1[i] + batch_grads_W1[i]
                                      for i in range(len(grads_by_W1))]
                    batch_grads_W2 = [grads_by_W2[i] + batch_grads_W2[i]
                                      for i in range(len(grads_by_W2))]
                    batch_grads_b = [grads_by_b[i] + batch_grads_b[i]
                                     for i in range(len(grads_by_b))]
                    batch_grads_out_layer_W = batch_grads_out_layer_W + grad_Loss_by_W
                    batch_grads_out_layer_b = batch_grads_out_layer_b + grads_by_b

                # taking mean of batch gradients:
                batch_grads_W1 = [W / m for W in self.W1s]
                batch_grads_W2 = [W / m for W in self.W2s]
                batch_grads_b = [b / m for b in self.bs]
                batch_grads_out_layer_W = batch_grads_out_layer_W / m
                batch_grads_out_layer_b = batch_grads_out_layer_b / m

                # updating weights:
                self.W1s = [self.W1s[i] - learning_rate * batch_grads_W1[i]
                            for i in range(len(self.W1s))]
                self.W2s = [self.W2s[i] - learning_rate * batch_grads_W2[i]
                            for i in range(len(self.W2s))]
                self.bs = [self.bs[i] - learning_rate * batch_grads_b[i]
                           for i in range(len(self.bs))]
                self.output_layer_W = self.output_layer_W - \
                    learning_rate * batch_grads_out_layer_W
                self.output_layer_b = self.output_layer_b - \
                    learning_rate * batch_grads_out_layer_b


if __name__ == "__main__":
    x_train, c_train, x_test, c_test = du.load_matlab_data_np_arrays(
        "datasets\\GMMData.mat")
    resNet = ff_residual_neural_network(5, [7, 4, 8], 5)
    resNet.gradient_test_resnet()
