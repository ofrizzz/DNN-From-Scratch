import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
import util

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
            [np.random.rand(output_layer_dimension,
                            hidden_layers_dimensions[-1] + 1)]

        # self.derivatives = [np.zeros((hidden_layers_dimensions[i], hidden_layers_dimensions[i-1] + 1)) for i in range(self.num_of_layers-1)]
        self.activations = []

    def feed_forward(self, X, C, Ws=None):
        if not Ws:
            Ws = self.weights
        self.activations = [X]
        # x is a batch of inputs
        # concatenate 1 at the bottom of x for bias
        for layer in range(self.num_of_layers - 1):
            ones_row = np.ones((1, X.shape[1]))
            X = np.vstack((X, ones_row))
            # X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
            if layer < self.num_of_layers - 2:
                X = Ws[layer] @ X
                X = activation_function(X)
            else:
                X = output_layer_function(X, Ws[-1], C)
            self.activations.append(X)
        # activations[i] is the output of layer i
        return X

    def fit(self, x_train, y_train, epochs=10, mini_batch_size=10):

        x_batches, y_batches = util.split_into_batches(
            x_train, y_train, mini_batch_size)

        for i in range(epochs):

            for x_batch, y_batch in zip(x_batches, y_batches):
                outputs = self.feed_forward(x_batch)  # add C to feed_forward
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

    def Grad_F_by_Theta(self, C):
        grad = np.array([loss_function_grad_theta(
            self.activations[-2], self.weights[-1], C)])
        grads_Ws = [grad]
        back_prop_grad = loss_function_grad_x(
            self.activations[-2], self.weights[-1], C)

        for X, weights in reversed(list(zip(self.activations[:-2], self.weights[:-1]))):
            grad = util.JacMV_f_by_theta_transpose(X, weights, back_prop_grad)
            grads_Ws.insert(0, grad)
            back_prop_grad = util.JacMV_f_by_x_transpose(
                X, weights, back_prop_grad)

        return grads_Ws

    def gradient_test_nn(self, C_shape):
        X_d = self.input_dimension
        X = np.random.rand(X_d)
        C = np.zeros(C_shape)
        for i in range(C_shape[0]):
            C[i][np.random.randint(0, C_shape[1])] = 1
        d0 = [np.random.rand(W.shape[0], W.shape[0]) for W in self.weights]
        eps0 = 0.5
        F0 = self.feed_forward(X, C)
        g0 = self.Grad_F_by_Theta(C)
        y1 = []
        y2 = []
        for i in range(10):
            epsilon = eps0 ** i
            d = [epsilon * W_i for W_i in d0]
            W_plus_d = [self.weights[i] + d[i]
                        for i in range(len(self.weights))]
            F1 = self.feed_forward(X, Ws=W_plus_d, C=C)
            F2 = np.dot(d.flatten(), g0.flatten())
            y1.append(np.abs(F1 - F0))
            y2.append(np.abs(F1 - F0 - F2))
        print(y1)
        print(y2)
        xs = np.arange(0, 10)
        plt.plot(xs, y1, label="first order approximation")
        plt.plot(xs, y2, label="second order approxination")
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('approximation')
        plt.title('gradient test by: ')
        plt.legend()
        plt.show()

    # Sample test input to initialize ff_standard_neural_network class


if __name__ == "__main__":
    input_dimension = 5  # For example, a neural network with 5 input features
    # Two hidden layers, first with 4 neurons and second with 3 neurons
    hidden_layers_dimensions = [4, 3]
    # Assuming a binary classification task, so 2 output neurons
    output_layer_dimension = 2

    # Initialize the ff_standard_neural_network instance
    network = ff_standard_neural_network(
        input_dimension, hidden_layers_dimensions, output_layer_dimension)
    # print([network.weights[i].shape for i in range(len(network.weights))])

    network = ff_standard_neural_network(3, [2], 3)

    # Test input: batch of 2 vectors, each with 3 features
    input_batch = np.array([[0.5, -0.5],  [0.3, 0.5], [-0.5,  0.3]])
    print(input_batch.shape)
    C = np.zeros((3, 2))
    for i in range(C.shape[0]):
        C[i][np.random.randint(0, C.shape[1])] = 1
    # Execute feed_forward
    print("loss for x: ", network.feed_forward(input_batch, C))
    activations = network.activations
    # Display the output
    for i, activation in enumerate(activations):
        print(f"Layer {i+1} activations:\n{activation}\n")

    print(network.Grad_F_by_Theta(C))
