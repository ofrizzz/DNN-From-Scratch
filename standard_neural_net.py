import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
import util
import data_utils as du


activation_function = np.tanh
activation_function_derivative = util.tanh_derivative
output_layer_function = util.soft_max_loss
loss_function_grad_theta = util.soft_max_regression_grad_by_theta
loss_function_grad_x = util.soft_max_regression_grad_by_x


class ff_standard_neural_network:
    def __init__(self, input_dimension, hidden_layers_dimensions, output_layer_dimension) -> None:
        self.input_dimension = input_dimension
        self.hidden_layers_dimensions = hidden_layers_dimensions
        self.output_layer_dimension = output_layer_dimension
        self.num_of_layers = len(hidden_layers_dimensions) + 2
        if len(hidden_layers_dimensions) > 0:
            self.weights = [np.random.rand(hidden_layers_dimensions[0], input_dimension + 1)] + \
                [np.random.rand(hidden_layers_dimensions[i], hidden_layers_dimensions[i-1] + 1) for i in range(1, len(hidden_layers_dimensions))] + \
                [np.random.rand(output_layer_dimension,
                                hidden_layers_dimensions[-1] + 1)]
        else:
            self.weights = [np.random.rand(
                output_layer_dimension, input_dimension + 1)]

        self.activations = []

    def feed_forward(self, X, C, Ws=None):
        if Ws == None:
            Ws = self.weights
        self.activations = [X]
        for layer in range(self.num_of_layers - 1):
            ones_row = np.ones((1, X.shape[1]))
            X_with_ones = np.vstack((X, ones_row))

            if layer < self.num_of_layers - 2:
                X = Ws[layer] @ X_with_ones
                X = activation_function(X)
            else:
                X = output_layer_function(X, Ws[-1], C)
            self.activations.append(X)
        return X

    def fit(self, x_train_, y_train_, x_test, c_test, learning_rate=0.1, epochs=10, mini_batch_size=10, plot=False):

        x_batches, y_batches = util.split_into_batches_T(
            x_train_, y_train_, mini_batch_size)
        train_succ_rates = []
        test_succ_rates = []
        for i in range(epochs):

            for x_batch, y_batch in zip(x_batches, y_batches):
                loss = self.feed_forward(x_batch, y_batch)
                grad = self.Grad_F_by_Theta(y_batch)

                self.weights = [self.weights[j] - learning_rate * grad[j]
                                for j in range(len(grad))]
            train_success_perc = self.compute_success_precent(
                x_train_, y_train_)
            test_success_perc = self.compute_success_precent(
                x_test, c_test)
            print(
                f"finished epoch {i} with train success percent: {train_success_perc * 100}%")
            train_succ_rates.append(train_success_perc * 100)
            test_succ_rates.append(test_success_perc * 100)

        if plot:
            xs = np.arange(epochs)
            plt.plot(xs, train_succ_rates, label='train')
            plt.plot(xs, test_succ_rates, label='test')
            plt.xlabel('epochs')
            plt.ylabel('success percentage')
            plt.title('GMM success percentages')
            plt.legend()
            plt.show()

    def Grad_F_by_Theta(self, C):
        grad = np.array(loss_function_grad_theta(
            self.activations[-2], self.weights[-1], C))
        grads_Ws = [grad]
        back_prop_grad = loss_function_grad_x(
            self.activations[-2], self.weights[-1], C)

        for X, weights in reversed(list(zip(self.activations[:-2], self.weights[:-1]))):
            grad = util.JacMV_f_by_theta_transpose(X, weights, back_prop_grad)
            grads_Ws.insert(0, grad)
            back_prop_grad = util.JacMV_f_by_x_transpose(
                X, weights, back_prop_grad)
        return grads_Ws

    def gradient_test_nn(self, m, iterations=20):
        X_d = self.input_dimension
        C_d = self.output_layer_dimension
        C_shape = (C_d, m)
        X = np.random.rand(X_d, m)
        C = np.zeros(C_shape)
        for i in range(C_shape[1]):
            C[np.random.randint(0, C_shape[0])][i] = 1
        d0 = [np.random.rand(W.shape[0], W.shape[1]) for W in self.weights]
        eps0 = 0.5
        F0 = self.feed_forward(X, C)
        g0 = self.Grad_F_by_Theta(C)
        for i, W in enumerate(self.weights):
            print(f"W in layer {i} is of shape: {W.shape}")
        for i, W in enumerate(g0):
            print(f"W grad in layer {i} is of shape: {W.shape}")
        y1 = []
        y2 = []
        for i in range(iterations):
            epsilon = eps0 ** i
            d = [epsilon * W_i for W_i in d0]
            W_plus_d = [self.weights[i] + d[i]
                        for i in range(len(self.weights))]
            F1 = self.feed_forward(X, C, W_plus_d)
            print(f"iteration {i} F1: {F1}")
            F2 = np.sum([np.dot((g0[j]).flatten(), (d[j]).flatten())
                         for j in range(len(d))])
            print(f"iteration {i} F2: {F2}")
            y1.append(np.abs(F1 - F0))
            y2.append(np.abs(F1 - F0 - F2))
        xs = np.arange(iterations)
        plt.plot(xs, y1, label="first order approximation")
        plt.plot(xs, y2, label="second order approxination")
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('approximation')
        plt.title('Full Neural Net Gradient Test')
        plt.legend()
        plt.show()

    def compute_success_precent(self, X_test, C_test):
        self.feed_forward(X_test, C_test)
        X = self.activations[-2]
        ones_row = np.ones((1, X.shape[1]))
        X_with_ones = np.vstack((X, ones_row))
        logits = util.stable_softmax(self.weights[-1] @ X_with_ones)
        pred = np.argmax(logits, axis=0)
        correct_predictions = np.sum(pred == np.argmax(C_test, axis=0))
        succ_rate = correct_predictions / C_test.shape[1]
        return succ_rate


if __name__ == "__main__":

    # network = ff_standard_neural_network(5, [20, 15], 3)
    x_train, c_train, x_test, c_test = du.load_matlab_data_np_arrays(
        "datasets\\GMMData.mat")
    rand_train_indices = np.random.choice(x_train.shape[1], 200)
    sub_x_train = x_train[:, rand_train_indices]
    sub_c_train = c_train[:, rand_train_indices]
    print("sub_x_train.shape ", sub_x_train.shape)
    print("sub_c_train ", sub_c_train.shape)
    network = ff_standard_neural_network(5, [10, 10], 5)
    network.fit(sub_x_train, sub_c_train, x_test, c_test, mini_batch_size=50,
                epochs=1000, learning_rate=0.05, plot=True)
    print("final test success percentages: ",
          network.compute_success_precent(x_test, c_test))
    # network.gradient_test_nn(30)
