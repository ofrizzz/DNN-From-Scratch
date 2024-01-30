import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt


def soft_max(x):
    exp_x = np.exp(x - np.max(x))
    softmax_values = exp_x / exp_x.sum(axis=0, keepdims=True)
    return softmax_values


# n_labels = [0, 1, 2]


def Loss_sample_v2(W, b, x, y):
    vec = [np.dot(W[:, i], x) + b[i] for i in n_labels]
    denominator = sum([np.exp(a - max(vec))
                       for a in vec])
    return vec[y] - max(vec) - np.log(denominator)


def Loss_sample(W, b, x, y):
    vec = [np.dot(W[:, i], x) + b[i] for i in n_labels]
    denominator = sum([np.exp(a - max(vec))
                       for a in vec])
    return np.log(np.exp(vec[y] - max(vec)) / denominator)


def Loss_sm_reg(W, b, mb_samples, mb_lables):
    return 0 - sum(Loss_sample_v2(W, b, x, y) for (x, y) in zip(mb_samples, mb_lables))


# list1 = np.array([1, 1, 1])
# list2 = np.eye(3)
# mat = np.eye(3)
# b = np.zeros((3, 1))

# W_example1 = np.array([[0.1, 0.2],
#                       [0.3, 0.4],
#                       [0.5, 0.6]])
# W_example1 = W_example1.transpose()
# b_example1 = np.array([0.01, 0.02, 0.03])
# mb_samples_example1 = np.array([[1.0, 2.0],
#                                 [2.0, 1.0],
#                                 [3.0, 2.5]])
# mb_labels_example1 = np.array([0, 1, 2])
# print(Loss_sm_reg(W_example1, b_example1, mb_samples_example1, mb_labels_example1)/3)

# print(f"expected output: {1.0704}")


def split_into_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def SGD_minimizer(grad_F, x0, labeled_data, mini_batch_size, plot=False, learning_rate=0.1, iterations=100, tolerance=1e-3):
    x = x0
    np.random.shuffle(labeled_data)
    mini_batches = split_into_batches(labeled_data, mini_batch_size)
    mini_batch_index = 0
    data = labeled_data[:, :-1]
    labels = labeled_data[:, -1].reshape(-1, 1)
    objectives = []
    for _ in range(iterations):
        current_data = mini_batches[mini_batch_index][:, :-1]
        current_labels = current_labels = mini_batches[mini_batch_index][:, -
                                                                         1].reshape(-1, 1)
        grad = grad_F(x, current_data, current_labels)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            break
        x = x - learning_rate * grad

        # Loss function:
        objectives.append(np.linalg.norm(data @ x - labels))

        mini_batch_index = (mini_batch_index + 1) % len(mini_batches)
    if plot:
        plt.figure()
        plt.semilogy()
        plt.plot([i for i in range(len(objectives))],
                 objectives)
        plt.show()
    return x


def load_matlab_data_np_arrays(mat_file_path):
    mat_data = sp.loadmat(mat_file_path)
    data_object = {}
    for var_name, data in mat_data.items():
        if type(data) == np.ndarray:
            data_object[var_name] = data
    return data_object


def LS_grad(x, data, labels):
    return data.T @ (data @ x - labels)


# mat_file_path = 'C:\\Users\\Daniel\\Desktop\\3rd_year\\DL\\assignment1\\DL-Course-Assignments-\\SwissRollData.mat'

# Load the MATLAB file
# mat_data = sp.loadmat(mat_file_path)

# Print each variable as a NumPy array
# for var_name, data in mat_data.items():
#     if type(data) == np.ndarray:  # This ensures we only print NumPy arrays
#         print(f"{var_name}:")
#         print(data)
#         print()  # Adds a newline for better readability

def read_iris_dataset():
    iris_df = pd.read_csv("IRIS.csv")
    iris_df["species"].replace("Iris-virginica", 1, inplace=True)
    iris_df["species"].replace("Iris-versicolor", 2, inplace=True)
    iris_df["species"].replace("Iris-setosa", 3, inplace=True)
    labeld_data = iris_df.to_numpy()
    return labeld_data


def minimize_iris_LS():
    labeld_data = read_iris_dataset()
    print(SGD_minimizer(LS_grad, np.ones((4, 1)), labeld_data, len(
        labeld_data) // 10, plot=True, learning_rate=0.00072, tolerance=0.001))

# TODO: Check dimensions for both cases of gradient (by W and by X)


def gradient_test(F, grad_F, X_shape, W_shape, C_shape):
    X = np.random.rand(X_shape)
    C = np.zeros(C_shape)
    for i in C_shape[0]:
        C[i][np.random.randint(0, C_shape[1])] = 1
    W = np.random.rand(W_shape)

    d = np.random.rand(n)
    epsilon = 0.5
    F0 = F(X, W, C)
    g0 = grad_F(X, W, C)
    y1 = []
    y2 = []
    eps = []
    for _ in range(10):
        epsilon = epsilon * 0.5
        eps.append(epsilon)
        F1 = F(x + epsilon * d)
        F2 = F0 + np.dot(d, g0)
        y1.append(np.abs(F0 - F1))
        y2.append(np.abs(F0 - F2))

    plt.plot(eps, F1, label="|F(x + e*d) - F(x)|")
    plt.plot(eps, F2, label="|F(x + e*d) - F(x) - e * d^Tgrad(x)|")
    plt.title('gradient test')
    plt.semilogy()
    plt.legend()
    plt.show()


class ff_standart_neural_network:

    def __init__(self, activation_function, Loss_function, layers_dimensions: list) -> None:
        self.activation_function = activation_function
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
        return X

    def tanh_derivative(x: float):
        return 1 - (np.tanh(x)) ** 2

    def soft_max_loss(X, W, C):
        m = X.shape[1]
        # C = np.zeros((m, W.shape[1]))
        # for i in range(len(y)):
        #     C[i][y[i]] = 1
        Z = np.exp(X.T @ W)
        summation = sum([np.exp(X.T @ W[:, i]) for i in range(len(W[0]))])
        U = np.log(Z / summation[:, None])
        return - 1 / m * np.sum((C * U), axis=1)

    # TODO: implement softmax grad

    def soft_max_grad_by_theta(X, W, C):
        m = X.shape[1]
        Z = np.exp(X.T @ W)
        summation = sum([np.exp(X.T @ W[:, i]) for i in range(len(W[0]))])
        return 1/m * X @ ((Z / summation[:, None]) - C)

    def soft_max_grad_by_x(X, W, C):
        m = X.shape[1]
        Z = np.exp(X.T @ W)
        summation = sum([np.exp(X.T @ W[:, i]) for i in range(len(W[0]))])
        return 1/m * W @ ((Z / summation[:, None]) - C)

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
            back_prop_grad = ff_standart_neural_network.Jac_f_by_x(
                self.weights, z, activation_function_derivative, back_prop_grad)
            grad_theta_i = ff_standart_neural_network.Jac_f_by_theta(
                self.weights[i+1].shape[1] - 1, self.weights[i+1].shape[0], X[i+1], z_next, activation_function_derivative, back_prop_grad)
            grad = np.concatenate(grad, grad_theta_i)

    def back_prop(self, x_train, y_train):  # TODO
        # for x in x_train:
        #     X = self.feed_forward(x)
        #     grad_F = self.Grad_F_by_Theta(X, soft_max_grad)
        #     SGD_minimizer(grad_F, self.weights)
        pass

# TODO design flow and then test everything


if __name__ == "__main__":
    gradient_test(soft_max_loss, soft_max_)
