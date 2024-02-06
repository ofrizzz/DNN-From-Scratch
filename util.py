import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z > 0, 1, 0)


def tanh_derivative(z):
    return 1 - np.tanh(z)**2


def split_into_batches(x_train, y_train, batch_size):
    x_batches = [x_train[i:i + batch_size]
                 for i in range(0, len(x_train), batch_size)]
    y_batches = [y_train[i:i + batch_size]
                 for i in range(0, len(y_train), batch_size)]
    return x_batches, y_batches


def softmax(logits):
    max_logit = np.max(logits)
    shifted_logits = logits - max_logit

    # Compute softmax probabilities
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return probabilities


def LS_grad(x, data, labels):
    return data.T @ (data @ x - labels)


def stable_softmax(Z):
    exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def soft_max_loss(X, W, C):
    logits = X.T @ W.T
    softmax_probs = stable_softmax(logits)
    log_probs = np.log(softmax_probs)
    return -np.mean(np.sum(C * log_probs, axis=1))


# Softmax Gradient with respect to W (Theta)
def soft_max_regression_grad_by_theta(X, W, C):
    m = X.shape[1]
    Z = X.T @ W.T  # Compute logits
    softmax = stable_softmax(Z)
    dL_dZ = (softmax - C) 
    grad= (X @ dL_dZ).T
    return grad / m

# Softmax Gradient with respect to X


def soft_max_regression_grad_by_x(X, W, C):
    m = X.shape[1]
    Z = X.T @ W.T  # Compute logits
    softmax = stable_softmax(Z)
    dL_dZ = (softmax - C) / m
    dL_dX = W.T @ dL_dZ.T
    return dL_dX

activation_function = np.tanh
output_layer_function = soft_max_loss
activation_function_derivative = tanh_derivative

def f_standart_layer(X, W):
    X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
    print(f"X.shape: {X.shape}")
    print(f"W.shape: {W.shape}")
    return activation_function(W @ X)

def Jac_f_by_x(X, W, V):
    X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
    Z = activation_function_derivative(W @ X)
    A = (Z * V)
    return (W.T @ A).T

def Jac_f_by_theta(X, W, V):
    X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)


    Z = activation_function_derivative(W @ X)
    A = (Z * V)
    return (A @ X.T).T


def gradient_test(F, grad_F, X_shape, W_shape, C_shape, by='X'):
    X_d, X_m = X_shape
    W_d, W_nlabels = W_shape
    X = np.random.rand(X_d, X_m)
    C = np.zeros(C_shape)
    for i in range(C_shape[0]):
        C[i][np.random.randint(0, C_shape[1])] = 1
    W = np.random.rand(W_d, W_nlabels)
    if by == 'X':
        d0 = np.random.rand(X_d, X_m)
    elif by == 'W':
        d0 = np.random.rand(W_d, W_nlabels)
    eps0 = 0.5
    F0 = F(X, W, C)
    g0 = grad_F(X, W, C)
    y1 = []
    y2 = []
    for i in range(10):
        epsilon = eps0 ** i
        d = epsilon * d0
        if by == 'X':
            F1 = F(X + d, W, C)
        elif by == 'W':
            F1 = F(X, W + d, C)
        F2 = np.dot(d.flatten(), g0.flatten())
        # F1 - F0
        # F1 - F0 - eps * d.T Grad(x)
        y1.append(np.abs(F1 - F0))
        y2.append(np.abs(F1 - F0 - F2))
    print(y1)
    print(y2)
    xs = np.arange(0, 10)
    plt.plot(xs, y1, label="first order approximation")
    plt.plot(xs, y2, label="second order approxination")
    plt.yscale('log')
    plt.title('gradient test by: ' + by)
    plt.legend()
    plt.show()

def JacMv_test(F, JacMv, X_shape, W_shape, by='X'):
    X_d, X_m = X_shape
    W_d, W_nlabels = W_shape
    X = np.random.rand(X_d, X_m)
    W = np.random.rand(W_d, W_nlabels)
    if by == 'X':
        d0 = np.random.rand(X_d, X_m)
    elif by == 'W':
        d0 = np.random.rand(W_d, X_m)
    eps0 = 0.5
    F0 = F(X, W)
    g0 = JacMv(X, W, d0)
    y1 = []
    y2 = []
    for i in range(10):
        epsilon = eps0 ** i
        d = epsilon * d0
        if by == 'X':
            F1 = F(X + d, W)
        elif by == 'W':
            F1 = F(X, W + d)
        F2 = np.dot(d.flatten(), g0.flatten())
        # F1 - F0
        # F1 - F0 - eps * d.T Grad(x)
        y1.append(np.abs(F1 - F0))
        y2.append(np.abs(F1 - F0 - F2))
    print(y1)
    print(y2)
    xs = np.arange(0, 10)
    plt.plot(xs, y1, label="first order approximation")
    plt.plot(xs, y2, label="second order approxination")
    plt.yscale('log')
    plt.title('gradient test by: ' + by)
    plt.legend()
    plt.show()



def jac_test(F, Jac_Mv):
    pass


def SGD_minimizer(grad_F, x0, labeled_data, mini_batch_size, plot=False, learning_rate=0.1, iterations=100, tolerance=1e-3):
    x = x0
    np.random.shuffle(labeled_data)

    mini_batches = split_into_batches(labeled_data[:, :-1], labeled_data[:, -1], mini_batch_size)
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


def test_soft_max_loss():
    # Test data
    X = np.array([[1, 2], [1, 4], [1, 6]])  # Add a bias term if necessary
    W = np.array([[0.5, -0.5], [-0.5, 0.5]])
    C = np.array([[1, 0], [0, 1], [1, 0]])  # Assuming two classes

    # Expected loss calculation
    # This is a simplified example. In practice, you would compute this
    # based on the softmax formula and the specific values of X, W, and C.
    expected_loss = 0.5  # This value should be calculated based on your test data

    # Calculate loss using your function

    calculated_loss = soft_max_loss(X, W, C)
    print(calculated_loss)

    # Assert that the calculated loss is close to the expected loss
    np.testing.assert_almost_equal(calculated_loss, expected_loss, decimal=5)

# Run the test


if __name__ == "__main__":
    # test_soft_max_loss()
    # gradient_test(soft_max_loss, soft_max_regression_grad_by_x,
    #             (30, 100), (10, 30), (100, 10), by='X')
    # gradient_test(soft_max_loss, soft_max_regression_grad_by_theta,
    #               (30, 100), (10, 30), (100, 10), by='W')
    JacMv_test(f_standart_layer, Jac_f_by_theta,
                (30, 100), (10, 31), by='W')
