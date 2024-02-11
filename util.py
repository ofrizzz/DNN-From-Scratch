import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt

MACHINE_EPS_FL32 = np.finfo(np.float32).eps


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


def split_into_batches_T(x_train, y_train, batch_size):
    x_batches = [x_train[:, i:i + batch_size]
                 for i in range(0, len(x_train), batch_size)]
    y_batches = [y_train[:, i:i + batch_size]
                 for i in range(0, len(y_train), batch_size)]
    return x_batches, y_batches


def softmax(logits):
    max_logit = np.max(logits)
    shifted_logits = logits - max_logit

    # Compute softmax probabilities
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return probabilities


def LS_grad(data, x, labels):
    return data.T @ (data @ x - labels)


def LS(data, x, labels):
    return np.linalg.norm(data @ x - labels)


def stable_softmax(Z):
    exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def soft_max_loss(X, W, C):
    ones_row = np.ones((1, X.shape[1]))
    X_with_ones = np.vstack((X, ones_row))
    logits = (W @ X_with_ones).T
    softmax_probs = stable_softmax(logits)
    log_probs = np.log(softmax_probs)
    return -np.mean(np.sum(C.T * log_probs, axis=1))


activation_function = np.tanh
output_layer_function = soft_max_loss
activation_function_derivative = tanh_derivative

# def soft_max_loss(X, W, C):
#     logits = W @ X
#     softmax_probs = stable_softmax(logits)
#     log_probs = np.log(softmax_probs)
#     return -np.mean(np.sum(C * log_probs, axis=1))


# Softmax Gradient with respect to W (Theta)
def soft_max_regression_grad_by_theta(X, W, C):
    m = X.shape[1]
    ones_row = np.ones((1, X.shape[1]))
    X_with_ones = np.vstack((X, ones_row))
    Z = X_with_ones.T @ W.T
    softmax = stable_softmax(Z)
    dL_dZ = (softmax - C.T)
    grad = (X @ dL_dZ).T
    grad = grad / m
    grad_by_b = np.sum(dL_dZ.T, axis=1) / m
    grad_by_theta = np.hstack((grad, grad_by_b.reshape((-1, 1))))
    return grad_by_theta


# def soft_max_regression_grad_by_theta(X, W, C):
#     m = X.shape[1]
#     Z = W @ X  # Compute logits
#     softmax = stable_softmax(Z)
#     dL_dZ = (softmax - C)
#     grad = X @ dL_dZ.T
#     return grad / m

# Softmax Gradient with respect to X


def soft_max_regression_grad_by_x(X, W, C):
    m = X.shape[1]
    ones_row = np.ones((1, X.shape[1]))
    X_with_ones = np.vstack((X, ones_row))
    Z = X_with_ones.T @ W.T
    softmax = stable_softmax(Z)
    dL_dZ = (softmax - C.T) / m
    dL_dX = (W[:, :-1]).T @ dL_dZ.T
    return dL_dX


def f_standart_layer(X, W):
    ones_row = np.ones((1, X.shape[1]))
    X_with_ones = np.vstack((X, ones_row))
    return activation_function(W @ X_with_ones)


def JacMV_f_by_x_transpose(X, W, V):
    ones_row = np.ones((1, X.shape[1]))
    X_with_ones = np.vstack((X, ones_row))
    Z = activation_function_derivative(W @ X_with_ones)
    A = (Z * V)
    return (W[:, :-1]).T @ A


def JacMV_f_by_theta_transpose(X, W, v):
    ones_row = np.ones((1, X.shape[1]))
    X_with_ones = np.vstack((X, ones_row))
    A = (activation_function_derivative(W @ X_with_ones) * v)
    jacmv_by_W = A @ X.T
    jacmv_by_b = np.sum(A, axis=1)
    return np.hstack((jacmv_by_W, jacmv_by_b.reshape(-1, 1)))


def JacMV_f_by_x(x, W, v):  # only for x of shape: (d, 1)
    ones_row = np.ones((1, x.shape[1]))
    x_with_ones = np.vstack((x, ones_row))
    return np.diag(activation_function_derivative(W@x_with_ones.flatten())) @ (W[:, :-1] @ v)


# def JacMV_f_by_W(x, W, v):
#     ones_row = np.ones((1, x.shape[1]))
#     x_with_ones = np.vstack((x, ones_row))
#     return np.diag(activation_function_derivative(W @ x_with_ones.flatten())) @ (np.kron(x.T, np.eye(W.shape[0])) @ v)


def JacMV_f_by_W(x, W, v):  # only for x of shape: (d, 1)
    ones_row = np.ones((1, x.shape[1]))
    x_with_ones = np.vstack((x, ones_row))
    Z = activation_function_derivative(
        W @ x_with_ones.flatten())
    jac_by_b = np.diag(Z)
    jac_by_W = jac_by_b @ np.kron(x.T, np.eye(W.shape[0]))
    jac_by_theta = np.hstack((jac_by_W, jac_by_b))
    return jac_by_theta @ v


# def JacMV_f_by_theta_transpose(X, W, V):
#     # X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)
#     Z = activation_function_derivative(W @ X)
#     A = (Z * V)
#     return np.outer(A, X)


def gradient_test(F, grad_F, X_shape, W_shape, C_shape, by='X', iter=20):
    X_d, X_m = X_shape
    W_d, W_nlabels = W_shape
    X = np.random.rand(X_d, X_m)
    C = np.zeros(C_shape)
    for i in range(C_shape[1]):
        C[np.random.randint(0, C_shape[0])][i] = 1
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
    for i in range(iter):
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
    xs = np.arange(0, iter)
    plt.plot(xs, y1, label="first order approximation")
    plt.plot(xs, y2, label="second order approxination")
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('approximation')
    plt.title('gradient test by: ' + by)
    plt.legend()
    plt.show()


def JacMv_test(F, JacMv, X_shape, W_shape, by='W', iterations=10):
    X_d, X_m = X_shape
    W_nlabels, W_d = W_shape
    X = np.random.rand(X_d, X_m)
    W = np.random.rand(W_nlabels, W_d)
    if by == 'X':
        d0 = np.random.rand(X_d, X_m)
    elif by == 'W':
        d0 = np.random.rand(W_nlabels, W_d)
    eps0 = 0.5
    F0 = F(X, W)
    y1 = []
    y2 = []
    for i in range(iterations):
        epsilon = eps0 ** i
        d = epsilon * d0
        if by == 'X':
            F1 = F(X + d, W)
            F2 = JacMv(X, W, d)
        elif by == 'W':
            F1 = F(X, W + d)
            d_w = d[:, :-1]
            d_b = d[:, -1]
            # reorganizing d to match the jacboian multiplication:
            d = np.append(d_w.flatten(order='F'), d_b)
            F2 = JacMv(X, W, d)
        # F1 - F0
        # F1 - F0 - eps * d.T Grad(x)
        y1.append(np.linalg.norm(F1 - F0))
        y2.append(np.linalg.norm(F1 - F0 - F2))
    xs = np.arange(iterations)
    plt.plot(xs, y1, label="first order approximation")
    plt.plot(xs, y2, label="second order approxination")
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('approximation')
    plt.title('JacMv test by: ' + by)
    plt.legend()
    plt.show()




def JacTMV_by_x_test(x_shape, W_shape, u_d, v_d):
    x = np.random.rand(x_shape[0], x_shape[1])
    W = np.random.rand(W_shape[0], W_shape[1])
    u = np.random.rand(u_d, 1)
    v = np.random.rand(v_d)
    return abs(u.T @ JacMV_f_by_x(x, W, v) - v.T @ JacMV_f_by_x_transpose(x, W, u)) < MACHINE_EPS_FL32


def JacTMV_by_W_test(x_shape, W_shape, u_d, v_d):
    x = np.random.rand(x_shape[0], x_shape[1])
    W = np.random.rand(W_shape[0], W_shape[1])
    u = np.random.rand(u_d, 1)
    v = np.random.rand(v_d)
    return abs(u.T @ JacMV_f_by_W(x, W, v) - v.T @ (JacMV_f_by_theta_transpose(x, W, u))) < MACHINE_EPS_FL32


def SGD_minimizer(loss, loss_grad, x0, x_train, y_train, mini_batch_size=10, plot=False, learning_rate=0.1, iterations=100, tolerance=1e-3, title="SGD minimzation"):
    x = x0
    # shuffle data and labels accordingly
    assert len(x_train) == len(y_train)
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]

    x_batches, y_batches = split_into_batches(
        x_train, y_train, mini_batch_size)
    mini_batch_index = 0
    loss_log = []
    grad_norms_log = []
    for _ in range(iterations):
        current_data = x_batches[mini_batch_index]
        current_labels = y_batches[mini_batch_index].reshape(-1, 1)
        grad = loss_grad(current_data, x, current_labels)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            break
        x = x - learning_rate * grad

        loss_log.append(loss(x_train, x, y_train))
        grad_norms_log.append(grad_norm)
        mini_batch_index = (mini_batch_index + 1) % len(x_batches)
    if plot:
        xs = np.arange(iterations)
        plt.figure()
        plt.yscale('log')
        plt.plot(xs, loss_log)
        plt.xlabel("iterations")
        plt.ylabel("loss function values")
        plt.title(title)
        plt.show()

        plt.figure()
        plt.yscale('log')
        plt.plot(xs, grad_norms_log)
        plt.xlabel("iterations")
        plt.ylabel("gradient norm values")
        plt.title(title)
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
    #               (30, 100), (10, 31), (10, 100), by='X')
    # gradient_test(soft_max_loss, soft_max_regression_grad_by_theta,
    #               (30, 100), (10, 31), (10, 100), by='W')
    # JacMv_test(f_standart_layer, JacMV_f_by_x,
    #            (10, 1), (5, 11), by='X')
    JacMv_test(f_standart_layer, JacMV_f_by_W,
               (10, 1), (3, 11), by='W')
    # print(JacTMV_by_x_test((30, 1), (10, 31), 10, 30))
    # print(JacTMV_by_W_test((30, 1), (10, 31), 10, 310))
