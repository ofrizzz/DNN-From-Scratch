import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt

def split_into_batches(x_train, y_train, batch_size):
    x_batches = [x_train[i:i + batch_size] for i in range(0, len(x_train), batch_size)]
    y_batches = [y_train[i:i + batch_size] for i in range(0, len(y_train), batch_size)]
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

def tanh_derivative(x: float):
    return 1 - (np.tanh(x)) ** 2

def stable_softmax(Z):
    exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

def soft_max_loss(X, W, C):
    logits = X.T @ W
    softmax_probs = stable_softmax(logits)
    log_probs = np.log(softmax_probs)
    return -np.mean(np.sum(C * log_probs, axis=1))
    

# Softmax Gradient with respect to W (Theta)
def soft_max_grad_by_theta(X, W, C):
    m = X.shape[1]
    Z = X.T @ W  # Compute logits
    softmax = stable_softmax(Z)
    dL_dZ = (softmax - C) / m
    grad = X @ dL_dZ
    return grad

# Softmax Gradient with respect to X
def soft_max_grad_by_x(X, W, C):
    m = X.shape[1]
    Z = X.T @ W  # Compute logits
    softmax = stable_softmax(Z)
    dL_dZ = (softmax - C) / m
    dL_dX = dL_dZ @ W.T
    return dL_dX.T

def gradient_test(F, grad_F, X_shape, W_shape, C_shape):
    X_d, X_m = X_shape
    W_d, W_nlabels = W_shape
    # C_m, C_nlabels = C_shape
    X = np.random.rand(X_d, X_m)
    C = np.zeros(C_shape)
    for i in range(C_shape[0]):
        C[i][np.random.randint(0, C_shape[1])] = 1
    W = np.random.rand(W_d, W_nlabels)
    d = np.random.rand(X_d,X_m)
    epsilon = 0.5
    F0 = F(X, W, C)
    g0 = grad_F(X, W, C)
    y1 = []
    y2 = []
    eps = []
    for _ in range(10):
        epsilon = epsilon * 0.5
        eps.append(epsilon)
        F1 = F(X + epsilon * d, W, C)
        F2 = F0 + np.dot(d, g0)
        y1.append(np.abs(F0 - F1))
        y2.append(np.abs(F0 - F2))
    plt.plot(eps, F1, label="|F(x + e*d) - F(x)|")
    plt.plot(eps, F2, label="|F(x + e*d) - F(x) - e * d^Tgrad(x)|")
    plt.title('gradient test')
    plt.semilogy()
    plt.legend()
    plt.show()

def jac_test(F, Jac_Mv):
    pass

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

if __name__ == "__main__":
    gradient_test(soft_max_loss, soft_max_grad_by_x, (30, 100), (30, 10), (100, 10))