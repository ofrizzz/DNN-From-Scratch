import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def stable_softmax(Z):
    exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

def cross_entropy_loss(F, C):
    log_F = np.log(F)
    return -np.mean(np.sum(C * log_F, axis=1))

def cross_entropy_grad(F, C):
    m = F.shape[1]
    dL_dF = -C / F
    return dL_dF / m


def SGD_minimizer(grad_F, x0, labeled_data, mini_batch_size, plot=False, learning_rate=0.1, iterations=100, tolerance=1e-3):
    x = x0
    np.random.shuffle(labeled_data)
    mini_batches = util.split_into_batches(labeled_data, mini_batch_size)
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



