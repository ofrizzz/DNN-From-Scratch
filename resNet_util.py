import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import util


activation_function = np.tanh
output_layer_function = util.soft_max_loss
activation_function_derivative = util.tanh_derivative


def soft_max_loss(x, W, b, y):
    logits = (W @ x + b).T
    softmax_probs = util.stable_softmax(logits)
    log_probs = np.log(softmax_probs)
    return np.sum(y.T * log_probs)

def soft_max_regression_grad_by_b(x, W, b, y):
    Z = (W @ x + b).T
    softmax = util.stable_softmax(Z)
    dL_dZ = (softmax - y.T)
    grad = np.sum(dL_dZ, axis=1).reshape(-1, 1)
    return grad


def soft_max_regression_grad_by_W(x, W, b, y):
    Z = (W @ x + b).T
    softmax = util.stable_softmax(Z)
    dL_dZ = (softmax - y.T)
    grad = (x @ dL_dZ).T
    return grad




def soft_max_regression_grad_by_x(x, W, b, y):
    Z = (W @ x + b).T
    softmax = util.stable_softmax(Z)
    dL_dZ = (softmax - y.T)
    dL_dX = W.T @ dL_dZ.T
    return dL_dX


def f_resnet_layer(x, W1, W2, b):
    z1 = W1 @ x + b
    a1 = activation_function(z1)
    z2 = x + W2 @ a1
    return z2


def Jac_f_by_b(x, W1, W2, b):
    z1 = W1 @ x + b
    return W2 @ np.diag(activation_function_derivative(z1).flatten())


def Jac_f_by_W1(x, W1, W2, b):
    z1 = W1 @ x + b
    return W2 @ np.diag(activation_function_derivative(z1).flatten()) @ np.kron(x.T, np.eye(W1.shape[0]))


def Jac_f_by_W2(x, W1, W2, b):
    z1 = W1 @ x + b
    return np.kron(activation_function(z1).T, np.eye(W2.shape[0]))


def Jac_f_by_x(x, W1, W2, b):
    return np.eye(x.shape[0]) + W2 @ np.diag(activation_function_derivative(W1 @ x + b).flatten()) @ W1

# W1 = n2xn1, W2= n1xn2
def gradient_test(F, grad_F, W_shape, by='x', iter=20):
    W_nlabels, W_d = W_shape
    x = np.random.rand(W_d, 1)

    y = np.zeros((W_nlabels, 1))
    y[np.random.randint(0, W_nlabels)][0] = 1

    W = np.random.rand(W_nlabels, W_d)
    b = np.random.rand(W_nlabels, 1)

    if by == 'x':
        d0 = np.random.rand(W_d, 1)
    elif by == 'W':
        d0 = np.random.rand(W_nlabels, W_d)
    elif by == 'b':
        d0 = np.random.rand(W_nlabels, 1)
    
    eps0 = 0.5
    F0 = F(x, W, b, y)
    g0 = grad_F(x, W, b, y)
    y1 = []
    y2 = []
    for i in range(iter):
        epsilon = eps0 ** i
        d = epsilon * d0
        if by == 'x':
            F1 = F(x + d, W, b, y)
        elif by == 'W':
            F1 = F(x, W + d, b, y)
        elif by == 'b':
            F1 = F(x, W, b + d, y)

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

def JacMv_test(F, Jac, W2_shape, by='W1', iterations=10):

    n1, n2 = W2_shape
    x_shape = n1
    b_shape = n2
    x = np.random.rand(x_shape, 1)
    b = np.random.rand(b_shape, 1)
    W1 = np.random.rand(n2, n1)
    W2 = np.random.rand(n1, n2)
    if by == 'x':
        d0 = np.random.rand(x_shape, 1)
    elif by == 'b':
        d0 = np.random.rand(b_shape, 1)
    elif by == 'W1':
        d0 = np.random.rand(n2, n1)
    elif by == 'W2':
        d0 = np.random.rand(n1, n2)
    eps0 = 0.5
    F0 = F(x, W1, W2, b)
    y1 = []
    y2 = []
    for i in range(iterations):
        epsilon = eps0 ** i
        d = epsilon * d0
        if by == 'x':
            F1 = F(x + d, W1, W2, b)

        elif by == 'b':
            F1 = F(x, W1, W2, b+d)
        elif by == 'W1':
            F1 = F(x, W1 + d, W2, b)
        elif by == 'W2':
            # TODO => test by W2 doesn't work
            F1 = F(x, W1, W2 + d, b)
        F2 = Jac(x, W1, W2, b) @ d.flatten('F').reshape(-1, 1)

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

if __name__ == "__main__":
    # JacMv_test(f_resnet_layer, Jac_f_by_x,
    #            (5, 11), by='x')
    # JacMv_test(f_resnet_layer, Jac_f_by_b,
    #            (5, 11), by='b')
    # JacMv_test(f_resnet_layer, Jac_f_by_W1,
    #            (5, 11), by='W1')


    JacMv_test(f_resnet_layer, Jac_f_by_W2,
            (5, 11), by='W2')

    gradient_test(soft_max_loss, soft_max_regression_grad_by_x, (5, 11), by='x')
    gradient_test(soft_max_loss, soft_max_regression_grad_by_W, (5, 11), by='W')
    gradient_test(soft_max_loss, soft_max_regression_grad_by_b, (5, 11), by='b')

