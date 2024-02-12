import numpy as np
import matplotlib.pyplot as plt
import util


activation_function = np.tanh
output_layer_function = util.soft_max_loss
activation_function_derivative = util.tanh_derivative

def f_standart_layer(X, W,b):
    # ones_row = np.ones((1, X.shape[1]))
    # X_with_ones = np.vstack((X, ones_row))
    print(W @ X.flatten() + b.flatten())
    return activation_function(W @ X+ b)



def JacMV_f_by_W_with_b(x, W, v):  # only for x of shape: (d, 1)
    ones_row = np.ones((1, x.shape[1]))
    x_with_ones = np.vstack((x, ones_row))
    Z = activation_function_derivative(
        W @ x_with_ones.flatten())
    jac_by_b = np.diag(Z)
    jac_by_W = jac_by_b @ np.kron(x.T, np.eye(W.shape[0]))
    jac_by_theta = np.hstack((jac_by_W, jac_by_b))
    return jac_by_theta @ v

def JacMV_f_by_W(x, W, b, v):  # only for x of shape: (d, 1)
    # ones_row = np.ones((1, x.shape[1]))
    # x_with_ones = np.vstack((x, ones_row))
    # print("x with ones shape ",x_with_ones.shape)
    Z = activation_function_derivative(
        W @ x + b)
    print("Z shape ",Z.shape)
    jac_by_W = np.diag(Z.flatten()) @ np.kron(x.T, np.eye(W.shape[0]))
    print("jac shape ",jac_by_W.shape)
    return jac_by_W @ v

def JacMV_f_by_b(x, W, b, v):  # only for x of shape: (d, 1)
    # ones_row = np.ones((1, x.shape[1]))
    # x_with_ones = np.vstack((x, ones_row))
    # print("x with ones shape ",x_with_ones.shape)
    print("x shape ",x.shape)
    print("x flatten shape ",x.flatten().shape)
    print("W@x shape ",(W @ x.flatten()).shape)
    Z = activation_function_derivative(
        W @ x + b)
    print("Z shape ",Z.shape)
    jac_by_b = np.diag(Z.flatten())
    print("jac shape ",jac_by_b.shape)
    return jac_by_b @ v


def JacMv_test(F, JacMv, X_shape, W_shape, by='W', iterations=10):
    X_d, X_m = X_shape
    W_nlabels, W_d = W_shape
    X = np.random.rand(X_d, X_m)
    W = np.random.rand(W_nlabels, W_d)
    b = np.random.rand(W_nlabels, 1)
    if by == 'X':
        d0 = np.random.rand(X_d, X_m)
    elif by == 'W':
        d0 = np.random.rand(W_nlabels, W_d)
    else: # by == 'b'
        d0 = np.random.rand(W_nlabels, 1)
    eps0 = 0.5
    F0 = F(X, W, b)
    y1 = []
    y2 = []
    for i in range(iterations):
        epsilon = eps0 ** i
        d = epsilon * d0
        if by == 'X':
            F1 = F(X + d, W,b)
            F2 = JacMv(X, W,b, d)
        elif by == 'W':
            F1 = F(X, W + d,b)
            print("F1 shape ",F1.shape)
            # reorganizing d to match the jacboian multiplication:
            F2 = JacMv(X, W,b, d.flatten('F')).reshape(F1.shape)
            print("F2 shape ",F2.shape)
        else: # by == 'b'
            F1 = F(X, W ,b+d)
            F2 = JacMv(X, W,b, d)
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


def JacMV_f_by_x(x, W,b, v):  # only for x of shape: (d, 1)

    return np.diag(activation_function_derivative
                   (W@x + b).flatten()) @ (W @ v)

# JacMv_test(f_standart_layer, JacMV_f_by_W, (3, 1), (4, 3), by='W')
# JacMv_test(f_standart_layer, JacMV_f_by_W,(10, 1), (3, 11), by='W')
# print(JacMV_f_by_b(np.array([[1], [2], [3]]), np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), np.array([[2],[1]]), np.array([[1], [2]])))
# JacMv_test(f_standart_layer, JacMV_f_by_b, (3, 1), (4, 3), by='b')
JacMv_test(f_standart_layer, JacMV_f_by_W, (3, 1), (4, 3), by='W')
# print(f_standart_layer(np.array([[1], [2], [3]]), np.array([[1, 0, 0], [0, 1, 0],[0,0,1]]), np.array([[2],[1],[0]])))
