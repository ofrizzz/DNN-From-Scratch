import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import data_utils as du
import util


def minimize_iris_LS():
    labeld_data = du.read_iris_dataset()
    data = labeld_data[:, :-1]
    labels = labeld_data[:, -1]
    print(util.SGD_minimizer(util.LS, util.LS_grad, np.ones((4, 1)), data, labels, len(
        labeld_data) // 10, plot=True, learning_rate=0.00032, tolerance=0.001, title="Least Squares SGD minimiztion"))


def minimize_softmax_regression():
    x_train, c_train, x_test, c_test = du.load_matlab_data_np_arrays(
        "datasets\\SwissRollData.mat")
    m = x_train.shape[0]
    l = c_train.shape[0]
    print(SGD_minimizer(util.soft_max_loss, util.soft_max_regression_grad_by_theta, np.random.rand(l, m), x_train,
          c_train, epochs=40, mini_batch_size=200,  plot=True, learning_rate=0.7, title="soft-max SGD minimiztion for W (Peaks dataset)"))


def compute_success_precent(W):
    x_train, c_train, x_test, c_test = du.load_matlab_data_np_arrays(
        "datasets\\SwissRollData.mat")
    train_subsample_size = len(x_train[0]) // 2
    test_subsample_size = len(x_test[0]) // 2
    train_indices = np.random.choice(
        len(x_train[0]), size=train_subsample_size, replace=False)
    test_indices = np.random.choice(
        len(x_test[0]), size=test_subsample_size, replace=False)
    sub_x_train = x_train[:, train_indices]
    sub_c_train = c_train[:, train_indices]
    sub_x_test = x_test[:, test_indices]
    sub_c_test = c_test[:, test_indices]
    Z_train = W @ sub_x_train
    Z_test = W @ sub_x_test
    logits_train = util.softmax(Z_train)
    logits_test = util.softmax(Z_test)
    pred_train = np.argmax(logits_train, axis=0)
    pred_test = np.argmax(logits_test, axis=0)
    correct_predictions_train = np.sum(
        pred_train == np.argmax(sub_c_train, axis=0))
    correct_predictions_test = np.sum(
        pred_test == np.argmax(sub_c_test, axis=0))
    succ_rate_train = correct_predictions_train / train_subsample_size
    succ_rate_test = correct_predictions_test / test_subsample_size
    return succ_rate_train, succ_rate_test


def SGD_minimizer(loss, loss_grad, x0, x_train, y_train, epochs=10,  mini_batch_size=10, plot=False, learning_rate=0.1, tolerance=1e-3, title="SGD minimzation"):
    W = x0
    # shuffle data and labels accordingly
    assert len(x_train[0]) == len(y_train[0])
    p = np.random.permutation(len(x_train[0]))
    x_train = x_train[:, p]
    y_train = y_train[:, p]

    x_batches, y_batches = util.split_into_batches_T(
        x_train, y_train, mini_batch_size)
    loss_log = []
    grad_norms_log = []
    train_success_rates = []
    test_success_rates = []
    for _ in range(epochs):

        for current_data, current_labels in zip(x_batches, y_batches):
            grad = loss_grad(current_data, W, current_labels.T)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < tolerance:
                break
            W = W - learning_rate * grad
        succ_rate_train, succ_rate_test = compute_success_precent(W)
        train_success_rates.append(succ_rate_train)
        test_success_rates.append(succ_rate_test)
        loss_log.append(loss(x_train, W, y_train.T))
        grad_norms_log.append(grad_norm)

    if plot:
        xs = np.arange(epochs)
        plt.figure()
        plt.yscale('log')
        plt.plot(xs, loss_log)
        plt.xlabel("epochs")
        plt.ylabel("loss function values")
        plt.title(title)
        plt.show()

        plt.figure()
        plt.yscale('log')
        plt.plot(xs, grad_norms_log)
        plt.xlabel("epochs")
        plt.ylabel("gradient norm values")
        plt.title(title)
        plt.show()

        plt.figure()
        plt.plot(xs, train_success_rates, label="train success rate")
        plt.plot(xs, test_success_rates, label="test success rate")
        plt.xlabel("epochs")
        plt.ylabel("success rate")
        plt.title("success rates for subsample of train/test sets")
        plt.legend()
        plt.show()

    return W


if __name__ == "__main__":
    minimize_softmax_regression()
    # train_success_rates = []
    # test_success_rates = []
    # for i in range(10):
    #     W = np.random.rand(5, 5)
    #     succ_train, succ_test = compute_success_precent(W)
    #     succ_rate_train, succ_rate_test = compute_success_precent(W)
    #     train_success_rates.append(succ_rate_train)
    #     test_success_rates.append(succ_rate_test)
    # xs = np.arange(10)
    # plt.figure()
    # plt.plot(xs, train_success_rates, label="train success rate")
    # plt.plot(xs, test_success_rates, label="test success rate")
    # plt.xlabel("epochs")
    # plt.ylabel("success rate")
    # plt.title("success rates for subsample of train/test sets")
    # plt.legend()
    # plt.show()
