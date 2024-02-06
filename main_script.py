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
    labeld_data = du.read_iris_dataset()
    data = labeld_data[:, :-1]
    labels = labeld_data[:, -1]
    labels = [int(l) for l in labels]
    classes = 3
    C = np.zeros((len(labels), classes))
    C[np.arange(len(labels)), labels] = 1
    print(util.SGD_minimizer(util.soft_max_loss, util.soft_max_regression_grad_by_theta, np.ones((3, 4)), data, C, len(
        labeld_data) // 10, plot=True, learning_rate=0.00032, tolerance=0.001, title="soft-max SGD minimiztion for W (Iris dataset)"))


if __name__ == "__main__":
    minimize_iris_LS()
