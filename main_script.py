import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
import data_utils as du
import util

def minimize_iris_LS():
    labeld_data = du.read_iris_dataset()
    print(util.SGD_minimizer(util.LS_grad, np.ones((4, 1)), labeld_data, len(
        labeld_data) // 10, plot=True, learning_rate=0.00072, tolerance=0.001))

