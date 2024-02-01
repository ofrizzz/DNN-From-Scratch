import scipy.io as sp
import pandas as pd
import numpy as np

def load_matlab_data_np_arrays(mat_file_path):
    mat_data = sp.loadmat(mat_file_path)
    data_object = {}
    for var_name, data in mat_data.items():
        if type(data) == np.ndarray:
            data_object[var_name] = data
    return data_object


def read_iris_dataset():
    iris_df = pd.read_csv("IRIS.csv")
    iris_df["species"].replace("Iris-virginica", 1, inplace=True)
    iris_df["species"].replace("Iris-versicolor", 2, inplace=True)
    iris_df["species"].replace("Iris-setosa", 3, inplace=True)
    labeld_data = iris_df.to_numpy()
    return labeld_data