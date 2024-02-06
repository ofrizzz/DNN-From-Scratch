import scipy.io as sp
import pandas as pd
import numpy as np


def load_matlab_data_np_arrays(mat_file_path):
    mat_data = sp.loadmat(mat_file_path)
    data_object = {}
    for var_name, data in mat_data.items():
        if type(data) == np.ndarray:
            data_object[var_name] = data
    x_train = data_object["Yt"]
    c_train = data_object["Ct"]

    x_test = data_object["Yv"]
    c_test = data_object["Cv"]
    return x_train, c_train, x_test, c_test


def read_iris_dataset():
    iris_df = pd.read_csv("datasets//IRIS.csv")
    iris_df["species"].replace("Iris-virginica", 0, inplace=True)
    iris_df["species"].replace("Iris-versicolor", 1, inplace=True)
    iris_df["species"].replace("Iris-setosa", 2, inplace=True)
    labeld_data = iris_df.to_numpy()
    return labeld_data


if __name__ == "__main__":

    print(x_train.shape)
    print(c_train.shape)
