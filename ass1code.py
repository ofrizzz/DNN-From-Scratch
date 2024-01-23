import numpy as np
import scipy.io as sp
import pandas as pd
def soft_max(x):
    exp_x = np.exp(x - np.max(x))
    softmax_values = exp_x / exp_x.sum(axis=0, keepdims=True)
    return softmax_values


# n_labels = [0, 1, 2]

def Loss_sample_v2(W, b, x, y):
    vec = [np.dot(W[:, i], x) + b[i] for i in n_labels]
    denominator = sum([np.exp(a - max(vec))
                       for a in vec])
    return vec[y] - max(vec) - np.log(denominator)


def Loss_sample(W, b, x, y):
    vec = [np.dot(W[:, i], x) + b[i] for i in n_labels]
    denominator = sum([np.exp(a - max(vec))
                       for a in vec])
    return np.log(np.exp(vec[y] - max(vec)) / denominator)


def Loss_sm_reg(W, b, mb_samples, mb_lables):
    return 0 - sum(Loss_sample_v2(W, b, x, y) for (x, y) in zip(mb_samples, mb_lables))


# list1 = np.array([1, 1, 1])
# list2 = np.eye(3)
# mat = np.eye(3)
# b = np.zeros((3, 1))

# W_example1 = np.array([[0.1, 0.2],
#                       [0.3, 0.4],
#                       [0.5, 0.6]])
# W_example1 = W_example1.transpose()
# b_example1 = np.array([0.01, 0.02, 0.03])
# mb_samples_example1 = np.array([[1.0, 2.0],
#                                 [2.0, 1.0],
#                                 [3.0, 2.5]])
# mb_labels_example1 = np.array([0, 1, 2])
# print(Loss_sm_reg(W_example1, b_example1, mb_samples_example1, mb_labels_example1)/3)

# print(f"expected output: {1.0704}")


def split_into_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def SGD_minimizer(grad_F, x0, learning_rate, labeled_data, mini_batch_size, tolerance = 1e-3):
    x = x0
    np.random.shuffle(labeled_data)
    mini_batches = split_into_batches(labeled_data, mini_batch_size)
    mini_batch_index = 0

    while True:
        current_data = mini_batches[mini_batch_index][:, :-1]
        current_labels = current_labels = mini_batches[mini_batch_index][:, -1].reshape(-1, 1)
        grad = grad_F(x, current_data , current_labels) 

        
        if isinstance(grad_norm, float):
            print(grad_norm)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            break
        x = x - learning_rate * grad
        mini_batch_index = (mini_batch_index + 1) % len(mini_batches)
    return x

def load_matlab_data_np_arrays(mat_file_path):
    mat_data = sp.loadmat(mat_file_path)
    data_object = {}
    for var_name, data in mat_data.items():
        if type(data) == np.ndarray:
            data_object[var_name] = data
    return data_object

def LS_grad(x, data, labels):
    return data.T @ (data @ x - labels)

mat_file_path = 'C:\\Users\\Daniel\\Desktop\\3rd_year\\DL\\assignment1\\DL-Course-Assignments-\\SwissRollData.mat'

# Load the MATLAB file
mat_data = sp.loadmat(mat_file_path)

# Print each variable as a NumPy array
# for var_name, data in mat_data.items():
#     if type(data) == np.ndarray:  # This ensures we only print NumPy arrays
#         print(f"{var_name}:")
#         print(data)
#         print()  # Adds a newline for better readability


iris_df = pd.read_csv("IRIS.csv")
iris_df["species"].replace("Iris-virginica", 1, inplace=True)
iris_df["species"].replace("Iris-versicolor", 2, inplace=True)
iris_df["species"].replace("Iris-setosa", 3, inplace=True)
# labels = iris_df["species"].to_numpy()
labeld_data = iris_df.to_numpy()
# print(split_into_batches(labeld_data,1))

print(SGD_minimizer(LS_grad, np.ones((4, 1)), 0.0001, labeld_data, len(labeld_data), 0.001))
# print("data:", data)
# print("labels", labels)
