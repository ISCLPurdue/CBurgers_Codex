print("From python: Within python module")

import os
import sys

HERE = os.getcwd()
sys.path.insert(0, HERE)

import numpy as np
import matplotlib.pyplot as plt


data_array = np.zeros(shape=(2001, 258))
x = np.arange(start=0, stop=2.0 * np.pi, step=2.0 * np.pi / 256)
iternum = 0


def collection_func(input_array):
    global data_array, iternum
    data_array[iternum, :] = input_array[:]
    iternum += 1
    return None


def analyses_func():
    global data_array, x

    plt.figure()
    for i in range(0, np.shape(data_array)[0], 400):
        plt.plot(x, data_array[i, 1:-1], label="Timestep " + str(i))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Field evolution")
    plt.savefig("Field_evolution.png")
    plt.close()

    data_array = data_array[:, 1:-1]
    print("Performing SVD")
    _, _, v = np.linalg.svd(data_array, full_matrices=False)

    plt.figure()
    plt.plot(x, v[0, :], label="Mode 0")
    plt.plot(x, v[1, :], label="Mode 1")
    plt.plot(x, v[2, :], label="Mode 2")
    plt.legend()
    plt.title("SVD Eigenvectors")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.savefig("SVD_Eigenvectors.png")
    plt.close()

    np.save("eigenvectors.npy", v[0:3, :].T)

    # Coefficient time series in retained POD basis.
    time_series = np.matmul(v[0:3, :], data_array.T).T
    time_series = np.nan_to_num(time_series, nan=0.0, posinf=1e6, neginf=-1e6)
    time_series = np.clip(time_series, -1e6, 1e6)

    num_timesteps = np.shape(time_series)[0]
    train_series = time_series[: num_timesteps // 2]
    test_series = time_series[num_timesteps // 2 :]

    from ml_module import standard_lstm

    ml_model = standard_lstm(train_series)
    ml_model.train_model()
    print("Performing inference on testing data")
    ml_model.model_inference(test_series)

    # Return basis as (num_modes, num_dofs) to match C++ inspection logic.
    return_data = v[0:3, :]
    return return_data


if __name__ == "__main__":
    pass
