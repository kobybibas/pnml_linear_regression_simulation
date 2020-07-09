import numpy as np


def normalize_vec(v):
    return v / np.sqrt(v.dot(v))


def execute_gram_schmidt(arr_input: np.ndarray) -> np.ndarray:
    arr = np.copy(arr_input)

    _, n = arr.shape

    arr[:, 0] = normalize_vec(arr[:, 0])

    for i in range(1, n):
        arr_i = arr[:, i]  # Take column
        for j in range(0, i):
            arr_j = arr[:, j]
            t = arr_i.dot(arr_j)  # Multiply between columns
            arr_i = arr_i - t * arr_j
        arr[:, i] = normalize_vec(arr_i)  # Store base vector

    return arr


def create_data_matrix(subspace_arr: np.ndarray, y_subspace: np.ndarray, set_size: int) -> (np.ndarray, np.ndarray):
    """
    Create data from a linear combination of a given array's columns.
    :param subspace_arr: array the used as a subspace, the data is generate by a linear combination of the columns
    :param set_size: number of data vector to generate
    :return: x_arr: Every row corresponds to feature vector [x_1, x_2, ..., x_{trainset_size}]^T
    """
    num_features, effective_subspace = subspace_arr.shape

    x_arr, labels = [], []
    for i in range(set_size):
        # alphas = np.random.randn(effective_subspace)
        alphas = 2 * np.random.rand(effective_subspace) - 1  # uniform distribution over [-1,1]
        x_i = np.array([alpha_i * v_i for alpha_i, v_i in zip(alphas, subspace_arr.T)]).sum(axis=0)
        x_arr.append(x_i)

        # Create the label, multiply the subspace labels by the same linear combination as the data
        labels.append(np.sum(alphas * y_subspace))
    x_arr = np.array(x_arr)
    labels = np.expand_dims(np.array(labels), 1)
    return x_arr, labels


def create_data(feature_size: int = 2000, effective_subspace: int = 60,
                trainset_size: int = 200, testset_size: int = 100, noise_var: float = 1e-2, is_debug: bool = False):
    # Create random subspace: real space is 100, subspace synthetic size is 1000
    # arr = np.random.randn(feature_size, effective_subspace)
    arr = 2 * np.random.rand(feature_size, effective_subspace) - 1
    arr_ort = execute_gram_schmidt(arr)

    # Create for each subspace base vector a corresponding label
    # y_subspace = np.random.randn(effective_subspace, 1)
    y_subspace = 2 * np.random.rand(effective_subspace, 1) - 1

    # Verify orthogonality
    if is_debug is True:
        projections = []
        for i in range(arr_ort.shape[1]):
            v1 = np.expand_dims(arr_ort[:, i], 1)
            for j in range(i):
                v2 = np.expand_dims(arr_ort[:, j], 1)
                projections.append(float(v1.T @ v2))
        print('Maximum inner product: ', np.max(projections))

    # # The true model parameters
    # theta_star = np.random.randn(effective_subspace)
    # theta_star = np.expand_dims(normalize_vec(theta_star), 1)

    # Create dataset
    # y = a1 v1 + a2 v2 + ... a100 v100

    # Every row corresponds to feature vector
    # X = [x_1, x_2, ..., x_{trainset_size}]^T
    x_train, y_train = create_data_matrix(arr_ort, y_subspace, trainset_size)
    x_test, y_test = create_data_matrix(arr_ort, y_subspace, testset_size)

    #### different way to create the labels
    # The true model parameters
    theta_star = np.random.randn(feature_size)
    theta_star = np.expand_dims(normalize_vec(theta_star), 1)
    y_train = x_train @ theta_star
    y_test = x_test @ theta_star

    # Create labels:
    # y_1 = x_1^T * arr_ort
    # ... = ...
    # y_N = x_N^T * arr_ort
    y_train += noise_var * np.random.randn(trainset_size, 1)
    y_test += noise_var * np.random.randn(testset_size, 1)

    # trainset size is 100, testset size is 100
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    create_data(is_debug=True)
