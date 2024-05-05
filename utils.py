import itertools
import numpy as np

def subset_n_a_k(n, a, k):
    """
    return log_2(n choose k / n -a choose k)
    """
    result = 0
    for i in range(k):
        result += np.log2(n-i) - np.log2(n-a-i)
    return result

def get_rotation_matrix(a, b):
    """
    Get rotation matrix M mapped unit vector a to unit vector b
    """
    assert a.shape == b.shape, "input vector shape should be same"
    assert a.shape[1] == 1, "vector shape should be (n, 1)"  # assert shape of a is (n, 1)   
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    theta = np.arccos(np.dot(a.T, b)).item()
    matrix_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    w = b - a.T @ b * a
    w = w / np.linalg.norm(w)
    tmp_matrix = np.concatenate((a, w), axis=1)
    r = np.eye(len(a)) - a @ a.T - w @ w.T  + tmp_matrix @ matrix_theta @ tmp_matrix.T
    return r


def decode_vector_with_k_nonzero_elements(v, n, k):
    """
    Decode vector v with dimension n and k non-zero elements(fill with 1 or -1) in random positions.
    """
    assert k <= n
    positions = np.argsort(np.abs(v))[-k:]
    a = np.zeros(n)
    a[positions] = np.sign(v[positions])
    return a

def n_k_nonzero_vector_iterator(n, k):
    """
    Generate all possible vectors with dimension n and k non-zero elements(fill with 1 or -1) in ordered positions.
    """
    assert k <= n
    for positions in itertools.combinations(range(n), k):
        # iterate all possible values in length k
        for values in itertools.product([-1, 1], repeat=k):
            yield positions, values

def get_angle_of_two_vectors(a: np.ndarray, b:np.ndarray, type="degree"):
    """
    Get the angle between two vectors
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    theta = np.arccos(np.clip(np.dot(a.T, b), -1, 1)).item()
    if type == "degree":
        theta = theta * 180 / np.pi
    return theta

def get_normal_vector(point_list, dimension):
    """
    get normal vector of the hyperplane which contains all points in point_list
    """
    if len(point_list) < dimension:
        raise Exception("dimension of hyperplane is larger than number of points")
    A = np.array(point_list)
    b = np.ones(len(point_list))
    x = np.linalg.solve(A, b)
    x = x / np.linalg.norm(x)
    return x