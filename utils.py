import itertools
import numpy as np
from scipy.stats import ortho_group


def log2c_alpha(dimension:int, alpha: int) -> float:
    """
    Calculate the logarithm of the number of combinations of dimension choose alpha.

    :param dimension: The total number of elements.
    :param alpha: The number of elements to select.
    :return: The logarithm of the number of combinations.
    """
    result = 0
    for i in range(alpha):
        result += np.log2(dimension-i) - np.log2(i+1)
    return result

def subset_n_a_alpha(n, a, alpha):
    """
    Calculate the logarithm of the probability that sampled different `a` elements are all in the subset of (n - alpha) elements given n elements.

    Parameters:
    n (int): The total number of elements.
    a (int): The number of elements to select.
    alpha (int): The number of elements are not in the subset.

    Returns:
    float: The logarithm of the ratio of probability.
    """
    result = 0
    for i in range(alpha):
        result += np.log2(n-i) - np.log2(n-a-i)
    return result

def get_naive_isometry_rotation_matrix(a, b):
    """
    Get naive isometry rotation matrix M mapped unit vector a to unit vector b
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
    R = np.eye(len(a)) - a @ a.T - w @ w.T  + tmp_matrix @ matrix_theta @ tmp_matrix.T
    return R


def codewords_iterator(n, alpha):
    """
    Generate all possible vectors/codewords with dimension n and alpha non-zero elements(fill with 1 or -1) in ordered positions.

    :param n: dimension
    :param alpha: number of non-zero elements
    :return: iterator of all possible vectors/codewords
    """
    assert alpha <= n
    for positions in itertools.combinations(range(n), alpha):
        # iterate all possible values in length k
        for values in itertools.product([-1, 1], repeat=alpha):
            yield positions, values

def get_angle_of_two_vectors(a: np.ndarray, b:np.ndarray, type="degree"):
    """
    Get the angle between two vectors

    :param a: vector a
    :param b: vector b
    :param type: "degree" or "radian"
    :retrun: angle between two vectors
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    theta = np.arccos(np.clip(np.dot(a.T, b), -1, 1)).item()
    if type == "degree":
        theta = theta * 180 / np.pi
    return theta


def random_unit_vector(n:int) -> np.ndarray:
    """
    Generate a random unit vector in dimension n.

    :param n: Dimension of the vector.
    :return: A unit vector of dimension n.
    """
    a = np.random.normal(size=n)
    a = a / np.linalg.norm(a)
    return a


def random_orthogonal_matrix(n: int) -> np.ndarray:
    """
    Randomly generate an orthogonal matrix.

    :param n: dimension of the matrix.
    :return: An isometric matrix Q of dimension n.
    """
    Q = ortho_group.rvs(dim=n)
    return Q


def random_perm_matrix(n) -> np.ndarray:
    """
    Generate random permutation matrix

    :param n: dimension of the matrix.
    :return: A random permutation matrix P of dimension n.
    """
    P = np.eye(n)
    np.random.shuffle(P)
    # each row with random flip
    for i in range(n):
        if np.random.rand() < 0.5:
            P[i, :] = -P[i, :]
    return P

# def get_normal_vector(point_list, dimension):
#     """
#     get normal vector of the hyperplane that is expanded/defined by points in point_list
#     """
#     if len(point_list) < dimension:
#         raise Exception("dimension of hyperplane is larger than number of points")
#     A = np.array(point_list)
#     b = np.ones(len(point_list))
#     x = np.linalg.solve(A, b)
#     x = x / np.linalg.norm(x)
#     return x