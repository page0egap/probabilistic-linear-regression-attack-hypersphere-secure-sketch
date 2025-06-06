import itertools
import numpy as np
import math
from scipy.stats import ortho_group


def translate_error_angle_to_error_noise(error_angle: float) -> float:
    """
    Convert error angle to error noise.

    :param error_angle: The error angle in degrees.
    :return: The error noise.
    """
    return math.tan(error_angle * np.pi / 180)


def translate_error_angle_to_error_noise_n(error_angle: float) -> float:
    """
    Convert error angle to error noise(multiple sketches version).

    :param error_angle: The error angle in degrees.
    :return: The error noise.
    """
    error_angle = math.acos(math.sqrt(math.cos(error_angle * np.pi / 180)))
    return math.tan(error_angle)

def log2c_alpha(dimension: int, alpha: int) -> float:
    """
    Calculate the logarithm of the number of combinations of dimension choose alpha.

    :param dimension: The total number of elements.
    :param alpha: The number of elements to select.
    :return: The logarithm of the number of combinations.
    """
    result = 0
    for i in range(alpha):
        result += np.log2(dimension - i) - np.log2(i + 1)
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
        result += np.log2(n - i) - np.log2(n - a - i)
    return result


def get_naive_isometry_rotation_matrix(a, b):
    """
    Get naive isometry rotation matrix M mapped unit vector a to unit vector b
    """
    assert a.shape == b.shape, "input vector shape should be same"
    assert a.shape[1] == 1, (
        "vector shape should be (n, 1)"
    )  # assert shape of a is (n, 1)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    theta = np.arccos(np.dot(a.T, b)).item()
    matrix_theta = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    w = b - a.T @ b * a
    w = w / np.linalg.norm(w)
    tmp_matrix = np.concatenate((a, w), axis=1)
    R = np.eye(len(a)) - a @ a.T - w @ w.T + tmp_matrix @ matrix_theta @ tmp_matrix.T
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


def get_angle_of_two_vectors(a: np.ndarray, b: np.ndarray, type="degree"):
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


def random_unit_vector(n: int) -> np.ndarray:
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


def __task(error_noise, a, dimension, alpha) -> bool:
    import sample
    import ironmask

    codeword = sample.sample_codeword(dimension, alpha)
    matrix_sketch = ironmask.generate_secure_sketch(a, codeword)
    noise_a = sample._generate_random_unit_vector_nearby(a, error_noise)
    recovered_c = ironmask.decode_codeword(matrix_sketch @ noise_a, dimension, alpha)
    return np.allclose(codeword, recovered_c)


def get_real_error_torelant_angle(dimension, alpha, desired_success_rate) -> int:
    """
    Get the real error torelant angle for a given dimension, alpha and desired success rate

    :param dimension: dimension of the vector
    :param alpha: number of non-zero elements in the codeword
    :param desired_success_rate: desired success rate
    :return: real error torelant angle
    """
    from tqdm.auto import tqdm
    from multiprocessing import Pool
    from functools import partial

    default_test_times_for_each_angle = 100
    angle_step = 1
    angle_list = np.arange(0, 90, angle_step)
    a = random_unit_vector(
        dimension
    )  # the randomness of input vector is not relavant to the result, since the mapped codeword is random
    with Pool() as pool:
        for angle in tqdm(
            angle_list,
            desc="Testing real error torelant angle",
            leave=False,
        ):
            error_noise = np.tan(angle / 180 * np.pi)
            fixed_task = partial(__task, error_noise, a, dimension, alpha)
            success_times = pool.starmap(
                fixed_task,
                [()] * default_test_times_for_each_angle,
            ).count(True)
            if success_times / default_test_times_for_each_angle < desired_success_rate:
                return angle - angle_step
    return 90


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
