import numpy as np
from scipy.stats import ortho_group
from tqdm.auto import tqdm

import utils


def random_unit_vector(n):
    """
    Generate random unit vector in dimension n
    """
    a = np.random.normal(size=n)
    a = a / np.linalg.norm(a)
    return a

def random_orthogonal_matrix(n):
    """
    random generate an orthogonal matrix
    input: dimension n
    output: isometric matrix Q
    """
    Q = ortho_group.rvs(dim=n)
    return Q

def random_orthogonal_matrix_with_given_vectors(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    Generate random orthogonal matrix mapped unit vector a to unit vector b
    """
    # normalize a and b, and make a, b be vector with shape (n, 1)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    # get a random isometric matrix
    q = random_orthogonal_matrix(len(a))
    # get the mapped vector of a as q * a
    a_mapped = q @ a
    # get the rotation matrix which is also the isometric matrix that mapped a_mapped to b
    r = utils.get_rotation_matrix(a_mapped, b)
    assert np.allclose(np.dot(r, a_mapped), b), print(np.dot(r, a_mapped), b)
    assert np.allclose(np.eye(len(a)), r @ r.T)
    return np.dot(r, q)

def random_perm_matrix(n) -> np.ndarray:
    """
    Generate random permutation matrix
    """
    P = np.eye(n)
    np.random.shuffle(P)
    # each row with random flip
    for i in range(n):
        if np.random.rand() < 0.5:
            P[i, :] = -P[i, :]
    return P

def random_vector_with_k_nonzero_elements(n, k):
    """
    Generate random vector with dimension n and k non-zero elements(fill with 1 or -1) in random positions.
    """
    assert k <= n
    a = np.zeros(n)
    a[np.random.choice(n, k, replace=False)] = np.random.choice([-1, 1], k)
    return a

def _generate_random_unit_vector_nearby(a, error_rate = 0.3):
    """
    Generate random unit vector nearby a with error_rate
    """
    noise = np.random.normal(size=len(a))
    noise = noise / np.linalg.norm(noise)
    a2 = a + error_rate * noise * np.linalg.norm(a)
    a2 = a2 / np.linalg.norm(a2)
    return a2

def _generate_puzzle(c, dimension, k, error_rate = 0.3):
    b = random_vector_with_k_nonzero_elements(dimension, k)
    # add noise to c
    c2 = _generate_random_unit_vector_nearby(c, error_rate)
    # get the orthogonal matrix that mapped c2 to b
    M2 = random_orthogonal_matrix_with_given_vectors(c2, b)
    # get the cosine error between M3*a and b
    coserror = np.dot(M2 @ c, b) / np.linalg.norm(M2 @ c) / np.linalg.norm(b)
    return b, M2, coserror


def generate_puzzle_all_matrixes(dimension, k, error_rate = 0.0):
    c = random_unit_vector(dimension)
    a, M1, _ = _generate_puzzle(c, dimension, k, 0)
    b, M2, coserror = _generate_puzzle(c, dimension, k, error_rate)
    return a, b, M1, M2, coserror

def generate_puzzle(dimension, k, error_rate = 0.3):
    c = random_unit_vector(dimension)
    a, M1, _ = _generate_puzzle(c, dimension, k, 0)
    b, M2, coserror = _generate_puzzle(c, dimension, k, error_rate)
    M3 = M2 @ M1.T
    return a, b, M3, coserror

def generate_puzzle_n(dimension, k, error_rate=0.3, **kwargs):
    """
    Generate n(default: dimension - 1) isometric matrixes and return the isometric matrixes and the coserror between the codeword and the template.
    """
    c = random_unit_vector(dimension)
    isometric_matrixes = []
    coserrors = []
    bs = []
    n = kwargs.get("n", dimension-1)
    for i in tqdm(range(n), desc="Generating Puzzles", disable=kwargs.get("disable_tqdm", False)):
        b, M, coserror = _generate_puzzle(c, dimension, k, error_rate)
        isometric_matrixes.append(M)
        coserrors.append(coserror)
        bs.append(b)
    return c, bs, isometric_matrixes, coserrors

def random_sub_matrix_generator(isometric_matrixes, rtimes, k_each_matrix=1):
    """
    random generate submatrix of isometric matrixes with given shape (number of isometric_matrixes * k_each_matrix, dimension)
    """
    dimension = isometric_matrixes[0].shape[0]
    num = len(isometric_matrixes)
    vstack_isometric_matrix = np.vstack(isometric_matrixes)
    positions_shift = [[i * dimension] * k_each_matrix for i in range(num)]
    positions_shift = [item for sublist in positions_shift for item in sublist]
    for _ in range(rtimes):
        if k_each_matrix == 1:
            positions = np.random.choice(dimension, num, replace=True)
        else:
            positions = [np.random.choice(dimension, k_each_matrix, replace=False) for _ in range(num)]
            positions = [item for sublist in positions for item in sublist]
        positions = [a + b for a, b in zip(positions, positions_shift)]
        submatrix = vstack_isometric_matrix[positions, :]
        yield submatrix

def random_sub_matrix_generator_with_known_places(isometric_matrixes, positions_bs, rtimes, k_each_matrix=1):
    """
    according to known positions_bs for each isomatric matrix, 
    sample k_each_matrix rows from each isometric matrix, 
    and stack them together to form a new matrix
    """
    assert len(isometric_matrixes) == len(positions_bs)
    dimension = isometric_matrixes[0].shape[0]
    num = len(isometric_matrixes)
    vstack_isometric_matrix = np.vstack(isometric_matrixes)
    positions_shift = [[i * dimension] * k_each_matrix for i in range(num)]
    positions_shift = [item for sublist in positions_shift for item in sublist]
    for _ in range(rtimes):
        # get random k_each_matrix choices for each row of positions_bs, and flatten the list to 1 dimension
        positions = [np.random.choice(positions_bs[i], k_each_matrix, replace=False) for i in range(num)]
        positions = [item for sublist in positions for item in sublist]
        # add by item
        positions = [a + b for a, b in zip(positions, positions_shift)]
        submatrix = vstack_isometric_matrix[positions, :]
        yield submatrix