import numpy as np
from tqdm.auto import tqdm

from ironmask import generate_secure_sketch, sample_codeword
from utils import random_unit_vector


def _generate_random_unit_vector_nearby(a, error_rate = 0.3):
    """
    Generate random unit vector nearby a with error_rate

    implementation: just add orthognal noise to a and normalize it.
    """
    noise = np.random.normal(size=len(a))
    # make sure noise be orthognal to vector a, where a might not be unit vector
    unit_a = a / np.linalg.norm(a)
    noise = noise - np.dot(noise, unit_a) * unit_a
    noise = noise / np.linalg.norm(noise)
    a2 = a + error_rate * noise * np.linalg.norm(a)
    a2 = a2 / np.linalg.norm(a2)
    return a2

def _generate_puzzle(unprotected_template, n, alpha, error_rate = 0.3):
    """
    Generate a puzzle with codeword b, and the secure sketch M2 that mapped c2 to b, and the cosine error between M2*unprotected_template and b, where c2 is the noisy version of unprotected_template.

    :param unprotected_template: the original unprotected template
    :param n: dimension of the vector
    :param alpha: number of non-zero elements in the codeword
    :param error_rate: error rate of the noisy version of unprotected_template
    :return: b, M2, coserror
    """
    b = sample_codeword(n, alpha)
    # add noise to c
    c2 = _generate_random_unit_vector_nearby(unprotected_template, error_rate)
    # get the orthogonal matrix that mapped c2 to b
    M2 = generate_secure_sketch(c2, b)
    # get the cosine error between M3*a and b
    coserror = np.dot(M2 @ unprotected_template, b) / np.linalg.norm(M2 @ unprotected_template) / np.linalg.norm(b)
    return b, M2, coserror

def generate_puzzle_n(dimension, alpha, error_rate=0.3, **kwargs):
    """
    Generate n(default: dimension - 1) isometric matrixes and return the isometric matrixes and the coserror between the noisy version of unprotected_template and the unprotected_template.

    :param dimension: dimension of the vector
    :param alpha: number of non-zero elements in the codeword
    :param error_rate: error rate of the noisy version of unprotected_template
    :param n: number of puzzles(default: dimension - 1)
    :return: unprotected_template, mapped_codeword_list, secure_sketch_list, coserror_list
    """
    unprotected_template = random_unit_vector(dimension)
    secure_sketch_list = []
    coserror_list = []
    mapped_codeword_list = []
    n = kwargs.get("n", dimension-1)
    for i in tqdm(range(n), desc="Generating Puzzles", disable=kwargs.get("disable_tqdm", False)):
        b, M, coserror = _generate_puzzle(unprotected_template, dimension, alpha, error_rate)
        secure_sketch_list.append(M)
        coserror_list.append(coserror)
        mapped_codeword_list.append(b)
    return unprotected_template, mapped_codeword_list, secure_sketch_list, coserror_list

def random_sub_matrix_generator(isometric_matrixes, rtimes, k_each_matrix=1):
    """
    Random generate submatrix of isometric matrixes with given shape ([number of isometric_matrixes] * k_each_matrix, dimension).

    Implementation: randonly sample k_each_matrix rows from each isometric matrix, and stack them together to form a new matrix.
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


