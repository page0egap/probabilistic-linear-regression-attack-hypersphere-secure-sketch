import numpy as np

import utils
from utils import random_orthogonal_matrix

def sample_codeword(n, alpha):
    """
    Generate random vector with dimension n and alpha non-zero elements(fill with 1 or -1) in random positions.

    Different from the implementation in paper, the non-zero element is filled with 1 or -1 instead of 1/sqrt(alpha) or - 1/sqrt(alpha)

    :param n: dimension
    :param alpha: number of non-zero elements
    :return: random vector with dimension n and alpha non-zero elements(fill with 1 or -1) in random positions
    """
    assert alpha <= n
    a = np.zeros(n)
    a[np.random.choice(n, alpha, replace=False)] = np.random.choice([-1, 1], alpha)
    return a

def decode_codeword(v, n, alpha):
    """
    Decode vector v to the closest codeword in dimension n filled with alpha non-zero elements(fill with 1 or -1).
    """
    assert alpha <= n
    positions = np.argsort(np.abs(v))[-alpha:]
    a = np.zeros(n)
    a[positions] = np.sign(v[positions])
    return a


def generate_secure_sketch(unprotected_template:np.ndarray, codeword:np.ndarray) -> np.ndarray:
    """
    Generate random orthogonal matrix(secure sketch) mapped unit vector unprotected_template to unit vector codeword

    :param unprotected_template: original unprotected template
    :param codeword: ecc codeword that needs to be mapped from unprotected_template
    :return: random orthogonal matrix(secure sketch) M that mapped unprotected_template to codeword
    """
    # normalize, ensure vectors be shape (n, 1)
    unprotected_template = unprotected_template / np.linalg.norm(unprotected_template)
    codeword = codeword / np.linalg.norm(codeword)
    unprotected_template = unprotected_template.reshape(-1, 1)
    codeword = codeword.reshape(-1, 1)
    # get a random isometric matrix
    Q = random_orthogonal_matrix(len(unprotected_template))
    # get the mapped vector of a as q * a
    a_mapped = Q @ unprotected_template
    # get the rotation matrix which is also the isometric matrix that mapped a_mapped to b
    R = utils.get_naive_isometry_rotation_matrix(a_mapped, codeword)
    assert np.allclose(np.dot(R, a_mapped), codeword), print(np.dot(R, a_mapped), codeword)
    assert np.allclose(np.eye(len(unprotected_template)), R @ R.T)
    return np.dot(R, Q)



