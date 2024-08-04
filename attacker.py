from typing import List
import scipy
import numpy as np
import torch
from tqdm.auto import tqdm
from multiprocessing import Pool, Manager
import ironmask
from sample import random_sub_matrix_generator, random_sub_matrix_generator_with_known_places
import utils
from utils import random_perm_matrix

#######################################################################
## SVD solver

def submatrix_solver_scipy(submatrix:np.ndarray):
    """
    Return the null vector of the submatrix(svd)
    """
    null_vector = scipy.linalg.null_space(submatrix)[:, 0]
    return null_vector

@torch.no_grad()
def submatrix_solver_numpy(submatrix:np.ndarray):
    """
    Return the null vector of the submatrix(svd)
    """
    null_vector = np.linalg.svd(submatrix)[2][-1, :]
    return null_vector

@torch.no_grad()
def submatrix_solver_torch(submatrix:torch.Tensor):
    """
    Return the null vector of the submatrix(svd)
    """
    _, _, Vh = torch.linalg.svd(submatrix)
    null_vector = Vh[-1, :]
    return null_vector


###################################################################
## LSA solver
def local_search_solver(dimension, k, matrix, outer_loop_number = 20, inner_loop_number = 800, disable_flag=False, **kwargs):
    """
    Local search algorithm solver
    outer_loop_number: outer loop maximum iteration number
    inner_loop_number: inner loop maximum iteration number
    threshold: esitmated error rate
    """
    a = kwargs.get("a", None)
    threshold = kwargs.get("threshold", 1e-1) / np.sqrt(dimension) * np.sqrt(k)
    for i in (pbar:=tqdm(range(outer_loop_number), desc="Iterating through all possible vectors, threshold: {}".format(threshold * np.sqrt(matrix.shape[0])), disable=disable_flag)):
        codeword = ironmask.sample_codeword(dimension, k)
        candidate_better_codeword = None
        normb = 16
        for j in range(inner_loop_number):
            tmpb = matrix @ codeword
            # get the norm of tmpb
            normb = np.linalg.norm(tmpb)
            # get the index of codeword where are not zero and iterate on it
            nonzero = np.where(np.abs(codeword) > 0.1)[0]
            candidate_better_codeword = None
            tmp_matrix = matrix.copy()
            # set columns in non_zero positions with 10 * np.ones
            tmp_matrix[:, nonzero] = 10 * np.ones((matrix.shape[0], len(nonzero)))
            for j1 in nonzero:
                tmpbb = tmpb - codeword[j1] * matrix[:, j1]
                # add
                tmpbbb = tmp_matrix + tmpbb[:, np.newaxis]
                normbbb = np.linalg.norm(tmpbbb, axis=0)
                # get min value and index of normbbb
                min_index = np.argmin(normbbb)
                min_value = normbbb[min_index]
                if min_value < normb:
                    candidate_better_codeword = codeword.copy()
                    candidate_better_codeword[j1] = 0
                    candidate_better_codeword[min_index] = 1
                    normb = min_value
                # sub
                tmpbbb =  tmpbb[:, np.newaxis] - tmp_matrix
                normbbb = np.linalg.norm(tmpbbb, axis=0)
                # get min value and index of normbbb
                min_index = np.argmin(normbbb)
                min_value = normbbb[min_index]
                if min_value < normb:
                    candidate_better_codeword = codeword.copy()
                    candidate_better_codeword[j1] = 0
                    candidate_better_codeword[min_index] = -1
                    normb = min_value
            pbar.set_postfix({"norm": normb, "runtime" : j})
            if candidate_better_codeword is not None:
                codeword = candidate_better_codeword
                continue
            else:
                break
        if normb < threshold * np.sqrt(matrix.shape[0]):
            return codeword
    return None

#################################################################################################################################
### one matrix
@torch.no_grad()
def solve_puzzle_with_one_matrix(isometric_matrix:np.ndarray, dimension, alpha, each_guessing=0, threshold=40, scale=1):
    """
    (SVD optimized)Solve the puzzle only with one matrix(computed by two sketches M1, M2 as M1^T * M2, thus mapping one codeword to another); iteration_times = scale * expected iteration times with no noise

    :param isometric_matrix: isometric matrix
    :param dimension: dimension of the vector
    :param alpha: number of non-zero elements in the codeword
    :param threshold: threshold of the angle between the decoded vector and the mapped vector
    :param scale: scale of the expected iteration times
    :param each_guessing: the number guessed zero-positions in first codeword, must be >= dimension//2
    """
    # estimate how many times we need to test
    rtimes = 1
    if each_guessing == 0:
        each_guessing = dimension // 2
    assert each_guessing >= dimension//2, "each_guessing should be larger than dimension//2"
    for i in range(each_guessing):
        rtimes = rtimes * (dimension - i) / (dimension - i - alpha)
    for i in range(each_guessing):
        rtimes = rtimes * (dimension - i) / (dimension - i - alpha)
    rtimes = int(rtimes * scale)
    angle_min = 180
    for _ in (pbar:=tqdm(range(rtimes), "Solving(svd)")):
        # random select dimension //2 positions, dtype should be int
        positions_a = np.random.choice(dimension, each_guessing, replace=False)
        positions_b = np.random.choice(dimension, each_guessing, replace=False)
        submatrix = isometric_matrix[positions_b, :][:, positions_a]
        null_vector = submatrix_solver_numpy(submatrix)
        assume_vector = np.zeros(dimension)
        for i in range(len(null_vector)):
            assume_vector[positions_a[i]] = null_vector[i]
        # get b
        assume_b = np.matmul(isometric_matrix, assume_vector)
        b = ironmask.decode_codeword(assume_b, dimension, alpha)
        assume_a = isometric_matrix.T @ b
        decode_a = ironmask.decode_codeword(assume_a, dimension, alpha)
        # if the angle between decode_a and assume_a is below threshold, return
        angle = utils.get_angle_of_two_vectors(assume_a, decode_a)
        angle_min = min(angle, angle_min)
        pbar.set_postfix({"angle": angle_min})
        if angle < threshold:
            return assume_vector, b
        else:
            continue
    return None, None

@torch.no_grad()
def solve_puzzle_with_one_matrix_TMTO(isometric_matrix:np.ndarray, dimension, k, threshold=40, scale=1):
    pass

def solve_puzzle_with_one_matrix_greedy(isometric_matrix:np.ndarray, dimension, k, threshold=30):
    rtimes = 1
    for i in range(k):
        rtimes = rtimes * (dimension - i) / (i + 1)
    rtimes = int(rtimes * (2 ** k))
    for positions, values in tqdm(utils.codewords_iterator(dimension, k), total=rtimes, desc="Iterate over all possible vectors"):
        assume_b = np.einsum('i,ji->j', values, isometric_matrix[:, positions])
        b = ironmask.decode_codeword(assume_b, dimension, k)
        angle = utils.get_angle_of_two_vectors(b, assume_b)
        if angle < threshold:
            c = np.zeros(dimension)
            c[list(positions)] = values
            return c, b
    return None

#################################################################################################################################
### n matrixes

def solve_puzzle_with_n_matrix_known_places(isometric_matrixes:List[np.ndarray], mapped_codeword_list, dimension, alpha, threshold=40, max_rtimes=1000, algorithm = "SVD", **kwargs):
    """
    Solve puzzle under the constraint that the linear equation sampler always sample "correct" matrix M, i.e. M w \\approx 0 where w is original template vector. We ensure that mapped_codeword_list is only used for sampling "correct" matrix, and no other usage.

    kwargs: disable_tqdm, return_runtimes, k_each_matrix(default 1)
    """
    # mapped_codeword_list is only used here to get the zero-element positions of these codewords for sampling "correct" matrix
    positions_bs = [np.argsort(np.abs(b))[:kwargs.get("num_select_positions", dimension-alpha)] for b in mapped_codeword_list]
    angle_min = 180
    run_times = 0
    get_result_times = 0
    if algorithm == "SVD":
        solver = submatrix_solver_scipy if len(isometric_matrixes) == (dimension - 1) else submatrix_solver_numpy
        isometric_matrixes_for_generate = isometric_matrixes
        positions_bs_for_generate = positions_bs
    elif algorithm == "LSA":
        solver = lambda x: local_search_solver(dimension, alpha, x, outer_loop_number=1, disable_flag=True, threshold=kwargs.get("error_rate", 1e-1))
        isometric_matrixes_for_generate = [isometric_matrixes[i] @ isometric_matrixes[0].T for i in range(1, len(isometric_matrixes))]
        positions_bs_for_generate = [positions_bs[i] for i in range(1, len(isometric_matrixes))]
    else:
        raise ValueError("Unknown algorithm")
    with tqdm(total=max_rtimes, desc="Solving({})".format(algorithm), disable=kwargs.get("disable_tqdm", False), leave=True) as pbar:
        for sub_matrix in random_sub_matrix_generator_with_known_places(isometric_matrixes_for_generate, positions_bs_for_generate, max_rtimes, k_each_matrix=kwargs.get("k_each_matrix", 1)):
            run_times += 1
            pbar.update()
            pbar.set_postfix({"angle_min": angle_min, "get_result_times": get_result_times})
            null_vector = solver(sub_matrix)
            if null_vector is None:
                continue
            get_result_times += 1
            matrix_1 = isometric_matrixes[0]
            if algorithm == "SVD":
                assume_vector = null_vector
                assume_b = np.matmul(matrix_1, assume_vector)
                b = ironmask.decode_codeword(assume_b, dimension, alpha)
            else:
                assume_b = b = null_vector
                assume_vector = isometric_matrixes[0].T @ b
            # threshold determinant
            matrix_2 = isometric_matrixes[1] @ matrix_1.T
            assume_c = matrix_2 @ b
            c = ironmask.decode_codeword(assume_c, dimension, alpha)
            angle = utils.get_angle_of_two_vectors(c, assume_c)
            angle_min = min(angle, angle_min)
            pbar.set_postfix({"angle_min": angle_min, "get_result_times": get_result_times})
            if angle < threshold:
                if kwargs.get("return_runtimes", False):
                    return assume_vector, b, run_times
                else:
                    return assume_vector, b
    if kwargs.get("return_runtimes", False):
        return None, None, run_times
    else:
        return None, None

def solve_puzzle_with_n_matrix(isometric_matrixes:List[np.ndarray], dimension, alpha, threshold=40, scale=1, algorithm = "SVD", **kwargs):
    """
    kwargs: disable_tqdm, k_each_matrix(default 1)
    """
    # estimate how many times we need to test
    rtimes = 1
    rtimes = rtimes * (2 ** (utils.subset_n_a_alpha(dimension, alpha, 1) * kwargs.get("k_each_matrix", 1)) ) ** len(isometric_matrixes)
    rtimes = int(rtimes * scale)
    angle_min = 180

    if algorithm == "SVD":
        solver = submatrix_solver_scipy if len(isometric_matrixes) == (dimension - 1) else submatrix_solver_numpy
        isometric_matrixes_for_generate = isometric_matrixes
    elif algorithm == "LSA":
        solver = lambda x: local_search_solver(dimension, alpha, x, outer_loop_number=1, disable_flag=True, threshold=kwargs.get("error_rate", 1e-1))
        isometric_matrixes_for_generate = [isometric_matrixes[i] @ isometric_matrixes[0].T for i in range(1, len(isometric_matrixes))]
    else:
        raise ValueError("Unknown algorithm")
    
    with tqdm(total=rtimes, desc="Solving({})".format(algorithm)) as pbar:
        for sub_matrix in random_sub_matrix_generator(isometric_matrixes_for_generate, rtimes, k_each_matrix=kwargs.get("k_each_matrix", 1)):
            pbar.update()
            pbar.set_postfix({"angle_min": angle_min})
            null_vector = solver(sub_matrix)
            if null_vector is None:
                continue
            if algorithm == "SVD":
                assume_vector = null_vector
                assume_b = np.matmul(isometric_matrixes[0], assume_vector)
                b = ironmask.decode_codeword(assume_b, dimension, alpha)
            else:
                assume_b = b = null_vector
                assume_vector = isometric_matrixes[0].T @ b
            matrix_2 = isometric_matrixes[1] @ isometric_matrixes[0].T
            assume_c = matrix_2 @ b
            c = ironmask.decode_codeword(assume_c, dimension, alpha)
            angle = utils.get_angle_of_two_vectors(c, assume_c)

            angle_min = min(angle, angle_min)
            pbar.set_postfix({"angle_min": angle_min})
            if angle < threshold:
                return assume_vector, b
    return None, None


#################################################################################################################
# one matrix MR with M has particular shape
def solve_puzzle_with_MR(MR, n, alpha, threshold=35, tqdm_disable=False):
    """
    solve original template if MR = T @ R where T is a permutation matrix with
    flip and R is a naive rotation matrix
    """
    max_indices = min(n, 8)
    collect_vectors = []
    all_num = 0
    for indices_num in tqdm(range(2, max_indices), disable=tqdm_disable):   
        for i in (pbar:=tqdm(range(n-1), disable=tqdm_disable)):
            indices = np.random.choice(range(n), indices_num, replace=False)
            test_num = 0
            while test_num < 1:
                M2 = random_perm_matrix(n)
                M2R = M2 @ MR     
                # get the submatrix i,j * i,j of MR
                M_sub = M2R[indices][:, indices]
                # get the null vector of M @ R
                null_vector = scipy.linalg.null_space(M_sub)
                if null_vector.shape[1] == 0:
                    test_num += 1
                    continue
                # recover the null vector to the n dimension space with index i, j
                # norm the null vector
                null_vector = null_vector[:, 0]
                null_vector = null_vector / np.linalg.norm(null_vector)
                result_vector = np.zeros(n)
                result_vector[indices] = null_vector.reshape(-1)
                tmp = MR @ result_vector
                # check tmp has exactly two non-zero elements
                if np.sum(np.abs(tmp) > 1e-6) != indices_num:
                    test_num += 1
                    continue
                else:
                    all_num += 1
                    collect_vectors.append(result_vector)
                    break
            pbar.set_postfix({'success number': all_num})
    
    selected_vectors = np.array(collect_vectors)
    null_vector = scipy.linalg.null_space(selected_vectors)
    for i in range(null_vector.shape[1]):
        b = MR @ null_vector[:, i]
        tmpb = ironmask.decode_codeword(MR @ null_vector[:, i], n, alpha)
        if utils.get_angle_of_two_vectors(tmpb, b.reshape(-1)) < threshold:
            return MR.T @ tmpb
        
    return None