import sample as sampler
import ironmask
import utils
import math

def calculate_extra_noise(dimension: int, alpha: int, initial_noise: float):
    """Calulate how much extra noise need to be added to make two templates's distance approximately equal to the distance of the random vector and its closest codeword.

    Args:
        dimension (int): The dimension of the vector space.
        alpha (int): The parameter of ironmask's error correcting codeword.
        initial_noise (float): The initial noise level of original templates, degree format.
        
    Returns:
        float: The extra noise needed to be added to the original templates.
    """
    mean_angle = calculate_angle_between_random_vector_and_its_nearest_codeword(dimension, alpha)
    extra_angle = math.acos(math.sqrt(math.cos(mean_angle * math.pi / 180) / math.cos(initial_noise * math.pi / 180)))
    return extra_angle


def calculate_angle_between_random_vector_and_its_nearest_codeword(
    dimension: int, alpha: int, iteration_times: int = 1000):
    """Calculate the angle between a random vector and its nearest codeword.

    Args:
        dimension (int): The dimension of the vector space.
        alpha (int): The parameter of ironmask's error correcting codeword.
        iteration_times (int, optional): The number of iterations. Defaults to 1000.
    
    Returns:
        float: The mean angle between the random vector and its nearest codeword.
    """
    angle_list = []
    for i in range(iteration_times):
        random_vector = sampler.random_unit_vector(dimension)
        decode_vector = ironmask.decode_codeword(random_vector, dimension, alpha)
        angle = utils.get_angle_of_two_vectors(random_vector, decode_vector)
        angle_list.append(angle)

    mean_angle = sum(angle_list) / iteration_times
    return mean_angle