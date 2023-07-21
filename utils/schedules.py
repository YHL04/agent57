

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_betas(i, N, beta):
    """
    Args:
        i (int): ID of actor associated with a combination of beta and gamma
        N (int): N constant representing total different combinations of betas and gammas
        beta (float): Maximum beta value

    Returns:
        beta (float): Betas associated with each env
    """
    if i == 0:
        return 0
    elif i == N - 1:
        return beta
    else:
        x = 10 * (2 * i - (N - 2)) / (N - 2)
        return beta * sigmoid(x)


def get_gammas(i, N, gamma_min, gamma_max):
    """
    Args:
        i (int): ID of actor associated with a combination of beta and gamma
        N (int): N constant representing total different combinations of betas and gamma
        gamma_min (float): Minimum gamma value
        gamma_max (float): Maximum gamma value

    Returns:
        gamma (float): Gamma associated with each env
    """
    numerator = (N - 1 - i) * np.log(1 - gamma_max) + i * np.log(1 - gamma_min)
    denominator = N - 1

    return 1 - np.exp(numerator / denominator)

