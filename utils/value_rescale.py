

import torch


def value_rescaling(z, eps=1e-3):
    """
    Rescaling the expected q values with the following function:
    h(x) = sign(x) * sqrt(abs(x) + 1) - 1) + eps * x
    
    Proposed in R2D2: https://openreview.net/pdf?id=r1lyTjAqYX (pg 3)

    Args:
        x (torch.tensor): value to be rescaled

    Returns:
        h (torch.tensor): rescaled value
    """

    z = torch.sign(z) * (torch.sqrt(torch.abs(z) + 1) - 1) + eps * z
    return z


def inverse_value_rescaling(z, eps=1e-3):
    """
    Inverse of h(x) function

    Derived in Never Give Up paper: https://arxiv.org/pdf/2002.06038.pdf (pg 20)
    """
    numerator = torch.sqrt(1 + 4 * eps * (torch.abs(z) + 1 + eps)) - 1
    denominator = 2 * eps

    z = torch.sign(z) * ((numerator / denominator) - 1)
    return z
