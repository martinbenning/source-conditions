from numpy import abs, argpartition, expand_dims, maximum, sign, where, zeros_like
from numpy.linalg import norm, svd

def l0_bound(argument, bound):
    abs_argument = abs(argument)
    flat_abs_argument = abs_argument.flatten()
    indices = argpartition(flat_abs_argument, -bound)[-bound:]
    mask = zeros_like(flat_abs_argument, dtype=bool)
    mask[indices] = True
    mask = mask.reshape(*abs_argument.shape)
    return where(mask, argument, 0)

def l2ball_projection(argument, axis=None):

    denominator = maximum(1, norm(argument, 2, axis))
    #return argument / denominator.reshape(-1, 1)
    return argument / expand_dims(denominator, axis)

def nuclear_norm(argument, threshold):

    U, S, Vt = svd(argument, full_matrices=False)
    S = (S - threshold).clip(0)
    return (U * S) @ Vt

def soft_thresholding(argument, threshold):

    output = argument/abs(argument) * (abs(argument) - threshold).clip(0)
    output[abs(argument) == 0] = 0
    return output

#    return sign(argument) * (abs(argument) - threshold).clip(0)

def subsampled_fourier_prox(argument, mask, F, data, scalar):    
    
    return F.T @ ((F @ argument + scalar * data) / (1 + scalar * mask))