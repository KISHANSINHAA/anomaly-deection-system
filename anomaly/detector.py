import numpy as np

def reconstruction_error(x, x_hat):
    """
    Mean Absolute Error across timesteps and features
    """
    return np.mean(np.abs(x - x_hat), axis=(1, 2))
