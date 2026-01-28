import numpy as np

def calculate_threshold(errors, method="std", factor=3):
    """
    Threshold = mean + factor * std
    """
    mean = np.mean(errors)
    std = np.std(errors)
    return mean + factor * std
