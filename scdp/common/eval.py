import numpy as np

def get_nmape(pred_density, density):
    # check instance...
    diff = pred_density - density
    return np.abs(diff).sum() / np.abs(density).sum() * 100.
