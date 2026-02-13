import numpy as np

import numpy as np

def get_next_index(tot_weights, x, y, currx, curry, curr_indexes, L):
    """
    Translated from MATLAB getNextIndex.m
    Determines the next index and corresponding (x,y) position for each electron.
    """
    num_e, cols = tot_weights.shape # num_e is the number of electrons beign simulated, cols is 
    n = cols - 2  # number of particle targets (minus 2 electrodes)

    LEFT_ELECTRODE = n + 1
    RIGHT_ELECTRODE = n + 2

    # Normalize hopping probabilities
    row_sums = np.sum(tot_weights, axis=1, keepdims=True)
    W_normalized = tot_weights / row_sums
    W_cdf = np.cumsum(W_normalized, axis=1)

    # Random choice via CDF
    rands = np.random.rand(num_e, 1)
    C = rands < W_cdf
    next_indices = np.argmax(C, axis=1) + 1  # MATLAB is 1-based, Python is 0-based; adjust later if needed

    # Initialize next positions
    nextx = np.zeros(num_e)
    nexty = np.zeros(num_e)

    # Beginning electrode (x=0, y=nan)
    mask_left = next_indices == LEFT_ELECTRODE
    nextx[mask_left] = 0
    nexty[mask_left] = np.nan

    # Right electrode (x=L, y=nan)
    mask_right = next_indices == RIGHT_ELECTRODE
    nextx[mask_right] = L
    nexty[mask_right] = np.nan

    # Particles in between
    mask_particles = (~mask_left) & (~mask_right)
    nextx[mask_particles] = x[next_indices[mask_particles] - 1]
    nexty[mask_particles] = y[next_indices[mask_particles] - 1]

    return next_indices, nextx, nexty