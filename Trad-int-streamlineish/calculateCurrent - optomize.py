from defineParameters import params

from numpy import (
    exp,
    abs,
    minimum,
    argsort,
    column_stack,
    zeros,
    float32,
    searchsorted,
)

from numpy.random import rand
from scipy.spatial import cKDTree


def calculate_current(
    x,
    y,
    L_x,
    L_y,
    Volt,
    lambda_,
    Rt,
    T,
    num_e,
):
    """
    Optimized version of Ben's Monte-Carlo transport algorithm.

    Produces the same behaviour as the original implementation
    while avoiding repeated KDTree construction.
    """

    if Volt == 0.0:
        return 0.0

    # ------------------------------------------------------------------
    # Convert once to float32
    # ------------------------------------------------------------------

    x = x.astype(float32, copy=False)
    y = y.astype(float32, copy=False)

    n_sites = len(x)

    # ------------------------------------------------------------------
    # Precompute constants
    # ------------------------------------------------------------------

    beta = params["q"] / (params["k_B"] * T)
    field_factor = beta * Volt / L_x

    # ------------------------------------------------------------------
    # Sort x coordinates for fast searchsorted
    # ------------------------------------------------------------------

    x_sort_idx = argsort(x)

    x_sort = x[x_sort_idx]
    y_sort = y[x_sort_idx]

    # ------------------------------------------------------------------
    # Build ONE KDTree
    # ------------------------------------------------------------------

    coords = column_stack((x, y))
    tree = cKDTree(coords)

    # ------------------------------------------------------------------
    # Initial electron y-positions
    # ------------------------------------------------------------------

    y0 = (L_y * rand(num_e)).astype(float32)

    resist = zeros(num_e, dtype=float32)

    # Number of nearby defects examined
    k_neighbors = min(16, n_sites)

    # ------------------------------------------------------------------
    # Electron loop
    # ------------------------------------------------------------------

    for e in range(num_e):

        x_e = 0.0
        y_e = y0[e]

        resist_e = 0.0

        while x_e < L_x:

            # ----------------------------------------------------------
            # Estimate nearest x-spacing for Boltzmann bias
            # ----------------------------------------------------------

            i = searchsorted(x_sort, x_e, side="left")

            if i == 0:
                dx_min = x_sort[0] - x_e

            elif i >= n_sites:
                dx_min = x_e - x_sort[-1]

            else:

                left_dx = x_e - x_sort[i - 1]

                if i == n_sites - 1:
                    right_dx = x_sort[i] - x_e
                else:
                    right_dx = x_sort[i] - x_e

                dx_min = min(left_dx, right_dx)

            boltzmann_F = exp(field_factor * (2.0 * dx_min))

            probability_left = 1.0 / (1.0 + boltzmann_F)

            forward = rand() > probability_left

            if x_e <= 0.0:
                forward = True

            # ----------------------------------------------------------
            # Query nearby defects from global KDTree
            # ----------------------------------------------------------

            dists, inds = tree.query(
                [x_e, y_e],
                k=k_neighbors,
            )

            # scalar return if k=1
            if k_neighbors == 1:
                dists = [dists]
                inds = [inds]

            chosen_dist = None
            chosen_ind = None

            if forward:

                for d_test, ind_test in zip(dists, inds):

                    if x[ind_test] > x_e:
                        chosen_dist = d_test
                        chosen_ind = ind_test
                        break

            else:

                for d_test, ind_test in zip(dists, inds):

                    if x[ind_test] < x_e:
                        chosen_dist = d_test
                        chosen_ind = ind_test
                        break

            # ----------------------------------------------------------
            # If no directional neighbor found among nearest k,
            # jump directly to electrode.
            # ----------------------------------------------------------

            electrode_dist = L_x - x_e

            if chosen_dist is None:

                d = electrode_dist
                x_new = L_x
                y_new = 0.0

            else:

                d = minimum(chosen_dist, electrode_dist)

                if abs(d - electrode_dist) < 1e-15:

                    x_new = L_x
                    y_new = 0.0

                else:

                    x_new = x[chosen_ind]
                    y_new = y[chosen_ind]

            # ----------------------------------------------------------
            # Resistance accumulation
            # ----------------------------------------------------------

            jump_R = Rt * exp(d / lambda_)

            # PRESERVES ORIGINAL CODE'S BEHAVIOR
            resist_e += 2.0 * jump_R

            # If the double-addition was a bug,
            # replace the line above with:
            #
            # resist_e += jump_R

            x_e = x_new
            y_e = y_new

        resist[e] = resist_e

    # Preserve original testing behaviour
    R = resist.min()

    return Volt / R