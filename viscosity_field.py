"""
Author: Ben Campbell

Purpose: The goal of the code herein is to define regions of the device in which residual stresses have led to fractures in the device.
These fractures are then able to allow freer passage of conductive nanoparticles. This is my proposed alternative to directly modeling the residual forces.
The way Sam set that up did not seem physical nor was it affecting any visible change in the nanoparticle and current dynamics.
"""

from numpy import asarray, zeros_like, exp, mean
from numpy.random import default_rng
from defineParameters import params

def make_globular_indicator(Lx=params["L_x"], Ly=params["L_y"], n_regions=params["n_regions"], blob_scale=params["blob_scale"], seed=params["seed"]):
    """
    Returns a function f(x, y) -> {0, 1} defining globular regions.

    Parameters
    ----------
    Lx, Ly : float
        Box dimensions.
    n_regions : int
        Approximate number of globular '1' regions.
    blob_scale : float
        Controls blob size relative to box size.
    seed : int or None
        Random seed.

    Returns
    -------
    f : callable
        Function f(x, y) that returns 0 or 1.
    """

    rng = default_rng(seed)

    # Blob centers
    centers = rng.uniform([0, 0], [Lx, Ly], size=(n_regions, 2))

    # Blob widths and amplitudes
    sigmas = blob_scale * min(Lx, Ly) * rng.uniform(0.7, 1.3, size=n_regions) # this determines how quickly the blob's influence decays
    amplitudes = rng.uniform(0.8, 1.2, size=n_regions)

    # Threshold chosen so regions remain disconnected and globular
    threshold = 0.5 * mean(amplitudes)

    def indicator(x, y):
        x = asarray(x)
        y = asarray(y)

        phi = zeros_like(x, dtype=float)
        for (cx, cy), A, s in zip(centers, amplitudes, sigmas):
            phi += A * exp(-((x - cx)**2 + (y - cy)**2) / (2 * s**2))

        return (phi > threshold).astype(int) # returns the Boolean value as 0 or 1; 1 if in the void

    return indicator



if __name__ == "__main__": # view the viscosity map if the script is run directly
    import matplotlib.pyplot as plt
    from numpy import linspace, meshgrid

    ind_func = make_globular_indicator()

    x = linspace(0,params["L_x"],1000)
    y = linspace(0,params["L_y"],1000)

    X, Y = meshgrid(x,y)

    Z = ind_func(X,Y)

    plt.imshow(Z)

    plt.title(f"Stress induced void. Params: n_regions={params["n_regions"]}, blob_scale={params["blob_scale"]}")

    plt.colorbar()

    #print(ind_func([0,0.2*params["L_x"]],[0,0.2*params["L_x"]]))

    plt.show()