"""
Author: Ben Campbell

Purpose: The goal of the code herein is to define regions of the device in which residual stresses have led to fractures in the device.
These fractures are then able to allow freer passage of conductive nanoparticles. This is my proposed alternative to directly modeling the residual forces.
The way Sam set that up did not seem physical nor was it affecting any visible change in the nanoparticle and current dynamics.
"""

from numpy.random import default_rng
from defineParameters import params
#from scipy.stats import beta # beta distrobution to push the voids towards the electrodes
from scipy.special import gamma
import jax
import jax.numpy as jnp


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
    x_centers = jnp.copy(Lx * rng.beta(a=0.5, b=0.5, size=n_regions))# for a = b = 0.5 the centers will be more likely to appear at the edges
    y_centers = jnp.copy((rng.uniform(0, 1, size=n_regions) - 0.5) * Ly)
    centers = jnp.column_stack([x_centers, y_centers])

    # Blob widths and amplitudes
    sigmas = jnp.copy(blob_scale * min(Lx, Ly) * rng.uniform(0.7, 1.3, size=n_regions)) # this determines how quickly the blob's influence decays
    amplitudes = jnp.copy(rng.uniform(0.8, 1.2, size=n_regions))

    # Threshold chosen so regions remain disconnected and globular
    threshold = 0.5 * jnp.mean(amplitudes)

    def indicator(x, y):
        x1 = jnp.copy(x)
        y1 = jnp.copy(y) # this serves to convert the input numpy arrays into JAX arrays

        #phi = np.zeros_like(x1)
        '''for (cx, cy), A, s in zip(centers, amplitudes, sigmas):
            phi += A * np.exp(-((x1 - cx)**2 + (y1 - cy)**2) / (2 * s**2))''' # pre JAX conversion
        '''cx = centers[:, 0, jnp.newaxis, jnp.newaxis] # by embeding the arrays in a higher dimension, this creates a matrix when subtrated from the x1 vector. Thus we compute in a matrix and sum along a single axis of the matrix to get our resultant vector.
        cy = centers[:, 1, jnp.newaxis, jnp.newaxis]
        A = amplitudes[:, jnp.newaxis, jnp.newaxis]
        s = sigmas[:, jnp.newaxis, jnp.newaxis]

        # Calculate all Gaussians simultaneously: shape becomes (N, H, W)
        phi_all = A * jnp.exp(-((x1 - cx)**2 + (y1 - cy)**2) / (2 * s**2))

        # Sum across the N dimension (the individual blobs)
        phi = jnp.sum(phi_all, axis=0)'''
        pts = jnp.stack([x1, y1], axis=-1)
        A = amplitudes[:, jnp.newaxis, jnp.newaxis]
        s = sigmas[:, jnp.newaxis, jnp.newaxis]
        cent = centers[:, None, None, :]
        diff = pts - cent 
        r2 = jnp.sum(diff**2, axis=-1)
        phi_all = A * jnp.exp(-r2 / (2 * s**2))
        phi = jnp.sum(phi_all, axis=0)

        return (phi > threshold) # returns the Boolean value; True (1) at element i if point in element i is in the region
    
    indicator = jax.jit(indicator) # turn function into a jax function - should speed up signifigantly

    return indicator

'''def beta_curve(x,alpha,beta):
    return x**(alpha-1) * (1-x)**(beta-1) * gamma(alpha+beta) / gamma(alpha) / gamma(beta)

desired_beta = lambda x: params["viscosity_multiplier"] * beta_curve(x/params['L_x'],0.5,0.5) # inverse bell curve - low in middle - high at edges''' #This didn't work because it blew up at the edges
def visc_dense_curve(x,a=2.5):
    return 1 + a - (a/ (1+(x/params["L_x"]-0.5)**2) )


#from numpy import zeros_like
def viscocity_gradient(x,y,indicator):
    '''out = jnp.zeros_like(x)
    out.at[indicator(x,y)==0].set(params["eta"] * desired_beta(x))
    out.at[indicator(x,y)!=0].set(params["eta"] * desired_beta(x))
    return out'''
    '''out = zeros_like(x)
    out_void_mask = (indicator(x,y)==0)
    out[out_void_mask] = (params["eta"] * visc_dense_curve(x[out_void_mask]))
    out[~out_void_mask] = params["eta"]
    return out'''
    mask = indicator(x, y)  # works for both grid and particles

    eta_dense = params["eta"] * visc_dense_curve(x)
    return jnp.where(mask, params["eta"], eta_dense)

#viscocity_gradient = jax.jit(viscocity_gradient)


#viscocity_gradient = jax.jit(viscocity_gradient)



if __name__ == "__main__": # view the viscosity map if the script is run directly
    import matplotlib.pyplot as plt
    from numpy import linspace, meshgrid

    ind_func = make_globular_indicator()

    x = linspace(0,params["L_x"],1000)
    y = linspace(-params["L_y"]/2.,params["L_y"]/2.,1000)

    X, Y = meshgrid(x,y)

    Z = ind_func(X,Y)

    plt.imshow(Z)

    plt.title(f"Stress induced void. Params: n_regions={params["n_regions"]}, blob_scale={params["blob_scale"]}")

    plt.colorbar()

    #print(ind_func([0,0.2*params["L_x"]],[0,0.2*params["L_x"]]))

    plt.show()

    Z = viscocity_gradient(X,Y,ind_func)
    plt.imshow(Z)
    plt.colorbar()
    plt.show()

    if False: # plot beta dist
        rng1 = default_rng(1)
        plt.hist(rng1.beta(a=0.5, b=0.5, size=1000))
        plt.show()