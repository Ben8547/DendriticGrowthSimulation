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
from scipy.integrate import quad
import numpy as np


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

        pts = jnp.stack([x1, y1], axis=-1)
        A = amplitudes[:, jnp.newaxis, jnp.newaxis]
        s = sigmas[:, jnp.newaxis, jnp.newaxis]
        cent = centers[:, None, None, :]
        diff = pts - cent 
        r2 = jnp.sum(diff**2, axis=-1)
        phi_all = A * jnp.exp(-r2 / (2 * s**2))
        phi = jnp.sum(phi_all, axis=0)

        return (phi > threshold) #jnp.ones_like(x,dtype=int) # (phi > threshold) # returns the Boolean value; True (1) at element i if point in element i is in the region
    
    indicator = jax.jit(indicator) # turn function into a jax function - should speed up signifigantly

    return indicator

def visc_dense_curve(x,a=2.5):
    return params["eta"] + a - (a/ (1+(x/params["L_x"]-0.5)**2) )

average_density = quad(visc_dense_curve,0,params["L_x"])[0] / params['L_x']


#from numpy import zeros_like
def viscocity_gradient(x,y,indicator,Visc_mult):
    '''out = jnp.zeros_like(x)
    out.at[indicator(x,y)==0].set(params["eta"] * desired_beta(x))
    out.at[indicator(x,y)!=0].set(params["eta"] * desired_beta(x))
    return out'''
    '''out = zeros_like(x)
    out_void_mask = (indicator(x,y)==0)
    out[out_void_mask] = (params["eta"] * visc_dense_curve(x[out_void_mask]))
    out[~out_void_mask] = params["eta"]
    return out'''
    #mask = indicator(x, y)  # works for both grid and particles
    density = visc_dense_curve(x)*Visc_mult

    return density#jnp.where(mask, Visc_mult*density, density ) # the visc mult now affects inside the void instead of outside


def void_borders(indicator, n = 170):
    x = jnp.linspace(0,params["L_x"],n)
    y = jnp.linspace(-params["L_y"]/2.,params["L_y"]/2.,n)
    x,y = jnp.meshgrid(x,y)
    Void = indicator(x,y)
    Border = jnp.zeros((n,n),dtype=int)

    center = Void[1:-1, 1:-1] == 1 # interior mask where Void == 1
 
    neighbor_zero = ( # check 4-neighbor zeros
        (Void[:-2, 1:-1] == 0) |   # up
        (Void[2:, 1:-1] == 0)  |   # down
        (Void[1:-1, :-2] == 0) |   # left
        (Void[1:-1, 2:] == 0)      # right
    )

    mask = center & neighbor_zero # should give the same points as the loop; but much faster

    Border = Border.at[1:-1, 1:-1].set(mask.astype(Border.dtype))
    '''for i in range(n):
        for j in range(n):
            if Void[i,j]==1 and i not in (0, n-1) and j not in (0,n-1):
                if Void[i-1,j] == 0 or Void[i+1,j] == 0 or Void[i,j-1] == 0 or Void[i, j-1] == 0:
                    Border = Border.at[i,j].set(1)'''
    
    #print(Border)
    #plt.imshow(Border)
    #plt.colorbar()
    #plt.show()

    return Border

def Void_potential(indicator, n=170):
    Border = void_borders(indicator, n)
    # now each point in the array set to 1 becomes a source of the force.
    border_points = jnp.argwhere(Border == 1) # returns an nx2 array of integers - each row is an index
    x_inds = border_points[:,0]
    y_inds = border_points[:,1]
    x_pos = x_inds * params["L_x"] / float(n)
    y_pos = y_inds * params["L_y"] / float(n) - (params["L_y"]/2.)
    x_pos = x_pos[:,None] # make 2D
    y_pos = y_pos[:,None]
    
    
    def potential(x,y):
        x_copy = np.empty((n,len(x)),dtype=np.float64)
        y_copy = np.empty((n,len(y)),dtype=np.float64)
        x_copy[:,:] = x[None,:]
        y_copy[:,:] = y[None,:] # fill the columns with the single value; vector fills the rows
        F_x = params["Barrier_Potential"] * np.sum((x - x_pos)/(np.sqrt((x - x_pos)**2 + (y - y_pos)**2))**3, axis = 0) # sum the columns - results in a vector as
        F_y = params["Barrier_Potential"] * np.sum((y - y_pos)/(np.sqrt((x - x_pos)**2 + (y - y_pos)**2))**3, axis = 0)

        return F_x, F_y 
        

    return potential



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

    Z = viscocity_gradient(X,Y,ind_func,0.8)
    plt.imshow(Z)
    plt.colorbar()
    plt.show()

    #void_borders(ind_func)
    Void_potential(ind_func)


    if False: # plot beta dist
        rng1 = default_rng(1)
        plt.hist(rng1.beta(a=0.5, b=0.5, size=1000))
        plt.show()