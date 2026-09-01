from numpy import sqrt, column_stack, zeros, where, array, add, maximum, divide, copy, zeros_like, pi, any
import numpy as np
#from numpy import max as npmax
from numpy.random import randint
from defineParameters import params
#from time import sleep # for debugging
from scipy.spatial import cKDTree # makes spatial searches much faster
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator


''' We need a function that creates a potential preventing entry into the regions - as you enter the region - it needs to act like a hole in the simulation domain - a wall
We could use some repelling force like 1/r or 1/r^2 but this might make the ODE stiff; we would also need to define the boundary of the region which is difficult. We could also we a very strong force field within the void, pointing outward but this would also
potentially cause particles to be shot out of the void which is not realistic, we only want to prevent entry. Another idea is to create some point at the center of the void and have an exponentially decaying force field eminate from that point.
Then we likely would not have the shooting motion, but we would not have such a clear barrier to entry.'''
#void_barrier_force = Void_potential(ind_func)

r = params["elec_transfer_radius"]

# Global simulation state
tindex = 0
V = params["V"]
rand_dirs_global_x, rand_dirs_global_y = None, None

alpha = params["alpha"]

def calculate_forces(t, states, params, wpa, ind_func, V_func = params["V_func"], resist=1.):
    """
    Python translation of MATLAB calculateForces.m
    Computes the time derivative dx/dt for all particle states.
    """

    global tindex, V, Volt
    global rand_dirs_global_x, rand_dirs_global_y # these being global variables is a remnant of the matlab code. I don't think they are needed, but I don't want to mess anything up my removing them.

    if not params['is_Voltage_constant']:
        Volt = V_func(t)
    else:
        Volt = V

    n = params["n"]
    eta = params["eta"]

    # Initialize global direction vectors if empty
    if rand_dirs_global_x is None:
        rand_dirs_global_x = 2 * randint(0, 2, n) - 1 # random numbers from 
        rand_dirs_global_y = 2 * randint(0, 2, n) - 1

    # ------------------------
    # Unpack state variables
    # ------------------------
    x_p = states[0:n]
    x_v = states[n:(2*n)]
    y_p = states[(2*n):(3*n)]
    y_v = states[(3*n):(4*n)]
    T = states[4*n] # temperature

    #----------------
    #Search Tree
    #----------------

    # Turn the points into a KDTree for easy spatial search
    coords = column_stack((x_p, y_p))

    #Tree = cKDTree(coords) # search tree - should speed up later distance dependant force computations

    #tree_data = (inside,is_void,out_void)

    # ------------------------
    # Forces
    # ------------------------
    Fa_x = applied_force(n, coords, params["alpha"], params["L_x"], Volt, use_chain=False)
    #F_vdWx, F_vdWy = vdW_Force_AND_Dipole_Force(coords,Tree,A=Hamaker)
    Fd_x, Fd_y = drag_force(x_p, y_p, x_v, y_v, params["Cd"], ind_func)
    Ft_x, Ft_y = temperature_fluctuations(
        n, eta, params["T_coeff"], T, rand_dirs_global_x, rand_dirs_global_y
    )
    F_barrier_x, F_barrier_y = void_potential(x_p,y_p)
    F_pin_x, F_pin_y = pinning_force(x_p,y_p,wpa)
    F_resid_x, F_resid_y = residual_force_convolved(x_p,y_p,params["L_x"],params['L_y'])

    if True: # switch the L-ennard-Jones potential on or off
        Fc_x = zeros(n) # initialize the contact forces
        Fc_y = zeros(n)
    else:
        Fc_x, Fc_y = vdW_Force_AND_Dipole_Force(coords,Tree)

    # If the particle has reached the end, set the velocity to zero
    fin_array = finishing_array(x_p, params["L_x"], params["fin"])

    # Total forces
    forces_x = Fa_x + Fd_x + Ft_x + Fc_x + F_barrier_x + F_pin_x + F_resid_x# + F_vdWx # resultant force in the x direction
    forces_y = 0.   + Fd_y + Ft_y + Fc_y + F_barrier_y + F_pin_y + F_resid_y# + F_vdWy  # resultant force in the y direction
    
    # ------------------------
    # Solve for dx/dt
    # ------------------------
    # evolution in x direction
    dxdt = zeros(4 * n + 1) # initialize the array of states vector derivatives (x here is not the x-direction, but the entire state vector)
    dxdt[0:n] = x_v * fin_array
    dxdt[n:(2*n)] = (forces_x / eta) * fin_array
    #evolution in y direction
    dxdt[(2*n):(3*n)] = y_v * fin_array
    dxdt[(3*n):(4*n)] = (forces_y / eta) * fin_array
    dxdt[4 * n] = (params["CT"] * V**2 / resist ) - params["k"] * (T - params["T_0"]) #temperature evolution
    return dxdt


# -----------------------------------------------------
# Subfunctions
# -----------------------------------------------------

def density_modulated_force(x, y, tree):

    n = len(x)
    coords = np.column_stack((x, y))

    r_density = 10. * params["Average_particle_Radius"]
    gamma = params["density_strength"]

    # ----------------------------
    # Compute local densities
    # ----------------------------
    density = np.zeros(n)

    for i in range(n):
        neighbors = tree.query_ball_point(coords[i], r_density)
        density[i] = len(neighbors)

    # ----------------------------
    # Compute density gradient
    # ----------------------------
    density_grad = np.zeros(n)

    # backward difference (since ordering preserved in y)
    density_grad[1:] = density[1:] - density[:-1]
    density_grad[0] = density_grad[1]

    Force_modifier = (1 - gamma * density_grad)

    # Clip to prevent negative runaway
    Force_modifier = np.clip(Force_modifier, 0, None)

    return Force_modifier

def applied_force(n, coords, alpha, Lx, Volt, use_chain = True): # (From Electric Field)
    # Initialize force to zero

    Fa_x = zeros(n)

    x_p = coords[:,0]
    y_p = coords[:,1]

    modifier = 1.#np.maximum(np.ones_like(x_p), density_modulated_force(x_p,y_p,kdTree)) #1.

    # Logical index of particles inside domain - prevents forward motion of particles outside of the bounded region
    inside = (x_p < Lx)
    #is_void = (ind_func(x_p, y_p) == 1)[0]

    if not use_chain: # Sam's orginal code; debug gate
        Fa_x[inside] = alpha[inside] * Volt / ((0.5) * Lx)
        return Fa_x * 1e15
    
    else:
        raise ProcessLookupError("Depreciated")

def drag_force(x, y, x_v, y_v, Cd, ind_func):

    p = params['eta']

    eta_prime = p#viscocity_gradient(x,y,ind_func,Visc_mult)

    Fd_x = -eta_prime * Cd * x_v
    Fd_y = -eta_prime * Cd * y_v

    mask = ind_func(x,y)[0] # 1 if inside the void, 0 if not; drag is less inside of the voids
    Fd_x[mask] = Fd_x[mask] / (p/2.)
    Fd_y[mask] = Fd_y[mask] / (p/2.)

    return Fd_x, Fd_y

def void_potential(x,y):
    #mask = ind_func(x,y) # 1 if inside the void, 0 if not
    #F_x, F_y = void_barrier_force(x,y)
    #return F_x, F_y
    return 0.,0.


def vdW_Force_AND_Dipole_Force(coords, tree, R=params["Average_particle_Radius"], A=1.):
    # The Hamaker potential has been extrordinarily problemeatic in the simualtions since it has a singularity at d = 2R.
    # I'm, switching to a Lenard-Jones potential because this won't have the same issues and we can tune the minimal energy location instead of tuning the Hamaker force magnetude.
    # It feels cheep; but the force not being defined at 2R has caused so many problems that I haven't been able to satisfacorially fix that I need to to change this force.
    x = coords[:,0]
    y = coords[:,1]

    # Define cutoffs (forces are negligible beyond these)
    # vdW dies at ~1/r^7, Dipole at ~1/r^4
    cutoff = 10. * R

    # Find all pairs within cutoff. Returns (i, j) indices of the coords matrix
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray') # returns nx2 matrix where n is the number of particles pairs within the cut off distance. Each row of the matrix contains a pair of indicies within the distance. Note that this returns particle index, i.e. the index of the row in the coords matrix


    i = pairs[:, 0] # list of the first coordinate in the pairs
    j = pairs[:, 1] # list of the second coordinate in the pairs

    dx = x[j] - x[i] # pair x-distances; dx points from i to j
    dy = y[j] - y[i] # pair y-distnaces; points from i to j
    dist2 = dx**2 + dy**2 # list of pair y-distances squaresd
    dist =  sqrt(dist2) # list of pair y-distances

    min_dist = 0.8 * params["LJ - Sigma"] # Caps the force at slightly less than 0.8*2R
    dist = maximum(min_dist,dist)

    # computing the force; I've read that it is quicker to manually compute the powers since python's generic power computation will be slow for these large numbers.
    invdist = params["LJ - Sigma"]/dist
    invdist3 = invdist * invdist * invdist
    invdist6 = invdist3 * invdist3
    
    F_vdW_mag = 48.*A * invdist*invdist6 * ( (invdist6) -  (0.5) ) / params["LJ - Sigma"] #Lenard-Jones Force 
    # Unit vectors; need to project the force onto the vectro in between the two particles
    ex = dx / dist
    ey = dy / dist # since dy is signed this encodes force direction as well; points from i to j

    # Force components on particle i due to j; i.e. the force should point from particle i to particle j
    # if particle i is behind j, ex is positive, we want to x force to point in the positive direction since this will pull i towards j
    # if j is behind i then ex is negative and so is the force. This is again desired as it pulls particle x back, towards j.
    Fx_vdW = F_vdW_mag * ex # ex and ey already contain the force direciton.
    Fy_vdW = F_vdW_mag * ey 

    # initialize force vectors forces
    f_vdW_x =  zeros(len(x))
    f_vdW_y =  zeros(len(y))

    # Add force to i, subtract (Newton's 3rd) from j
    add.at(f_vdW_x, i,  Fx_vdW)
    add.at(f_vdW_x, j, -Fx_vdW) # the force on j is opposite the force on i
    add.at(f_vdW_y, i,  Fy_vdW)
    add.at(f_vdW_y, j, -Fy_vdW)
        
    return f_vdW_x, f_vdW_y


def pinning_force(x_p, y_p, w_pin=params['w_pin'], x_pin=params['x_pin'], y_pin=params['y_pin'], R_pin=params['R_pin']): # actually the defect potential, but it's fine
    # Shapes for my reference:
    # particles: (n, 1)
    # pin sites: (1, m)
    dx = x_p[:, None] - x_pin[None, :] # (n, m)
    dy = y_p[:, None] - y_pin[None, :] # (n, m)

    d2 = dx**2 + dy**2 # (n, m)

    # Expand to (1, m) for vecotization purposes
    w = w_pin[None, :]
    R2 = (R_pin**2)[None, :]
    F = (2 * w / R2) * np.exp(-d2 / R2)  # (n, m)

    # sum
    Fp_x = np.sum(F * dx, axis=1) # (n,)
    Fp_y = np.sum(F * dy, axis=1) # (n,)

    return Fp_x, Fp_y



def temperature_fluctuations(n, eta, T_coeff, T, rand_x, rand_y):
    noise_scale =  sqrt(eta * T_coeff * T)
    Ft_x = rand_x * noise_scale
    Ft_y = rand_y * noise_scale
    return Ft_x/60., Ft_y/60. # scaled so that these forces are not dominant - they should be small relative to the other forces

def residual_force_convolved(x_p,y_p,Lx,Ly,cell_size=5.0,R_resid=None,w_resid=params["w_resid"]):

    if R_resid is None:
        R_resid = 1.5 * cell_size

    # Grid
    Nx = max(2, int(np.ceil(Lx / cell_size)))
    Ny = max(2, int(np.ceil(Ly / cell_size)))

    x_edges = np.linspace(0, Lx, Nx + 1)
    y_edges = np.linspace(-Ly / 2, Ly / 2, Ny + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ------------------------
    # Occupancy field
    # ------------------------
    density, _, _ = np.histogram2d(
        y_p,
        x_p,
        bins=[y_edges, x_edges]
    )

    radius = int(np.ceil(4 * R_resid / cell_size))

    ky = np.arange(-radius, radius + 1) * cell_size
    kx = np.arange(-radius, radius + 1) * cell_size

    KX, KY = np.meshgrid(kx, ky)

    kernel = np.exp(-(KX**2 + KY**2) / (R_resid**2))

    kernel_dx = (-(2.0 / R_resid**2) * KX * kernel)
    kernel_dy = (-(2.0 / R_resid**2)* KY* kernel)

    Fx_grid = -w_resid * fftconvolve(density,kernel_dx,mode="same")
    Fy_grid = -w_resid * fftconvolve(density,kernel_dy,mode="same")
    
    interp_fx = RegularGridInterpolator(
        (y_centers, x_centers),
        Fx_grid,
        bounds_error=False,
        fill_value=0.0)
    interp_fy = RegularGridInterpolator(
        (y_centers, x_centers),
        Fy_grid,
        bounds_error=False,
        fill_value=0.0)

    pts = np.column_stack((y_p, x_p))

    Fr_x = interp_fx(pts)
    Fr_y = interp_fy(pts)

    return Fr_x, Fr_y

def residual_force_filtered(
    x_p,y_p,Lx,Ly,cell_size=5.0,R_resid=None,w_resid=params["w_resid"]):

    if R_resid is None:
        R_resid = 1.5 * cell_size

    # Grid
    Nx = max(2, int(np.ceil(Lx / cell_size)))
    Ny = max(2, int(np.ceil(Ly / cell_size)))

    x_edges = np.linspace(0, Lx, Nx + 1)
    y_edges = np.linspace(-Ly / 2, Ly / 2, Ny + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ------------------------
    # Occupancy field
    # ------------------------
    density, _, _ = np.histogram2d(
        y_p,
        x_p,
        bins=[y_edges, x_edges]
    )

    # Gaussian smoothing
    sigma = R_resid / cell_size

    smoothed = gaussian_filter(
        density,
        sigma=sigma,
        mode="constant"
    )

    dy_field, dx_field = np.gradient(smoothed, cell_size, cell_size) # Spatial gradient

    force_x_grid = 2.0 * w_resid * dx_field # Convert gradient to force
    force_y_grid = 2.0 * w_resid * dy_field

    # ------------------------
    # Interpolate to particles
    # ------------------------
    interp_fx = RegularGridInterpolator(
        (y_centers, x_centers),
        force_x_grid,
        bounds_error=False,
        fill_value=0.0,
    )

    interp_fy = RegularGridInterpolator(
        (y_centers, x_centers),
        force_y_grid,
        bounds_error=False,
        fill_value=0.0,
    )

    pts = np.column_stack((y_p, x_p))

    Fr_x = interp_fx(pts)
    Fr_y = interp_fy(pts)

    return Fr_x, Fr_y


def finishing_array(x_p, L, fin):
    """
    Check if particles have reached the end.
    Returns a boolean array (or True if fin == 0).
    """
    if fin == 1:
        return x_p < (2 * L - 5e-6)
    else:
        return True