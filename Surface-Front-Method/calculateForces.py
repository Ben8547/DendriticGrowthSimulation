from numpy import sqrt, column_stack, zeros, abs, append, ones, zeros_like
#from numpy import max as npmax
from numpy.random import randint
from defineParameters import params
#from time import sleep # for debugging
from viscosity_field import make_globular_indicator, viscocity_gradient
from scipy.spatial import cKDTree # makes spatial searches much faster


ind_func = make_globular_indicator() # get a function to check viscocity state; index function of the void set

r = params["elec_transfer_radius"]

# Global simulation state
tindex = 0
V = params["V"]

alpha = params["alpha"]

def calculate_forces(t, states, params, Visc):
    """
    Python translation of MATLAB calculateForces.m
    Computes the time derivative dx/dt for all particle states.
    """

    if not params['is_Voltage_constant']:
        Volt = params["V_func"](t)
    else:
        Volt = V

    n = params["n"]
    eta = params["eta"]

    # ------------------------
    # Unpack state variables
    # ------------------------
    x_p = states[0:n]
    x_v = states[n:(2*n)]
    y_p = states[(2*n):(3*n)]
    y_v = states[(3*n):(4*n)]
    T = states[4*n] # temperature

    coords = column_stack((x_p, y_p))

    #tree_data = (inside,is_void,out_void)

    # ------------------------
    # Forces
    # ------------------------
    Fa_x = applied_force(coords, params["alpha"], Volt, params["L_x"])
    Fd_x, Fd_y = drag_force(x_p, y_p, x_v, y_v, params["Cd"], Visc)
    F_ix = biharmonic_smoothing(x_p)
    # If the particle has reached the end, set the velocity to zero
    fin_array = finishing_array(x_p, params["L_x"], params["fin"])

    # Total forces
    forces_x = Fa_x + Fd_x + F_ix # resultant force in the x direction
    forces_y = 0.   + Fd_y + 0. # resultant force in the y direction
    
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
    dxdt[4 * n] = (params["CT"] * params["Q"]) - params["k"] * (T - params["T_0"]) #temperature evolution; should make this local at some point and effect the density
    return dxdt


# -----------------------------------------------------
# Subfunctions
# -----------------------------------------------------

def compute_curvature(x,dy):
    x_prev = x[:-2]  # x0 to x_{n-2}
    x_curr = x[1:-1] # x1 to x_{n-1}
    x_next = x[2:]   # x2 to x_n

    numerator = abs(x_prev - 2 * x_curr + x_next)
    denominator = dy**2 * ( 1 + ( (x_next - x_prev) / (2*dy))**2 )**(1.5)

    kappa = numerator / denominator
    # add the leading and trailing ones
    kappa = append(ones(1),kappa)
    kappa = append(kappa,ones(1))

    return kappa # curvature of a parabala passing through three points - we use this to modulate the 

def applied_force(coords, alpha, Volt, Lx): # (From Electric Field)
    # Initialize force to zero
    Fa_x = zeros(params['n']) # intialize the force vector
    x_p = coords[:,0]

    inside = (x_p < Lx)
    curvature = compute_curvature(x_p,params['dy'])
    Fa_x[inside] = alpha[inside] * Volt * curvature[inside] / ((0.5) * Lx) # set forces inside the device
    return Fa_x * 1e2

def drag_force(x, y, x_v, y_v, Cd,Visc_mult):

    eta_prime = viscocity_gradient(x,y,ind_func,Visc_mult)

    Fd_x = -eta_prime * Cd * x_v
    Fd_y = -eta_prime * Cd * y_v

    return Fd_x, Fd_y

def interaction_force(x,y):
    # simple spring force between adjacent particles; we ignore the y-force and only return the x-force, y-force can be added later.
    k = params['spring']
    x_left = x[:-2]
    x_center = x[1:-1]
    x_right = x[2:]

    F_spring = -k*(2*x_center - x_left - x_right)
    F_spring = append(-k*(x[0] - x[1]), F_spring)
    return append(F_spring, -k*(x[-1] - x[-2]))



def temperature_fluctuations(n, eta, T_coeff, T, rand_x, rand_y):
    noise_scale =  sqrt(eta * T_coeff * T)
    Ft_x = rand_x * noise_scale
    Ft_y = rand_y * noise_scale
    return Ft_x/60., Ft_y/60. # scaled so that these forces are not dominant - they should be small relative to the other forces

def biharmonic_smoothing(x, k4=params['spring']):

    F = zeros_like(x)

    F[2:-2] = -k4 * (
        x[:-4]
        - 4*x[1:-3]
        + 6*x[2:-2]
        - 4*x[3:-1]
        + x[4:]
    )

    return F


def finishing_array(x_p, L, fin):
    """
    Check if particles have reached the end.
    Returns a boolean array (or True if fin == 0).
    """
    if fin == 1:
        return x_p < (2 * L - 5e-6)
    else:
        return True