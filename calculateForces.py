import numpy as np
from defineParameters import params
from calculateCurrent import calculate_current
#from time import sleep # for debugging
from viscosity_field import make_globular_indicator


ind_func = make_globular_indicator() # get a function to check viscocity state; index function of the void set

r = params["elec_transfer_radius"]

# Global simulation state
tindex = 0
V = params["V"]
I_saved, t_saved, I_last = [], [], 0
rand_dirs_global_x, rand_dirs_global_y = None, None

integrable_calcualte_forces = lambda t,x: calculate_forces(t,x,params) # this function only takes t and x so it can be used by an integrator

def calculate_forces(t, states, params):
    """
    Python translation of MATLAB calculateForces.m
    Computes the time derivative dx/dt for all particle states.
    """

    global tindex, V, I_saved, t_saved, I_last, Volt
    global rand_dirs_global_x, rand_dirs_global_y

    if not params['is_Voltage_constant']:
        Volt = params["V_func"](t)
    else:
        Volt = V

    n = params["n"]
    eta = params["eta"]

    # Initialize global direction vectors if empty
    if rand_dirs_global_x is None:
        rand_dirs_global_x = 2 * np.random.randint(0, 2, n) - 1 # random numbers from 
        rand_dirs_global_y = 2 * np.random.randint(0, 2, n) - 1

    # ------------------------
    # Unpack state variables
    # ------------------------
    x_p = states[0:n]
    x_v = states[n:(2*n)]
    y_p = states[(2*n):(3*n)]
    y_v = states[(3*n):(4*n)]
    T = states[4*n] # temperature

    # ------------------------
    # Compute current at intervals
    # ------------------------
    if tindex < len(params["tspan"]) and t > params["tspan"][tindex]:
        rand_dirs_global_x = 2 * np.random.randint(0, 2, n) - 1
        rand_dirs_global_y = 2 * np.random.randint(0, 2, n) - 1

        if tindex % 2 == 0:
            if Volt == 0.: I_last = 0. # this should save a lot of time
            else:
                I_last = calculate_current(
                    x_p, y_p, params["L_x"],params["L_y"], Volt,
                    params["lambda"], params["Rt"], T, params['num_e']#params["steps"], params["num_e"]
                )

        tindex += 1
        if tindex > 1000 and Volt != 0 and params["is_Voltage_constant"]: # after 1000 interations set voltage to 0; only do if the voltage is constant
            Volt = 0
            print("Voltage set to zero after 1000 iterations")

        print(f"Time index {tindex} / {len(params['tspan'])}")
        print(f"Computed I at t = {tindex}, I = {I_last:.3e}")

        I_saved.append(I_last)
        t_saved.append(t)

    # ------------------------
    # Forces
    # ------------------------
    Fa_x = applied_force(n, x_p, y_p, params["alpha"], Volt, params["L_x"],t)
    Fd_x, Fd_y = drag_force(n, x_p, y_p, x_v, y_v, params["eta"], params["Cd"])
    FI_x = interfacial_force(n, x_p, y_p, params["wI"], params["RI"], params["L_x"])
    Fp_x, Fp_y = pinning_force(
        n, x_p, y_p, params["w_pin"], params["x_pin"], params["y_pin"], params["R_pin"], params["L_x"]
    )
    Ft_x, Ft_y = temperature_fluctuations(
        n, eta, params["T_coeff"], T, rand_dirs_global_x, rand_dirs_global_y
    )
    
    Fr_x, Fr_y = residual_force(n, x_p, y_p, params["L_x"], params["L_y"])

    Fc_x = np.zeros(n) # initialize the contact forces
    Fc_y = np.zeros(n)

    # If the particle has reached the end, set the velocity to zero
    fin_array = finishing_array(x_p, params["L_x"], params["fin"])

    # Total forces
    forces_x = Fa_x + Fd_x + FI_x + Fp_x + Ft_x + Fr_x + Fc_x # resultant force in the x direction
    forces_y = 0    + Fd_y + 0    + Fp_y + Ft_y + Fr_y + Fc_y # resultant force in the y direction
    
    # ------------------------
    # Solve for dx/dt
    # ------------------------
    # evolution in x direction
    dxdt = np.zeros(4 * n + 1) # initialize the array of states vector derivatives (x here is not the x-direction, but the entire state vector)
    dxdt[0:n] = x_v * fin_array
    dxdt[n:(2*n)] = (forces_x / eta) * fin_array
    #evolution in y direction
    dxdt[(2*n):(3*n)] = y_v * fin_array
    dxdt[(3*n):(4*n)] = (forces_y / eta) * fin_array
    dxdt[4 * n] = (params["CT"] * params["Q"]) - params["k"] * (T - params["T_0"]) #temperature evolution
    #print(dxdt[0])
    #sleep(0.5)
    return dxdt


# -----------------------------------------------------
# Subfunctions
# -----------------------------------------------------

def distances(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    d = np.sqrt(dx ** 2 + dy ** 2)
    return d, dx, dy

def distance(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    d = np.sqrt(dx ** 2 + dy ** 2)
    return d


def applied_force(n, x_p, y_p, alpha, V, Lx, t): # (From Electric Field)
    # Initialize force to zero
    Fa_x = np.zeros(n)

    # Logical index of particles inside domain
    inside = (x_p < Lx)
    inside_y = y_p[inside]

    # Apply scaling only to particles within [0,Lx]
    # only apply force to particle inside of a void
    Fa_x[inside][ind_func(inside,inside_y) == 0] = alpha[inside][ind_func(inside,inside_y) == 0] * Volt / ((0.5) * Lx)

    '''
    This section deals in adding back a force to those particles inside of the void if there 
    - there a electromagnetic edge effects that we will ignore for now (i.e. purterbations to the field at the edge of a conductor)
    We will set some radius (in params) for which there must be a particle *behind* the current particle in order for the field to proagate to the current particle
    If this condition is met, we will add the force back.
    '''

    n = len(Fa_x[inside][ind_func(inside,inside_y) == 1]) #number of particles inside the voids
    for i in range(n):

        dist_vec = distance(x_p[inside][ind_func(inside,inside_y) == 1][i], x_p[x_p[inside][ind_func(inside,inside_y) == 1][i] > x_p], y_p[inside][ind_func(inside,inside_y) == 1][i], y_p[x_p[inside][ind_func(inside,inside_y) == 1][i] > x_p]) # only comparing with those behind
        dist_vec = dist_vec[dist_vec > 0] # remove the current particle from the list

        if (dist_vec < r).any(): # There is a particle behind the current particle - within a small enough range
            Fa_x[inside][ind_func(inside,inside_y) == 1][i] = alpha[inside][ind_func(inside,inside_y) == 0][i] * Volt / ((0.5) * Lx)


    return Fa_x


def drag_force(n, x, y, x_v, y_v, eta, Cd):

    eta_prime = np.array([ eta if ind_func(x[i],y[i]) == 1 else params['viscosity_multiplier']*eta for i in range(len(x_v))])

    Fd_x = -eta_prime * Cd * x_v
    Fd_y = -eta_prime * Cd * y_v

    return Fd_x, Fd_y


def interfacial_force(n, x_p, y_p, wI, RI, Lx):
    """
    Compute interfacial force
    """
    # Initialize force array
    FI_x = np.zeros(n)

    # Logical index of particles inside [0, Lx]
    inside = (x_p > 0) & (x_p < Lx)

    # Apply force only for those particles satisfying the logical index
    term1 = x_p[inside] * np.exp(-(x_p[inside]**2) / RI**2)
    term2 = (x_p[inside] - Lx) * np.exp(-((x_p[inside] - Lx)**2) / RI**2)

    FI_x[inside] = -(2 * wI / RI**2) * (term1 + term2)

    # Debug print (optional)
    #print(FI_x[0])
    #sleep(0.5)

    return FI_x



def pinning_force(n, x_p, y_p, w_pin, x_pin, y_pin, R_pin, Lx):
    """
    Vectorized pinning-force computation.
    Parameters follow MATLAB mapping exactly.
    """
    
    # Initialize force to zero
    Fp_x = np.zeros(n)
    Fp_y = np.zeros(n)

    # Loop per pinning site (small number, so OK)
    for i in range(len(w_pin)):
        # vector: particle-to-pin deltas
        dx = x_pin[i] - x_p
        dy = y_pin[i] - y_p

        # distances
        d = np.sqrt(dx*dx + dy*dy)

        # Avoid division by zero (particles exactly at pin location)
        #d_safe = np.where(d == 0, 1e-30, d)

        # Gaussian force magnitude
        F = (2 * w_pin[i] / (R_pin[i]**2)) * np.exp(-(d**2)/(R_pin[i]**2))

        # Direction components (unit vector * magn.)
        Fp_x += F * dx
        Fp_y += F * dy

    return Fp_x, Fp_y



def temperature_fluctuations(n, eta, T_coeff, T, rand_x, rand_y):
    noise_scale = np.sqrt(eta * T_coeff * T)
    Ft_x = rand_x * noise_scale
    Ft_y = rand_y * noise_scale
    return Ft_x, Ft_y

# # ------------------------
# # TODO: Residual Force
# # Fast local density gradient approximation and then force based on that
# # ------------------------

def residual_force(n, x_p, y_p, Lx, Ly):
    return 0., 0. # remove the residual force - instead we use the drag force to contain 
    cell_size = 5.0 #Size of each grid cell
    w_resid = params["w_resid"] #Strength of residual force
    Nx = max(1, int(np.ceil(Lx / cell_size)))
    Ny = max(1, int(np.ceil(Ly / cell_size)))

    # Shift y to [0, Ly] if needed (keeps bins consistent)
    y_shifted = y_p - np.min(y_p) # now spans ~[0, Ly]
    # ---- Bin indices (1-based) ----
    ix = np.clip(np.floor(x_p / cell_size).astype(int), 0, Nx-1)
    iy = np.clip(np.floor(y_shifted / cell_size).astype(int), 0, Ny-1)

    count = np.zeros((Ny, Nx))
    np.add.at(count, (iy, ix), 1) # should be the analog of the accumarray MATLAB function

    lin_idx = iy * Nx + ix
    count.flat[lin_idx] = np.maximum(count.flat[lin_idx] - 1, 0)

    '''def shift(arr, dx, dy):
        return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)'''

    # Left neighbor: shift right, duplicate leftmost column
    count_left = np.hstack((count[:, [0]], count[:, :-1]))

    # Right neighbor: shift left, duplicate rightmost column
    count_right = np.hstack((count[:, 1:], count[:, [-1]]))

    # Up neighbor: shift down, duplicate last row
    count_up = np.vstack((count[1:], count[[-1], :]))

    # Down neighbor: shift up, duplicate first row
    count_down = np.vstack((count[[0], :], count[:-1]))
    """count_left = np.pad(count[:, :Nx - 1], ((0, 0), (1, 0)), mode="edge")
    count_right = np.pad(count[:, 1:], ((0, 0), (0, 1)), mode="edge")
    count_up = np.pad(count[1:, :], ((0, 1), (0, 0)), mode="edge")
    count_down = np.pad(count[:-1, :], ((1, 0), (0, 0)), mode="edge")"""

    rho_left  = count_left.flat[lin_idx]
    rho_right = count_right.flat[lin_idx]
    rho_up    = count_up.flat[lin_idx]
    rho_down  = count_down.flat[lin_idx]

    Fr_x = w_resid * (rho_left - rho_right)
    Fr_y = w_resid * (rho_down - rho_up)
    return Fr_x, Fr_y


"""def finishing_array(x_p, L, fin):
    if fin == 1:
        return np.where(x_p >= 2 * L - 5e-6, 0, 1)
    else:
        return np.ones_like(x_p)"""

def finishing_array(x_p, L, fin):
    """
    Check if particles have reached the end.
    Returns a boolean array (or True if fin == 0).
    """
    if fin == 1:
        return x_p < (2 * L - 5e-6)
    else:
        return True