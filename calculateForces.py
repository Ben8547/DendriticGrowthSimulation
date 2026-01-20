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
        '''if tindex > 1000 and Volt != 0 and params["is_Voltage_constant"]: # after 1000 interations set voltage to 0; only do if the voltage is constant
            Volt = 0
            print("Voltage set to zero after 1000 iterations")''' # for relaxation models

        print(f"Time index {tindex} / {len(params['tspan'])}")
        print(f"Computed I at t = {tindex}, I = {I_last:.3e}")

        I_saved.append(I_last)
        t_saved.append(t)

    # ------------------------
    # Forces
    # ------------------------
    Fa_x = applied_force(n, x_p, y_p, params["alpha"], Volt, params["L_x"],t)
    F_vdWx, F_vdWy = vdW_Force_AND_Dipole_Force(x_p,y_p)
    Fd_x, Fd_y = drag_force(x_p, y_p, x_v, y_v, params["eta"], params["Cd"])
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
    forces_x = Fa_x + Fd_x + FI_x + Fp_x + Ft_x + Fr_x + Fc_x + F_vdWx # resultant force in the x direction
    forces_y = 0.   + Fd_y + 0.   + Fp_y + Ft_y + Fr_y + Fc_y + F_vdWy  # resultant force in the y direction
    
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

    # Logical index of particles inside domain - prevents forward motion of particles outside of the bounded region
    inside = (x_p < Lx)

    if False: # Sam's orginal code; debug gate
        Fa_x[inside] = alpha[inside] * Volt / ((0.5) * Lx)

    if True: # debug gate
        # Apply scaling only to particles within [0,Lx]
        # only apply force to particle inside of a void
        out_void = (ind_func(x_p,y_p) == 0) # logcal index: particle is a void region <=> insulated from the electric field
        Fa_x[inside & (out_void)] = alpha[inside & (out_void)] * Volt / ((0.5) * Lx) # assign force as normal if not in a void region and inside the domain

        '''
        This section deals in adding back a force to those particles inside of the void if there 
        - there a electromagnetic edge effects that we will ignore for now (i.e. purterbations to the field at the edge of a conductor)
        We will set some radius (in params) for which there must be a particle *behind* the current particle in order for the field to proagate to the current particle
        If this condition is met, we will add the force back.
        '''
        in_void = (ind_func(x_p,y_p) == 1) # logical index of particles in a void
        n = len(Fa_x[inside & in_void]) #number of particles inside the voids

        x_in_void = x_p[inside & in_void]
        y_in_void = y_p[inside & in_void]

        #good_angle = np.array([ np.abs(np.arctan((x_p[i]-x_p)/(y_p[i]-y_p))) for i in range(n) ]) # not vectorized unfortunately. Probably slows the programs considerably
        #good_angle = (good_angle < params["angle"]/2.)[0]
        # the above commented code was meant to restrict the angle of electric field transmission, but on further thought this is not physical.

        for i in range(n):

            dist_vec = distance(x_in_void[i], x_p[x_in_void[i] > x_p], y_in_void[i], y_p[x_in_void[i] > x_p]) # only comparing with those behind
            #dist_vec = dist_vec[dist_vec > 1e-14] # remove the current particle from the list; don't need this anymore since we only compare to those strictly behind

            if (dist_vec < r).any(): # There is a particle behind the current particle - within a small enough range
                Fa_x[inside & in_void][i] = alpha[inside & in_void][i] * Volt / ((0.5) * Lx) # if the electric field can propagate, 


    return Fa_x


def drag_force(x, y, x_v, y_v, eta, Cd):

    eta_prime = np.array([ eta if ind_func(x[i],y[i]) == 1 else eta*params['viscosity_multiplier'] for i in range(len(x_v))])

    Fd_x = -eta_prime * Cd * x_v
    Fd_y = -eta_prime * Cd * y_v

    return Fd_x, Fd_y

#from scipy.spatial.distance import pdist, squareform # pdist takes advantage of the symmetry of distance calculations

def compute_dist_matrix(x, y): # compute a matrix of distances - sees use below
    # Reshape to column vectors to enable broadcasting
    # (N, 1) - (1, N) results in an (N, N) matrix of differences
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    
    # Apply the Euclidean distance formula: sqrt(dx^2 + dy^2)
    dist_matrix = np.sqrt(dx**2 + dy**2)
    
    return dist_matrix, dx, dy


def vdW_Force_AND_Dipole_Force(x,y, R=params["Average_particle_Radius"], A=params["Hamaker Constant"], Ex=params["V"]/params["L_x"], Ey=0., eps_r=1.):
    # since both use a distance matrix, combine them for efficiency - only compute matrix once
    '''
    Docstring for vdW_Force

    This computes the van der Waals attractive force between each particle.
    
    :param x: array if x positions
    :param y: array of y positiosn
    :param R: Radius of the silver nanoparticles (meters).
    :param A: Hamaker constant (Joules).
    '''

    distance_matrix, dx, dy = compute_dist_matrix(x,y) # each row is a particles and the columns are its distance to each other particle. The diagonal is 0. The matrix is symmetric.
    mask = distance_matrix > 2.001 * R  # Only compute for particles not in contact
    f_mag = np.zeros_like(distance_matrix)
    r = distance_matrix[mask]
    r2, R2 = r**2, R**2
    
    # Derivative components of the Hamaker potential
    term1 = (4. * R2 * r) / (r2 - 4. * R2)**2.
    term2 = (4. * R2) / (r**3)
    term3 = (8. * R2) / (r * (r2 - 4. * R2))
    
    # Scalar force magnitude (positive value indicates attraction in this logic)
    f_mag[mask] = (A / 6.) * (term1 + term2 - term3)
    
    # 5. Project Magnitudes onto X and Y Axes
    # To get the force ON particle i BY particle j:
    # Since it is attractive, the force points from i toward j.
    # The vector from i to j is (x_j - x_i), which is -dx[i, j].
    fx_matrix = np.zeros_like(distance_matrix)
    fy_matrix = np.zeros_like(distance_matrix)
    fx_matrix[mask] = f_mag[mask] * (-dx[mask] / distance_matrix[mask])
    fy_matrix[mask] = f_mag[mask] * (-dy[mask] / distance_matrix[mask])
    
    # 6. Sum across rows to get net force on each particle i
    fx_net = np.sum(fx_matrix, axis=1)
    fy_net = np.sum(fy_matrix, axis=1)

    '''Now compute the Dipole Forces'''

    N = len(x)
    eps_0 = params['eps_0']
    eps_m = eps_r * eps_0
    
    # 1. Induced Dipole Moment (p = alpha * E)
    # For silver (metal), the Clausius-Mossotti factor is ~1
    p_mag_factor = 4. * np.pi * eps_m * (R**3)
    px, py = p_mag_factor * Ex, p_mag_factor * Ey
    p_dot_p = px**2. + py**2.
    
    # 2. Distance and Displacement
    # r_vec is the vector from i to j: (x_j - x_i)
    # Note: we use (x[j] - x[i]) which is -dx from our previous convention
    
    # 4. Unit Vectors (r_hat)
    ux = np.zeros_like(distance_matrix)
    uy = np.zeros_like(distance_matrix)
    ux[mask] = dx[mask] / distance_matrix[mask]
    uy[mask] = dy[mask] / distance_matrix[mask]
    
    # 5. Dot Products (p_dot_r_hat)
    p_dot_u = px * ux + py * uy
    
    # 6. Compute Force Components using the vector formula
    # Force on i due to j
    prefactor = np.zeros_like(distance_matrix)
    prefactor[mask] = 3. / (4. * np.pi * eps_m * distance_matrix[mask]**4)
    
    # Term-by-term calculation for F_x and F_y
    # F = prefactor * [ (p_dot_u)*px + (p_dot_u)*px + (p_dot_p)*ux - 5*(p_dot_u)*(p_dot_u)*ux ]
    fx_matrix = prefactor * (2. * p_dot_u * px + p_dot_p * ux - 5. * (p_dot_u**2) * ux)
    fy_matrix = prefactor * (2. * p_dot_u * py + p_dot_p * uy - 5. * (p_dot_u**2) * uy)
    
    # 7. Sum to get net force on each particle
    fx_net_di = np.sum(fx_matrix, axis=1)
    fy_net_di = np.sum(fy_matrix, axis=1)
    
    return fx_net+fx_net_di, fy_net+fy_net_di

def interfacial_force(n, x_p, y_p, wI, RI, Lx):
    return 0. # I'm setting this force to zero becuase I can't think of physical force that would scale strictly off of distance like this 
    """
    Compute interfacial force  - direct translation from Sam's matlab files 
    """
    # Initialize force array
    FI_x = np.zeros(n)

    # Logical index of particles inside [0, Lx]
    inside = (x_p > 0) & (x_p < Lx)

    # Apply force only for those particles satisfying the logical index
    term1 = x_p[inside] * np.exp(-(x_p[inside]**2) / RI**2)
    term2 = (x_p[inside] - Lx) * np.exp(-((x_p[inside] - Lx)**2) / RI**2)

    FI_x[inside] = -(2 * wI / RI**2) * (term1 + term2)

    # Debug
    #print(FI_x[0])
    #sleep(0.5)

    return FI_x



def pinning_force(n, x_p, y_p, w_pin, x_pin, y_pin, R_pin, Lx):
    return 0., 0. # for simplicity (temporary)
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