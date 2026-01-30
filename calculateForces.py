from numpy import sqrt, column_stack, zeros, where, array, add, maximum, divide, copy, zeros_like, pi, any
from numpy import max as npmax
from numpy.random import randint
from defineParameters import params
#from time import sleep # for debugging
from viscosity_field import make_globular_indicator
from scipy.spatial import cKDTree # makes spatial searches much faster


ind_func = make_globular_indicator() # get a function to check viscocity state; index function of the void set

r = params["elec_transfer_radius"]

# Global simulation state
tindex = 0
V = params["V"]
rand_dirs_global_x, rand_dirs_global_y = None, None

integrable_calcualte_forces = lambda t,x: calculate_forces(t,x,params) # this function only takes t and x so it can be used by an integrator
alpha = params["alpha"]

def calculate_forces(t, states, params):
    """
    Python translation of MATLAB calculateForces.m
    Computes the time derivative dx/dt for all particle states.
    """

    global tindex, V, Volt
    global rand_dirs_global_x, rand_dirs_global_y # these being global variables is a remnant of the matlab code. I don't think they are needed, but I don't want to mess anything up my removing them.

    if not params['is_Voltage_constant']:
        Volt = params["V_func"](t)
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

    Tree = cKDTree(coords) # search tree - should speed up later distance dependant force computations

    # Identify indices of particles in a void
    inside = (x_p < params['L_x'])
    is_void = (ind_func(x_p, y_p) == 1)[0]
    out_void = ~is_void

    void_indices =  where(inside & is_void)[0] # converts the boolean array to an array of indicies at which the boolean array was true; find the indicies of the particles inside of the voids
    
    # query_ball_point finds all particles within radius 'r' for each void particle
    neighbors_list = Tree.query_ball_point(coords[void_indices], r) # this returns an array of lists. The list in the ith element of the array contains the indicies of the points within a distance r of the ith particle

    null_forces = [] # list of forces to set to zero (if they are positive) - structural implications

    for i, void_idx in enumerate(void_indices): # ideally we would not appeal to a loop, but given how the code has been built thus far I cannot think of a way to avoid it.
        # neighbors_list[i] contains indices of all particles within radius r
        potential_neighbors =  array(neighbors_list[i]) # neigbors of the ith particle
        
        # only care about neighbors that are strictly behind (smaller x). also exclude the particle itself (distance 0)
        xi = x_p[void_idx]
        is_behind = x_p[potential_neighbors] < xi

        if any(is_behind) == True:
            null_forces.append(void_idx) # add the index to a list of forces to set to zero (if they are positive)

    tree_data = (inside,is_void,out_void,is_behind)

    # ------------------------
    # Forces
    # ------------------------
    Fa_x = applied_force(n, coords, tree_data, params["alpha"], Volt, params["L_x"],t)
    F_vdWx, F_vdWy = vdW_Force_AND_Dipole_Force(coords,Tree)
    Fd_x, Fd_y = drag_force(x_p, y_p, x_v, y_v, params["eta"], params["Cd"])
    FI_x = interfacial_force(n, x_p, y_p, params["wI"], params["RI"], params["L_x"])
    Fp_x, Fp_y = pinning_force(
        n, x_p, y_p, params["w_pin"], params["x_pin"], params["y_pin"], params["R_pin"], params["L_x"]
    )
    Ft_x, Ft_y = temperature_fluctuations(
        n, eta, params["T_coeff"], T, rand_dirs_global_x, rand_dirs_global_y
    )
    
    Fr_x, Fr_y = residual_force(n, x_p, y_p, params["L_x"], params["L_y"])

    Fc_x = zeros(n) # initialize the contact forces
    Fc_y = zeros(n)

    # If the particle has reached the end, set the velocity to zero
    fin_array = finishing_array(x_p, params["L_x"], params["fin"])

    # Total forces
    forces_x = Fa_x + Fd_x + FI_x + Fp_x + Ft_x + Fr_x + Fc_x + F_vdWx # resultant force in the x direction
    forces_y = 0.   + Fd_y + 0.   + Fp_y + Ft_y + Fr_y + Fc_y + F_vdWy  # resultant force in the y direction

    forces_x[null_forces] = where(forces_x[null_forces] > 0, 0. ,forces_x[null_forces]) # set the force to zero if the particle is unsupported and the force is positive
    x_v[null_forces] = where(x_v[null_forces] > 0,0.,x_v[null_forces]) # same with the velocities; I beleive this is vectorized
    # leave the y movement unaffected - it sees little purtubation as is and would have less bearing anyways

    
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
    dxdt[4 * n] = (params["CT"] * params["Q"]) - params["k"] * (T - params["T_0"]) #temperature evolution
    return dxdt


# -----------------------------------------------------
# Subfunctions
# -----------------------------------------------------

'''def distances(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    d = sqrt(dx ** 2 + dy ** 2)
    return d, dx, dy

def distance(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    d = sqrt(dx ** 2 + dy ** 2)
    return d
'''  #depreciated; slow

def applied_force(n, coords, treeData, alpha, V, Lx, t): # (From Electric Field)
    # Initialize force to zero
    Fa_x = zeros(n)

    x_p = coords[:,0]
    y_p = coords[:,1]

    # Logical index of particles inside domain - prevents forward motion of particles outside of the bounded region
    inside = (x_p < Lx)
    is_void = (ind_func(x_p, y_p) == 1)[0]

    if True: # Sam's orginal code; debug gate
        Fa_x[inside] = alpha[inside] * Volt / ((0.5) * Lx)
        return Fa_x * 1e15

    else: # debug gate
        # Apply scaling only to particles within [0,Lx]
        # only apply force to particle inside of a void

        inside, is_void, out_void, is_behind = treeData
        void_indices =  where(inside & is_void)[0] # converts the boolean array to an array of indicies at which the boolean array was true; find the indicies of the particles inside of the voids
        
        '''
        This section deals in adding back a force to those particles inside of the void if there 
        - there a electromagnetic edge effects that we will ignore for now (i.e. purterbations to the field at the edge of a conductor)
        We will set some radius (in params) for which there must be a particle *behind* the current particle in order for the field to proagate to the current particle
        If this condition is met, we will add the force back.
        '''
        force_val = alpha * Volt / (0.5 * Lx)

        """# Identify indices of particles in a void
        void_indices =  where(inside & is_void)[0] # converts the boolean array to an array of indicies at which the boolean array was true; find the indicies of the particles inside of the voids
        
        # query_ball_point finds all particles within radius 'r' for each void particle
        #neighbors_list = kdTree.query_ball_point(coords[void_indices], r) # this returns an array of lists. The list in the ith element of the array contains the indicies of the points within a distance r of the ith particle
        _, neighbors_list = kdTree.query(coords[void_indices], k=20, distance_upper_bound=r) # this gives a rectangluar array which is preffered for the below usage; max number of neighbors is 20
        neighbors_list[neighbors_list == 350] = 0 # fix the padding

        for i, void_idx in enumerate(void_indices):
            # neighbors_list[i] contains indices of all particles within radius r
            potential_neighbors =  array(neighbors_list[i]) # neigbors of the ith particle
            
            # only care about neighbors that are strictly behind (smaller x). also exclude the particle itself (distance 0)
            xi = x_p[void_idx]
            is_behind = x_p[potential_neighbors] < xi # moved to the parent function
    
            
            if  any(is_behind):
                Fa_x[void_idx] = force_val[void_idx] # this is vectorized (poorly) below"""

        '''cond = x_p[void_indices] < npmax(x_p[neighbors_list],axis=1) # should see if each particles x distance is less than the largest x-distance of its neighbors
        is_behind = where( cond ,True,False)
        print(is_behind)
        Fa_x = where(inside & is_void & is_behind,force_val, zeros_like(force_val))'''
                    
        return Fa_x * 1e15 # Sam oringally had the charge at 10^5 C - this was rediculous. Instead I scale the force directly so that I can use a more realistic charge across all forces.

def drag_force(x, y, x_v, y_v, eta, Cd):

    #eta_prime =  array([ eta if ind_func(x[i],y[i]) == 1 else eta*params['viscosity_multiplier'] for i in range(len(x_v))])
    # the above can be vectorized for increased efficiency
    VoidMask = (ind_func(x,y) == 1)
    eta_prime =  where(VoidMask[0], eta, eta * params['viscosity_multiplier']) # results in the same vector as before, but much quicker

    Fd_x = -eta_prime * Cd * x_v
    Fd_y = -eta_prime * Cd * y_v

    return Fd_x, Fd_y

#from scipy.spatial.distance import pdist, squareform # pdist takes advantage of the symmetry of distance calculations

def vdW_Force_AND_Dipole_Force(coords, tree, R=params["Average_particle_Radius"], A=params["Hamaker Constant"], Ex=params["V"]/params["L_x"], Ey=0., eps_r=150.):
    # since both use a distance matrix, combine them for efficiency - only compute matrix once
    '''
    Docstring for vdW_Force

    This computes the van der Waals attractive force between each particle.
    
    :param x: array if x positions
    :param y: array of y positiosn
    :param R: Radius of the silver nanoparticles (meters).
    :param A: Hamaker constant (Joules).
    '''
    #return 0.,0. # debug
    # In the Hamacker model, the van der Waals force between two spheres of radius R1 and R2 with separation r is (A)(R1)(R2) / 6(R1+R2)r^6
    # both particles feels the force reciprically via Newton's third law

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

    mask = (dist < 2.001 * R)  # logical index of particles that are less than two partilce radii apart; we only compute vdw force for *not* these to prevent collisions; we choose 2.001 to prevent the stiffness of the force to cuase particles to "slingshot" about

    # van der Waals Calculation
    
    h = dist - 2. * R # Surface-to-surface separation

    h_min = 1e-10 * R
    h =  maximum(h, h_min) # Prevent numerical blow-up (we will be dividing by a power of h later on)

    # Hamaker vdW force magnitude 
    d = h+2.*R
    F_vdW_mag =  A/6 * ( (4.*R**2.*d)/(d**2. -4.*R**2.)**2 + (4.*R**2. * d)/(d**4.) + (2.*d)/(d**2.-4.*R**2.) - 2./(d)  ) #A * R / (12.0 * h**2) # Derjaguin approximation - h << R

    # Unit vectors; need to project the force onto the vectro in between the two particles
    ex = dx / dist
    ey = dy / dist # since dy is signed this encode force direction as well; points from i to j

    # Force components on particle i due to j; i.e. the force should point from particle i to particle j
    # if particle i is behind j, ex is positive, we want to x force to point in the positive direction since this will pull i towards j
    # if j is behind i then ex is negative and so is the force. This is again desired as it pulls particle x back, towards j.
    Fx_vdW = F_vdW_mag * ex # ex and ey already contain the force direciton.
    Fy_vdW = F_vdW_mag * ey 
    Fx_vdW[mask] = 0. # zero the forces when the distances are too small
    Fy_vdW[mask] = 0.

    # initialize force vectors forces
    f_vdW_x =  zeros(len(x))
    f_vdW_y =  zeros(len(y))

    # Add force to i, subtract (Newton's 3rd) from j
    add.at(f_vdW_x, i,  Fx_vdW)
    add.at(f_vdW_x, j, -Fx_vdW) # the force on j is opposite the force on i
    add.at(f_vdW_y, i,  Fy_vdW)
    add.at(f_vdW_y, j, -Fy_vdW)
        
    #return f_vdW_x, f_vdW_y
    # coulomb force

    eps0 = params["eps_0"]

    q = params["alpha"] # charge vector
    q1 = q[i]
    q2 = q[j]

    f_c_x  =  zeros_like(f_vdW_x)
    f_c_y =  copy(f_c_x)

    f_coulomb_mag = - divide(q1*q2 , (eps0*eps_r * h**2 * 4. *  pi)) # this should be repulsive so we add the minus sign

    F_c_x = f_coulomb_mag * ex # ex and ey already contain the force direciton.
    F_c_y = f_coulomb_mag * ey 
    F_c_x[mask] = 0. # zero the forces when the distances are too small
    F_c_y[mask] = 0.

    add.at(f_c_x, i,  F_c_x)
    add.at(f_c_x, j, -F_c_x) # the force on j is opposite the force on i
    add.at(f_c_y, i,  F_c_y)
    add.at(f_c_y, j, -F_c_y)
    
    return f_c_x + f_vdW_x, f_c_y + f_vdW_y

def interfacial_force(n, x_p, y_p, wI, RI, Lx):
    return 0. # I'm setting this force to zero becuase I can't think of physical force that would scale strictly off of distance like this 



def pinning_force(n, x_p, y_p, w_pin, x_pin, y_pin, R_pin, Lx):
    return 0., 0. # for simplicity (temporary) - on further consideration, the pinning potential is endemic to semiconductors, of which these memristor are not made of. Thus I fail t understand why we ever had a pinning potential.



def temperature_fluctuations(n, eta, T_coeff, T, rand_x, rand_y):
    noise_scale =  sqrt(eta * T_coeff * T)
    Ft_x = rand_x * noise_scale
    Ft_y = rand_y * noise_scale
    return Ft_x/60., Ft_y/60. # scaled so that these forces are not dominant - they should be small relative to the other forces

# # ------------------------
# # TODO: Residual Force
# # Fast local density gradient approximation and then force based on that
# # ------------------------

def residual_force(n, x_p, y_p, Lx, Ly):
    return 0., 0. # remove the residual force - instead we use the drag force to contain 


def finishing_array(x_p, L, fin):
    """
    Check if particles have reached the end.
    Returns a boolean array (or True if fin == 0).
    """
    if fin == 1:
        return x_p < (2 * L - 5e-6)
    else:
        return True