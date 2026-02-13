from defineParameters import params
from numpy import exp, ones, sum, copy, max, minimum, argmin, sqrt, argsort, column_stack, vstack, zeros, float32, searchsorted #tile, full, isnan, sqrt, min, nan, newaxis
from numpy.random import rand
from scipy.spatial import cKDTree # makes spatial searches much faster
#from numba import njit

#from getNextIndex import get_next_index

def distances(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    d = sqrt(dx ** 2 + dy ** 2)
    return d

def boltz_fact(x_i,x_j,Volt, T, L_x):
    '''
    x_i is the current location of the electron and x_j is the location of the electron in the proposed step
    '''
    V_i = Volt * (x_i/L_x)
    V_j = Volt * (x_j/L_x)
    return exp( (V_j - V_i) * params['q'] / params['k_B'] / T ) # \Delta energy is -(V_j - V_i) * -params['q']
# the above energies assume a paralell plate capacitor of infinte area. This is hopefully a decent approximation

def boltz_fact_dist(d,Volt, T, L_x):
    '''
    x_i is the current location of the electron and x_j is the location of the electron in the proposed step
    '''
    Delta_V = Volt * (d/L_x)
    return exp( (Delta_V) * params['q'] / params['k_B'] / T ) # \Delta energy is -(V_j - V_i) * -params['q']
# at a temperature of 300 K, params['q'] / params['k_B'] / T = 38.681727071833606
# the above energies assume a paralell plate capacitor of infinte area. This is hopefully a decent approximation


#@njit # for speed
def calculate_current(x, y, L_x, L_y, Volt, lambda_, Rt, T, num_e): # Ben's Monte-Carlo method
    # What this function does:
    # 0) If voltage is 0, return 0 since that will be the end result anyway
    # 1) Introduce num_e electrons at random y-axis locations.
    # 2) for each particle choose to move forward with boltzman probability or backwards so sum is unity
    # 3) Find 3 closest particles in the specified direction
    # 4) Jump to that particle and record the resistence of the jump
    # 5) Interate until we reach the end electrode
    # 6) Sum up the resistances of the jumps
    # 7) Average the resistences together
    # 8) If not too slow - repeat and average again
    # 9) Multiply the final reciprial resistence by the voltage - gives the current; return thus value

    

    if Volt == 0. or Volt == 0: # step 0
        return 0.

    #Step 1 - introduce electrons on the left electrode
    e_position = vstack([zeros((1,num_e)),L_y * rand(1,num_e)],dtype=float32) # use float 32 for performance; most of the error comes from integration anyway
    # row 0 are x-values; row 1 are y-values

    #x_now = copy(x)
    #y_now = copy(y)
    resist = zeros(num_e, dtype=float32)

    # Turn the points into a KDTree for easy spatial search
    x_sort_idx = argsort(x) # implmenent a binary search for efficiency
    x_sort = x[x_sort_idx]
    y_sort = y[x_sort_idx] # sort both by the x indicies to preserve order

    for e in range(num_e): #iterate over all simulated electrons
        prev_i = -1
        x_e, y_e = (e_position[0,e],e_position[1,e]) # initial electron position
        resist_e = 0. # resitances along the path of a single electron
        while x_e < L_x: # while we have not reached the right electrode
            #print(x_e)
            # step 2: choose direction; use shortest distance to next particle in line to proxy jump distance
            i = searchsorted(x_sort,x_e,side='left') # gives me the index of x_e in the sorted list; need to change side to left so that we don't count x_e

            if i == 0:
                dx_min = x_sort[0] - x_e
            elif i == len(x_sort)-1 or x_e > x_sort[-1]: # include x_e > x_sort[-1] because all of the particles are initially behind the electrode
                dx_min = x_e - x_sort[-1]
            else:
                dx_min = min(x_sort[i+1] - x_e, x_e - x_sort[i-1]) # estimate; x distance considered only

            Boltzmann_F = boltz_fact_dist(2*dx_min,Volt,T,L_x) # this gives a ratio of occupancies at approximate closet state to the left vs the appproximate closest state to the right

            # thus we can say that the probability of going right is approximately Boltzmann_F/Boltzmann_B times the probability of going left
            Probability_left = 1./(1. + Boltzmann_F) # probability of moving left
            '''if rand() > Probability_left:
                Forward = True # choose to move forward with probability according to the electric field energy bias
            else:
                Forward = False
            if x_e <= 0.:
                Forward = True # force forward motion if we are at or (somehow) behind the electrode'''
            Forward = rand() > Probability_left # more concise than the above
            if x_e <= 0.:
                Forward = True
            
            if Forward: # if moving forward
                x_possible = x_sort[i+1:] # i+1 to avoind stationarity
                y_possible = y_sort[i+1:]
            else:
                x_possible = x_sort[0:i]
                y_possible = y_sort[0:i]
            coords = column_stack((x_possible, y_possible))
            Tree = cKDTree(coords) # search tree - should speed up later distance dependant force computations
            # find nearest particle in front of x_i
            d, ind = Tree.query([x_e,y_e]) # find closest point to (x_e,y_e) using the search tree
            # at this point d is an empty array when x_possible is empty
            d = minimum(d,L_x-x_e) # If the final electrode is closer then jump there instead of a particle.
            resist_e +=  Rt * exp( d / lambda_) # add to the total path resistance
            if d - (L_x-x_e) > 1e-15:
                x_e, y_e = ( x_possible[ind], y_possible[ind]) # update the electron positions
            else: # d = L_x - x_e
                x_e, y_e = ( L_x, 0.) # y_position doesn't matter at the end electrode

            resist_e += Rt * exp(d / lambda_)
            # in the future I would like to make it so that this chooses between the three loswest values with some weight; for now the nearest neighbor model works
        resist[e] = resist_e
    #R = sum(resist)/num_e # average the resitances
    R = min(resist) # only use for testing, use above comment for regular use; actually could make some sense if all of the electrons are forced through the few paths of least resistance - could incoorperate some weighted average
    return Volt / R


'''def calculate_current(x, y, L, Volt, lambda_, Rt, T, num_e):
    """
    Translated from MATLAB calculateCurrent.m
    Simulates electron hopping and returns current I.

    x is a vector of particle positions (abscissa)
    y is a vector of particle positions (ordinate)
    """
    threshold_dist = 50
    x_ind_sort = argsort(x) # sort the x_values from least to most. Return the inicies
    x_sort = x[x_ind_sort]
    y_sort = y[x_ind_sort]

    # Skip simulation if device too long
    min_distance_to_end = min(L - x) # minimum distance of a particle to the end electrode
    if min_distance_to_end > threshold_dist:
        return 0
    
    # the goal is to estimate the sum R_M \sum_{i=0}^{N-1} R_t * exp( (x_{i+1} - x_i) / lambda ) which is the resistance
    # along one possible path. There are infintely many possible paths. If we consider each possible path a resistor in paralell then
    # the total system resistance is given by $ 1/R_tot = 1/R_{M,1} + 1/R_{M,1} + 1/R_{M,2} + 1/R_{M,3} + ... $.
    # Thus the goal should be to compute this sum and then take the reciprical. This is simular to the problem of computing a partition
    # function for a system with infinite states; we can do it with monte-carlo methods. We do need to decide which paths contribute the most
    # to the final resistance. If we assume that there are $N$ total paths (the paths that we end up sampling) the the total resistence is 
    # this is the resistance along one possible path for the electron.
    # To simplify the number of paths to choose from we use the nearest neighbor model - i.e. the only path the matters is the path between the two closest particles.
    # Other paths will have a much lower change of beign taken. Thus we start with each particle; find the next closest with a higher x-value and so on.
    # Eventually we get to the farthest right particle.
    # This affords us a decent estimate of the resistance of the material.
    # Then the current may be computed with $V = IR \implies I = V/R$ (then inversion of the sum is not needed since we only need 1/R anyway).
    # One last note is that the tunneling probability should be additionally modified by the Boltzmann factor bias that
    # forces migration accordig to the voltage bias. This should allows us to see pulses in the current that mimic pulses in the voltage.

    x_max = x_sort[-1]
    inv_Resists = []
    for i in range(num_e):
        # start an electron at the y-level of each particles' x-value if that particle is one of the num_e closest particles to the left eleectrode
        x_i = x_sort[i]
        y_i = y_sort[i]
        Resist_i = []
        Resist_i.append( Rt * exp( x_i / lambda_) * boltz_fact(0,x_i,Volt,T,L)  )
        while x_i <= x_max:
            x_possible = x[x > x_i]
            # find nearest particle in front of x_i
            d = distances(x_i,x_possible,y_i,y[x > x_i]) # distance vector comparing each other point x_j to x_i where x_j > x_i
            # at this point d is an empty array when x_possible is empty
            d = app(d,L-x_i) # If the final electrode is closer then jump there instead of a particle.
            j = argmin(d)
            
            if j == len(d)-1: # if the electrode is the closest jump; always the case when x = x_max
                Resist_i.append( Rt * exp( d[j] / lambda_) * boltz_fact(x_i,L,Volt,T,L)  )
                x_i = x_max+1 # terminate the while loop
            else:
                x_j = x_possible[j]
                # compute resistance of the tunnel
                Resist_i.append( Rt * exp( d[j] / lambda_) * boltz_fact(x_i,x_j,Volt,T,L)  )
                x_i = copy(x_j)
                y_i = x[j]

        inv_Resists.append( sum(1./array(Resist_i)) )

    inv_R = sum(inv_Resists)

    return inv_R * Volt # return an estimate of the classical current
'''
# below is the old model
'''    LEFT_ELECTRODE = n + 1
    RIGHT_ELECTRODE = n + 2

    # Start all electrons at left electrode
    curr_indexes = full(num_e, LEFT_ELECTRODE) # array of length of the number of simulated electrons filled with n+1
    currx = zeros(num_e)
    curry = full(num_e, nan) # array of length of the number of simulated electrons filled with nan

    for _ in range(steps):
        dx = tile(x, (num_e, 1)) - currx[:, newaxis] # generates a len(x) by num_e matrix. Rows are copies of x minus some value.
        dy = tile(y, (num_e, 1)) - curry[:, newaxis] 
        dy[isnan(dy)] = 0 # on first iteration it sets the dy matrix to 0 - not sure why we don't just set it to zero from the start
        d = sqrt(dx ** 2 + dy ** 2)

        positive = dx >= 0 # Bollean array, returns True at index i if x[i] >=0 and False otherwise
        resist = Rt * exp(d / lambda_) # resistance 
        dist_weights = positive * (1 / resist)

        resist_end = Rt * exp( (L - currx) / lambda_ )
        to_end = 1. / resist_end
        to_end[curr_indexes == RIGHT_ELECTRODE] = 1000  # stay at end
        to_begin = zeros(num_e)

        tot_weights = hstack([dist_weights, to_begin[:, None], to_end[:, None]]) # makes a 3 x 

        curr_indexes, currx, curry = get_next_index(tot_weights, x, y, currx, curry, curr_indexes, L)

    I = sum(currx == L) # after the interations, the current is computed to be the 
    return I'''