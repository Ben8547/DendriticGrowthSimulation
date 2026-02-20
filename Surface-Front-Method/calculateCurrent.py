from defineParameters import params
from numpy import exp, ones, abs, copy, max, minimum, argmin, sqrt, argsort, column_stack, vstack, zeros, float32, searchsorted, mean #tile, full, isnan, sqrt, min, nan, newaxis
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

    """#Step 1 - introduce electrons on the left electrode
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
            if abs(d - (L_x-x_e)) > 1e-15:
                x_e, y_e = ( x_possible[ind], y_possible[ind]) # update the electron positions
            else: # d = L_x - x_e
                x_e, y_e = ( L_x, 0.) # y_position doesn't matter at the end electrode

            resist_e += Rt * exp(d / lambda_)
            # in the future I would like to make it so that this chooses between the three loswest values with some weight; for now the nearest neighbor model works
        resist[e] = resist_e
    #R = sum(resist)/num_e # average the resitances"""
    resist = Rt * exp(mean(L_y - x) / lambda_)
    #R = min(resist) # only use for testing, use above comment for regular use; actually could make some sense if all of the electrons are forced through the few paths of least resistance - could incoorperate some weighted average
    return Volt / resist

