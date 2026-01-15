# Ported into python from matlab by Ben Campbell
# Original Author: Samuel Moore-Smith and Jason Yuan
# Further updates by Ben Campbell
'''
Purpose: Define Parameters and Setup Simulation for Gold Memristor Simulation
'''

import numpy as np
from numpy.random import rand, seed
from scipy.constants import Boltzmann, elementary_charge

s = 10

seed(s) # for consistant testing; may comment out for actual runs

params = { # create a dictionary to hold parameters
    #--------------------------------
    # Integration method
    #--------------------------------

    "BS" : False, # if True, uses custom bulirsch-stoer algorithm, if False it uses scipy.integrate.solve_ivp()
    'BS_error_per_unit_time': 1e-3,
    "scipy_tag" : "RK45", # should be a string for the method argument of scipy.integrate.solve_ivp()
    "rtol" : 1e-3, # relative error param for scipy integration
    "atol" : 1e-3, # absolute error param for scipy integration

    #--------------------------------
    # Basic Simulation Parameters
    #--------------------------------

    'i' : 1, # Simulation number
    "fin" : 1, #If 1, stops movement of particles once they hit the other end

    #--------------------------------
    # Well-understood Physical Constants
    #--------------------------------

    'k_B' : Boltzmann, # Boltzmann's constant in J/K
    "T_coeff" : 3000, # !!! Temperature coefficient [K^-1] -- this is a guess, not sure what to use here

    #--------------------------------
    # Simulation Length and Particle Count
    # --------------------------------
    
    "n" : 500, # Number of particles
    "simulation_length" : 0.6, #6, # in seconds

    #--------------------------------
    # Initial position box and material size
    #--------------------------------
    'L_x' : 60, #100, # Size of material in x-direction
    'L_y' : 100, # Size of material in y-direction

    #--------------------------------
    # Heat related parameters
    #--------------------------------
    'T_0' : 300, # Initial temperature in K
    'Q' : 1, # Thermal Energy [J]
    'alpha' : 1e-5, # Thermal diffusivity in m^2/s
    # Jasons code?
    'CT' : 0.24*5e-9, # !!! Heat capacity = specific heat capacity*mass [J/K] http://www2.ucdsb.on.ca/tiss/stretton/database/Specific_Heat_Capacity_Table.html -- check for gold

    #--------------------------------
    # Pinning Force Parameters
    #--------------------------------
    'm' : 100, # 100 # number of pinning sites
    # Pinning Site Locations and Forces
    "wpa_repulse" : 2000, #2000,
    "wpa_attract" : 500, #500,

    #--------------------------------
    # Applied Electric Field Parameters
    #--------------------------------
    
    "is_Voltage_constant" : False,
    "V" : 3.,#3., # applies voltage

    #--------------------------------
    #Residual Stress Parameters
    #--------------------------------
    'w_resid' : 1., #1.0, #Strength of residual force

    'E' : 323.5e9, # Young's modulus of AlN in Pa = N/m^2
    'nu' : 0.23, # Poisson's ratio of AlN, not sure if this should be used or be constant
    "k_C" : 8.99e9, # Coulomb's constant in N*m^2/C^2
    "q" : elementary_charge, # Elementary charge in C (charge of an electron / proton)
    "r_Ag" : 126e-12, # Ionic radius of Ag in m

    #-----------------------------------
    # Stress induced void map parameters
    #-----------------------------------
    'n_regions': 20,
    'blob_scale': 0.08,
    'num_lattice_points': int(1e6),
    "viscosity_multiplier": 2.5
}

# add to params

# Applied force in void radius
params["elec_transfer_radius"] = 0.01 * params["L_x"] # we ahve the atomic radius of silver atoms stored, however this is a nanoparticle - many atoms. This parameter should be propotional to the particle size though because electric field should only be propagating to touching particles.

# simulation parameters
params['dt'] = params["simulation_length"]/500.  # Time step for simulation; originally 2000
params["tspan"] = np.arange(0, params["simulation_length"], params["dt"]) # Time span for simulation output

# material and electrode dimensions
params["electrode_width"] = params["L_x"] / 2 # Width of electrodes on material
params["electrode_height"] = params["L_y"] # Height of electrodes on material

#pinning parameters
params['wp_attract']= (-(rand(1,params['m']//2))*params['wpa_attract'] - params['wpa_attract']/2) # update the value from a scalar to an array
params['wp_repulse']= ((rand(1,params['m']//2))*params['wpa_repulse'] + params['wpa_repulse']/2)
params['w_pin'] = np.concatenate((params['wp_attract'], params['wp_repulse']), axis=None) # Combine attractive and repulsive pinning forces into one array
params['x_pin'] = rand(params['m'],1) * params['L_x'] # Randomly distribute pinning sites in x-direction
params['y_pin'] = rand(params['m'],1) * params['L_y'] - 0.5*params['L_y'] # Randomly distribute pinning sites in y-direction
#params['w_pin'] = 100 # Pinning potential amplitude
params['R_pin'] = rand(params['m'],1) * 10. # Pinning potential distance

# Jasons code?
params["k"] = 420./params['L_x']; # !!! Heat transfer coefficient [W/m^2K] from https://www.spiraxsarco.com/learn-about-steam/steam-engineering-principles-and-heat-transfer/heat-transfer

#applied electric field parameters
params['alpha'] = 0.5e5 #rand(params['n']) * 1e5 #; I'm not sure why Sam had this set to be random values, let's keep it consistant for testing purposes at least

#--------------------------------
# Drag Force Parameters
#--------------------------------

params['eta'] = 1.#1 # viscosity
params["Cd"] = 50. # 1.8e4 # Drag Coefficient

#--------------------------------
# Interfacial Potential Parameters
#--------------------------------

params['wI'] = 2e4 # Interfacial potential amplitude only with clusters
params['RI'] = 25 #Interfacial potential distance only with clusters

#--------------------------------
# Current Calculation Parameters
#--------------------------------
params['num_e'] = 20 #100 # Number of electrons to simulate
params['Rt'] = 1 # Rt is the tunnelling resistance amplitude (assumed the same for all islands)
params['lambda'] = 2 #2. # effective tunneling length
params['steps'] = 120

#--------------------------------
# Voltage function
#--------------------------------
if not params['is_Voltage_constant']:
    def voltage(t):
        time = t % 2e-2 # period of 2 centi seconds
        if time > 1e-2: # 0 until 1e-2 s and then V onwards; repeats from 2e-2 s
            return params["V"]
        else: 
            return 0
    params["V_func"] = voltage

#--------------------------------
# File Settings
#--------------------------------

params["filename"] = f"dendrite_growth_simulation-pulses-newcurrent-{params['num_e']}_electrons-visc-{params['viscosity_multiplier']}.gif"

params["seed"] = s

# if we run this file directly, do some stuff
if __name__ == "__main__": # graph the voltage
    if not params["is_Voltage_constant"]:
        import matplotlib.pyplot as plt
        times = np.linspace(0,5e-1,1000)
        voltages = [ params["V_func"](t) for t in times ]
        plt.plot(times,voltages)
        plt.title("The Voltage Pulses")
        plt.show()