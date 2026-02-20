#####################################################################
##################### RUN DENDRITIC SIMULATION ######################
#####################################################################

from time import time
from defineParameters import params

def run(Hamaker, Visc):
    from runSimulation import run_simulation
    start = time()
    t, states = run_simulation(params, Hamaker, Visc)
    ellapsed = time() - start
    print(f"Simulation finished successfully in {ellapsed} seconds!")
    #print(f"Final time: {t[-1]:.3f}")
    print(f"Final time: {t:.3f}")
    print(f"State vector shape: {states.shape}")

if __name__ == "__main__": # Only run the following code if this file is being run directly, not when it’s imported by another script
    #run(params['Hamaker Constant'],params['viscosity_multiplier'])
    run(.5,.95)