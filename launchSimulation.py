#####################################################################
##################### RUN DENDRITIC SIMULATION ######################
#####################################################################

from defineParameters import params
from runSimulation import run_simulation
from time import time

if __name__ == "__main__": # Only run the following code if this file is being run directly, not when it’s imported by another script
    start = time()
    t, states = run_simulation(params)
    ellapsed = time() - start
    print(f"Simulation finished successfully in {ellapsed} seconds!")
    #print(f"Final time: {t[-1]:.3f}")
    print(f"Final time: {t:.3f}")
    print(f"State vector shape: {states.shape}")