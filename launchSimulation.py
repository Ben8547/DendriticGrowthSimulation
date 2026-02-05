#####################################################################
##################### RUN DENDRITIC SIMULATION ######################
#####################################################################

from time import time


def run():
    from defineParameters import params # I don't know how the compiler work well enough to trust that it will work correctly if I load before I change the constnats. This seems safer.
    from runSimulation import run_simulation
    start = time()
    t, states = run_simulation(params)
    ellapsed = time() - start
    print(f"Simulation finished successfully in {ellapsed} seconds!")
    #print(f"Final time: {t[-1]:.3f}")
    print(f"Final time: {t:.3f}")
    print(f"State vector shape: {states.shape}")

if __name__ == "__main__": # Only run the following code if this file is being run directly, not when it’s imported by another script
    run()