from numpy import random, zeros, linspace, concatenate, zeros_like, arange
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from defineParameters import params

from calculateForces import calculate_forces
from calculateCurrent import calculate_current
from time import time

Start_Volt = 0.
End_Volt = 10.
increment = 0.5
sim_time = 10.
    

def run(Hamaker, Visc, time_to_run,V_func):
    start = time()
    c_max = run_simulation(params, Hamaker, Visc, time_to_run,V_func)
    ellapsed = time() - start
    print(f"Simulation finished successfully in {ellapsed} seconds!")
    return c_max


def run_simulation(params, Hamaker, Visc, time_to_run, V_func):

    integrable_calcualte_forces = lambda t,x: calculate_forces(t,x,params,Hamaker,Visc,V_func) # this function only takes t and x so it can be used by an integrator

    random.seed(1)

    # ------------------------
    # Unpack parameters
    # ------------------------
    n = params["n"]
    To = params["T_0"]
    tspan = params["tspan"]
    if params["Until_end"]:
        tspan = linspace(tspan[0],time_to_run,len(tspan)) # manual control; 0.771
    electrode_height = params["electrode_height"]

    # ------------------------
    # Randomize initial particle positions
    # ------------------------
    xo = 2.5 * random.rand(n)  - 2.5 # start behind the device - should prevent starting in voids
    yo = electrode_height * random.rand(n) - 0.5 * electrode_height

    # ------------------------
    # Zero all initial particle velocities
    # ------------------------
    vox = zeros(n)
    voy = zeros(n)

    # ------------------------
    # Initial state vector
    # ------------------------
    initial = concatenate([xo, vox, yo, voy, [To]]) # To is wrapped in a list so that concatenate resolves

    # ------------------------
    # Run ODE simulation
    # ------------------------
    print("Starting solver...")
    V = V_func(0.)

    # now run the simulation
    t = tspan[0]
    current_time_indicator = 1 # the next time at which to capture the animation frame; index of the tspan array

    max_current = 0.

    # use scipy integrator
    solver = RK45(
    fun=integrable_calcualte_forces,
    t0=tspan[0],
    y0=initial,
    t_bound=tspan[-1],
    rtol=params['rtol'],
    atol=params['atol'],
    max_step=min(tspan[1]-tspan[0],1e-2) # enables the animation frames to be evenly spaced
    )
    current_time_indicator = 1
    while solver.status == "running": # run the solver
        solver.step() # one step
        t = solver.t # current time
        y = solver.y # current state vector
        # now that everything is updated, we need to stream this data into an animation file
        while ( current_time_indicator < len(tspan)
                and t >= tspan[current_time_indicator]): # changed from if to allow catch up if a jump is too large
            current_time_indicator += 1
            # get the electric current data
            if not params['is_Voltage_constant']: # set the voltage
                Volt = params["V_func"](t)
            else:
                Volt = V
            if Volt == 0.: pass # this should save a lot of time
            else:
                max_current = max(max_current, (Volt/abs(Volt) * calculate_current(
                y[0:n], y[2*n:3*n], params["L_x"],params["L_y"], abs(Volt),
                params["lambda"], params["Rt"], y[-1], params['num_e']#params["steps"], params["num_e"]
                )))
    t = solver.t

        # now we have completed the integration
    # now the file writer is closed
    print("Simulation complete.")
    return max_current


if __name__ == "__main__":
    Volts = arange(Start_Volt, End_Volt+increment,increment)
    currents = zeros_like(Volts)
    for i_V in range(len(Volts)):
        def voltage(t):
            return Volts[i_V]
        currents[i_V] = run(1.,1.,sim_time,voltage)

    plt.plot(Volts,currents,ls=':',marker='.')
    plt.tight_layout()
    plt.grid()
    plt.xlabel("voltage")
    plt.ylabel("maximal current")
    print("Volts:",Volts)
    print("currents:",currents)
    plt.show()