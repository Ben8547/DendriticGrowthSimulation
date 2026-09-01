from time import time
from defineParameters import params
import matplotlib.pyplot as plt
from numpy import random, zeros, meshgrid, linspace, concatenate, array, exp, average, zeros_like, savetxt, concatenate
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from viscosity_field import make_globular_indicator
from calculateForces import calculate_forces
from viscosity_field import viscocity_gradient

def run_simulation(params, wpa, numVoids, VOLT):
    """
    Translated from MATLAB runSimulation.m
    Runs the full dendritic growth simulation with visualization and GIF output.
    """

    '''# Initialize globals (same as MATLAB)
    from calculateForces import I_saved, t_saved
    I_saved.clear()
    t_saved.clear()'''

    voltage_hist = [0.]

    current_resist = 1.

    ind_func = make_globular_indicator(numVoids) # get a function to check viscocity state; index function of the void set

    WPA = concatenate((params['wp_attract'], params['wp_repulse']), axis=None)

    integrable_calcualte_forces = lambda t,x: calculate_forces(t,x,params, WPA, ind_func, V_func = lambda t: VOLT, resist=current_resist) # this function only takes t and x so it can be used by an integrator

    random.seed(1)

    Lx = params["L_x"]

    # ------------------------
    # Unpack parameters
    # ------------------------
    n = params["n"]
    To = params["T_0"]
    tspan = params["tspan"]

    tspan = linspace(tspan[0],7.,len(tspan)) # manual control; 0.771

    electrode_width = params["electrode_width"]
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
    tindex = 1
    V = params["V"]



    # now run the simulation
    from calculateCurrent import calculate_current
    t = tspan[0]
    current_time_indicator = 1 # the next time at which to capture the animation frame; index of the tspan array
    from scipy.integrate import RK45, BDF, Radau # various integrators to try

    # use scipy integrator
    current_histroy = []
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
    times = [0.]
    while solver.status == "running": # run the solver
        solver.step() # one step
        t = solver.t # current time
        y = solver.y # current state vector
        
        # now that everything is updated, we need to stream this data into an animation file
        while ( current_time_indicator < len(tspan)
                and t >= tspan[current_time_indicator]): # changed from if to allow catch up if a jump is too large
            current_time_indicator += 1
            # get the electric current data
            # set the voltage
            Volt = VOLT
            times.append(t) # add to time history
            if Volt == 0.: current_histroy.append(0.) # this should save a lot of time
            else:
                current = calculate_current(
                y[0:n], y[2*n:3*n], params["L_x"],params["L_y"], abs(Volt),
                params["lambda"], params["Rt"], y[-1], params['num_e']#params["steps"], params["num_e"]
                )
                current_resist = Volt/current
                current_histroy.append(Volt/abs(Volt) * current)
            #print(f"time: {t}, current {current_histroy[-1]}") # debug tool
            voltage_hist.append(Volt)
    states = solver.y
    t = solver.t

        # now we have completed the integration
    # now the file writer is closed
    savetxt(f"STATES-m={params['m']}-pin={wpa}-voids={numVoids}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",states)
    #savetxt(f"Current-m={params['m']}-pin={params['wpa_attract']}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",current_histroy)
    #savetxt(f"Current-m={params['m']}-pin={params['wpa_attract']}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",voltage_hist)
    print("Simulation complete.")
    return t, states, times, current_histroy, voltage_hist

def run(WPA, NumVoid, VOLT):
    start = time()
    t, states, t_hist, c_hist, v_hist = run_simulation(params, WPA, NumVoid, VOLT = VOLT)
    ellapsed = time() - start
    print(f"Simulation finished successfully in {ellapsed} seconds!")
    print(f"Final time: {t:.3f}")
    print(f"State vector shape: {states.shape}")
    return max(c_hist)

if __name__ == "__main__": # Only run the following code if this file is being run directly, not when it’s imported by another script
    volts = []
    wpas = [exp(i) for i in range(0,11,1)]
    for wpa in wpas:
        found = False
        initial_volt = 10.
        volt2 = [0.,initial_volt]
        curr_left = 0.
        curr_right = 1e3
        while not found:
            new_V = (volt2[1]+volt2[0])/2
            max_c = run(wpa, 80, VOLT = new_V)
            if max_c < 5e-2:
                if volt2[1] >= initial_volt:
                    volt2[1] = volt2[1]*2.
                else:
                    volt2[0] = new_V
            else:
                volt2[1] = new_V
            if abs(volt2[0] - volt2[1]) < 1e-4:
                found = True
                print(f"Voltage for wpa={wpa} is ", (volt2[0]+volt2[1])/2.)
        volts.append((volt2[0]+volt2[1])/2.)
    print(volts)
    print(wpas)
    plt.plot(wpas,volts)
