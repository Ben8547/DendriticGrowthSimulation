from numpy import random, zeros, meshgrid, linspace, concatenate, array, exp, average, zeros_like, savetxt
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from defineParameters import params
from viscosity_field import make_globular_indicator
from calculateForces import calculate_forces
from viscosity_field import viscocity_gradient


def run_simulation(params, wpa, numVoids):
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

    integrable_calcualte_forces = lambda t,x: calculate_forces(t,x,params, WPA, ind_func, resist=current_resist) # this function only takes t and x so it can be used by an integrator

    random.seed(1)

    Lx = params["L_x"]

    # ------------------------
    # Unpack parameters
    # ------------------------
    n = params["n"]
    To = params["T_0"]
    tspan = params["tspan"]
    if params["Until_end"]: # the point of this section is to estimate how long we need to run the integrator for in order to get an electron to the other node
        # we make the assumption that the only forces are the drag force and the electric field force with all particles stating from x=0 and rest. Then the model reduces
        # to a second order linear, constant coeficient equation that we can solve exactly; it is then possible to find the time at which the particles reach the end electrode.
        # since the viscosity of the medium changes, we take it to be the average value - computed via some integral over the density function on the interval.
        #from scipy.optimize  import root # find polynomial roots - to get eigenvalues of the ODE; fsolve to find the time
        # i actually just solved for the general solution's eigen values so we don't need the root function.
        from viscosity_field import average_density
        from scipy.optimize import fsolve # need this otherwise we would have to appeal to the Lambert W function (probably - I didn't actually do the algebra to that point)
        a = average(params["alpha"]) * params["V"]/ (0.5 * params["L_x"]) * 1e15# electric field acceleration
        b = params["Cd"] * average_density # drag coeficient (x-component)
        #print(b)
        c_2 = a/(b*b)
        c_1 = -c_2
        particularAndGeneralSolution = lambda t: c_1 + c_2 * exp(-b*t) + (a/b)*t - Lx
        derivative = lambda t: -c_2*b*exp(-b*t) + (a/b)
        tspan[-1] = 2.1 * fsolve(particularAndGeneralSolution,
                        array([0.1]),
                        fprime=derivative)[0] # since voltage is off half of the time we need to add a bit more time
        print(f"anticipated {tspan[-1]} seconds to be simulated")
        tspan = linspace(tspan[0],tspan[-1],len(tspan))

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

    
    #Update: I'm changing this to RKF45 since the Verlet method had too small time steps. I think this is because the verlet integrator requires the use of the next force vector, but we can only approximate this in the current scheme because the force depends upon the velocities (drag force). I will not be changing the variable labels, so keep that in mind if something mentions the verlet alorithm.
    # we write the velocity verlet function here and include the ability to stream the simualtion directly into a file. Thus we don't need to wait at the end for the animation to compile
    #error_tol = params['Verlet_error_per_unit_time']  
    from matplotlib.animation import FFMpegWriter # for streaming the frames into  a file

    # set up the animation figrue and artists
    writer = FFMpegWriter(fps=10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    plt.tight_layout(pad=3)

    # ---- Top plot: particle evolution ----
    ax1.set_xlim([-electrode_width - 5, Lx + electrode_width + 5])
    ax1.set_ylim([-electrode_height / 2 - 5, electrode_height / 2 + 5])
    ax1.set_xlabel("x-position")
    ax1.set_ylabel("y-position")
    ax1.set_title("Dendritic Growth Simulation")

    # Draw electrodes
    ax1.fill_betweenx(
        [-electrode_height / 2, electrode_height / 2],
        -electrode_width,
        0,
        color=(1.0, 0.8431, 0.0),
        alpha=0.3,
    )
    ax1.fill_betweenx(
        [-electrode_height / 2, electrode_height / 2],
        Lx,
        Lx + electrode_width,
        color=(1.0, 0.8431, 0.0),
        alpha=0.3,
    )

    view = "pinning"

    x = linspace(0.,params["L_x"],1000)
    y = linspace(-params["L_y"]/2.,params["L_y"]/2.,1000)
    X, Y = meshgrid(x,y)

    if view == "density":

        Z = viscocity_gradient(X,Y,ind_func,Visc) #ind_func(X,Y)
        #pcm = ax1.pcolormesh(X, Y, Z, cmap="pink", shading="auto", alpha=0.3)
        pcm = ax1.pcolormesh(x, y, Z, cmap="pink", shading="auto", alpha=0.3)

        #ax1.imshow(Z)
        plt.colorbar(pcm, ax=ax1,)
        #pcm.set_clim(vmin=0.,vmax=params["eta"])
    elif view == "pinning":
        def compute_pinning_potential(X, Y, params):
            x_pin = params['x_pin']
            y_pin = params['y_pin']
            w_pin = params['w_pin']
            R_pin = params['R_pin']

            U = zeros_like(X)

            # Sum contributions from all pinning sites
            for i in range(len(x_pin)):
                dx = X - x_pin[i]
                dy = Y - y_pin[i]
                d2 = dx**2 + dy**2
                U += w_pin[i] * exp(-d2 / (R_pin[i]**2))

            return U

        # Compute potential
        U_pin = compute_pinning_potential(X, Y, params)

        # Overlay (or replace viscosity plot if desired)
        pcm_pin = ax1.pcolormesh(
            x, y, U_pin,
            cmap="coolwarm",
            shading="auto",
            alpha=0.4
        )
        
        #plt.colorbar(pcm_pin, ax=ax1, label="Pinning Potential")


    current_histroy = [0.] # initialize the current history

    # now run the simulation
    from calculateCurrent import calculate_current
    t = tspan[0]
    current_time_indicator = 1 # the next time at which to capture the animation frame; index of the tspan array
    from scipy.integrate import RK45, BDF, Radau # various integrators to try

    with writer.saving(fig, f"m={params['m']}-pin={params['wpa_attract']}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+'.mp4', dpi=60): # this enables streaming the simulation data to a file concurrent with the simualtion - enables an approximate 5x speed up # f"dendrite_growth_simulation-pulses-newcurrent-{params['num_e']}_electrons-visc-{Visc}_vdW-{Hamaker}_Ly-{params["L_y"]}-{params['n']}particles"
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
        times = [0.]
        while solver.status == "running": # run the solver
            solver.step() # one step
            t = solver.t # current time
            y = solver.y # current state vector
            
            # now that everything is updated, we need to stream this data into an animation file
            while ( current_time_indicator < len(tspan)
                    and t >= tspan[current_time_indicator]): # changed from if to allow catch up if a jump is too large
                current_time_indicator += 1
                #particles.set_data(y[0:n], y[2*n:3*n])
                # get the electric current data
                if not params['is_Voltage_constant']: # set the voltage
                    Volt = params["V_func"](t)
                else:
                    Volt = V
                times.append(t) # add to time history
                if Volt == 0.: current_histroy.append(0.) # this should save a lot of time
                else:
                    current = calculate_current(
                    y[0:n], y[2*n:3*n], params["L_x"],params["L_y"], abs(Volt),
                    params["lambda"], params["Rt"], y[-1], params['num_e']#params["steps"], params["num_e"]
                    )
                    current_resist = Volt/current
                    current_histroy.append(Volt/abs(Volt) * current)
                #current_graph.set_data(times,current_histroy)
                #writer.grab_frame() # write the most recent frame to the file
                print(f"time: {t}, current {current_histroy[-1]}") # debug tool
                voltage_hist.append(Volt)
        states = solver.y
        t = solver.t

        # now we have completed the integration
    # now the file writer is closed
    #savetxt(f"TIMES-m={params['m']}-pin={wpa}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",t)
    #savetxt(f"STATES-m={params['m']}-pin={wpa}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",states)
    #savetxt(f"Current-m={params['m']}-pin={params['wpa_attract']}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",current_histroy)
    #savetxt(f"voltage-m={params['m']}-pin={params['wpa_attract']}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".csv",voltage_hist)
    particles, = ax1.plot(initial[0:n], initial[2*n : 3*n], "b.", markersize=2) # this is the artist to be updated
    current_graph, = ax2.plot(voltage_hist, current_histroy)
    ax2.set_xlim(0.,tspan[-1])
    ax2.set_ylabel("Current I")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    #ax2.set_ylim(0.,max(current_histroy))
    fig.savefig(f"STATES-m={params['m']}-pin={wpa}-eta={params['eta']}-V={params['V']}-lambda={params['lambda']}"+".png")
    print("Simulation complete.")
    return t, states, times, current_histroy, voltage_hist

