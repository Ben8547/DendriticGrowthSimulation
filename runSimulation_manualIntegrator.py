from numpy import random, zeros, meshgrid, linspace, concatenate, array, exp,average, inf
#from math import exp
#from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from defineParameters import params
from ManualRK4 import rk45
from calculateForces import calculate_forces, ind_func
from viscosity_field import viscocity_gradient


def run_simulation(params, Hamaker, Visc):
    """
    Translated from MATLAB runSimulation.m
    Runs the full dendritic growth simulation with visualization and GIF output.
    """

    '''# Initialize globals (same as MATLAB)
    from calculateForces import I_saved, t_saved
    I_saved.clear()
    t_saved.clear()'''

    integrable_calcualte_forces = lambda t,x: calculate_forces(t,x,params,Hamaker,Visc) # this function only takes t and x so it can be used by an integrator

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
    x = linspace(0.,params["L_x"],1000)
    y = linspace(-params["L_y"]/2.,params["L_y"]/2.,1000)

    X, Y = meshgrid(x,y)

    Z = viscocity_gradient(X,Y,ind_func,Visc) #ind_func(X,Y)
    #pcm = ax1.pcolormesh(X, Y, Z, cmap="pink", shading="auto", alpha=0.3)
    pcm = ax1.pcolormesh(x, y, Z, cmap="pink", shading="auto", alpha=0.3)

    #ax1.imshow(Z)
    plt.colorbar(pcm, ax=ax1,)
    #pcm.set_clim(vmin=0.,vmax=params["eta"])

    particles, = ax1.plot(initial[0:n], initial[2*n : 3*n], "b.", markersize=2) # this is the artist to be updated

    # graph the current

    current_histroy = [0.] # initialize the current history

    current_graph, = ax2.plot([0.], current_histroy)
    ax2.set_ylim(0.,2e-4)
    ax2.set_xlim(0.,tspan[-1])
    ax2.set_ylabel("Current I")
    ax2.set_xlabel("Time")
    ax2.grid(True)


    # now run the simulation
    from calculateCurrent import calculate_current
    t = tspan[0]
    current_time_indicator = 1 # the next time at which to capture the animation frame; index of the tspan array

    with writer.saving(fig, f"dendrite_growth_simulation-pulses-newcurrent-{params['num_e']}_electrons-visc-{Visc}_vdW-{Hamaker}_Ly-{params["L_y"]}-{params['n']}particles-manualIntegrator"+'.mp4', dpi=50): # this enables streaming the simulation data to a file concurrent with the simualtion - enables an approximate 5x speed up
        # use scipy integrator
        solver = rk45(
        f=integrable_calcualte_forces,
        t0=tspan[0],
        y0=initial,
        t_end=tspan[-1],
        rtol=params['rtol'],
        atol=params['atol'],
        dt_max=min(tspan[1]-tspan[0]) # enables the animation frames to be evenly spaced
        )
        current_time_indicator = 1
        times = [0.]
        while solver.status == "running": # run the solver
            solver.step() # one step
            t = solver.t # current time
            y = solver.y # current state vector
            # now that everything is updated, we need to stream this data into an animation file
            if t >= tspan[current_time_indicator]: 
                current_time_indicator += 1
                particles.set_data(y[0:n], y[2*n:3*n])
                # get the electric current data
                if not params['is_Voltage_constant']: # set the voltage
                    Volt = params["V_func"](t)
                else:
                    Volt = V
                times.append(t) # add to time history
                if Volt == 0.: current_histroy.append(0.) # this should save a lot of time
                else:
                    current_histroy.append(calculate_current(
                    y[0:n], y[2*n:3*n], params["L_x"],params["L_y"], Volt,
                    params["lambda"], params["Rt"], y[-1], params['num_e']#params["steps"], params["num_e"]
                    ))
                current_graph.set_data(times,current_histroy)
                ax2.set_ylim(0.,max(current_histroy))
                writer.grab_frame() # write the most recent frame to the file
                print(f"time: {t}, current {current_histroy[-1]}") # debug tool
        states = solver.y
        t = solver.t

        # now we have completed the integration
    # now the file writer is closed
    print("Simulation complete.")
    return t, states

# these functions below are no longer in use
"""def RKF45_step(t,initial,dt,err_tol=1e-3):

    # Change in positions
        initial_copy = copy(initial) # as to not mutate the original array
        k1 = dt*integrable_calcualte_forces(t,initial_copy)
        k2 = dt*integrable_calcualte_forces(t+0.25*dt, initial_copy + 0.25*k1)
        k3 = dt*integrable_calcualte_forces(t+(3./8.)*dt, initial_copy+(3./32.)*k1 + (9./32.)*k2)
        k4 = dt*integrable_calcualte_forces(t+(12./13.)*dt, initial_copy+(1932./2197.)*k1 + (7200./2197.)*k2 + (7296./2197.)*k3)
        k5 = dt*integrable_calcualte_forces(t+dt, initial_copy + (439./216.) * k1 - 8.*k2 + (5380./513.)*k3 - (845./4104.)*k4)
        k6 = dt*integrable_calcualte_forces(t + 0.5*dt, initial_copy - (8./27.)*k1 + 2.*k2 - (3544./2565.)*k3 + (1859./4104.)*k4 - (11./40.)*k5 )

        rk4 = initial_copy + (25./216.)*k1 + (1408./2565.)*k3 + (2197./4104.)*k4 - 0.2*k5 # RK4 approx
        rk5 =  initial_copy + (16./135.)*k1 + (6656./12825.)*k3 + (28561./56430.)*k4 - (9./50.)*k5 + (2./55.)*k6 # rk5 approx

        scale = 1.#atol + rtol * np.maximum(np.abs(rk4), np.abs(rk5))
        approx_err = linalg.norm((rk5 - rk4) / scale) / sqrt(len(initial)) # should help temper the size of the error 
        ideal_step_size = dt*(err_tol*dt/approx_err/2)**(0.25) * 0.9 # 0.9 is a safetey factor; this sets the ideal time step for the next iteration

        return rk5, approx_err, ideal_step_size

def velocity_verlet_step(t,initial,n,time_step):
    # Change in positions
        initial_copy = copy(initial)
        force_vec = integrable_calcualte_forces(t,initial)
        initial_copy[0:n] += initial[n:(2*n)]*time_step + 0.5 * time_step**2 * force_vec[n:(2*n)] # change in x coordinate
        initial_copy[(2*n):(3*n)] += initial[(3*n):(4*n)]*time_step + 0.5 * time_step**2 * force_vec[(3*n):(4*n)] # change in y coordinate

        # change in velocities: need to compute acceleration at the next step
        '''The below is not accurate because the force depends on velocity'''
        force_vec_next = integrable_calcualte_forces(t,initial_copy) # actually accelerations, not forces, divided by eta technically because of the inclusion of the drag force, this is not exact, but it's hopefully close enough
        initial_copy[n:(2*n)] += 0.5*(force_vec[n:(2*n)] + force_vec_next[n:(2*n)] )*time_step # change in x velocity
        initial_copy[(3*n):(4*n)] += 0.5*(force_vec[(3*n):(4*n)] + force_vec_next[(3*n):(4*n)] )*time_step  # change in y velocity
        initial_copy[4*n] += time_step * force_vec[4*n] # tempurature evolution

        return initial_copy"""
