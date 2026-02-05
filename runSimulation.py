from numpy import random, zeros, meshgrid, linspace, concatenate, where
#from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
#import imageio

from calculateForces import integrable_calcualte_forces, ind_func


def run_simulation(params):
    """
    Translated from MATLAB runSimulation.m
    Runs the full dendritic growth simulation with visualization and GIF output.
    """

    '''# Initialize globals (same as MATLAB)
    from calculateForces import I_saved, t_saved
    I_saved.clear()
    t_saved.clear()'''

    random.seed(1)

    # ------------------------
    # Unpack parameters
    # ------------------------
    n = params["n"]
    To = params["T_0"]
    tspan = params["tspan"]

    Lx = params["L_x"]
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

    Z = ind_func(X,Y)
    pcm = ax1.pcolormesh(X, Y, Z, cmap="pink", shading="auto", alpha=0.3)

    ax1.imshow(Z)
    plt.colorbar(pcm, ax=ax1)

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
    from scipy.integrate import RK45, BDF, Radau # various integrators to try

    with writer.saving(fig, f"Sim_Out_Ham={params["Hamaker Constant"]}-DensityMult={params["viscosity_multiplier"]}.mp4", dpi=100): # this enables streaming the simulation data to a file concurrent with the simualtion - enables an approximate 5x speed up
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
            '''if not params['structual_in_force']: # zero out forward momentum in the voids if no particles are behind - forward movement cannot be supported in the void without existing protrusion
                x_p = y[0:n]
                y_p = y[2*n:3*n]
                inside = (x_p < Lx)
                is_void = (ind_func(x_p, y_p) == 1)[0]
                void_indices = where(inside & is_void)[0]
                # The idea is that particles should not be able to move forward in a void if there are no particles behind them - to mimmic this we zero the momenta here'''
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
