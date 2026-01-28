import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
#import imageio

from calculateForces import integrable_calcualte_forces


def run_simulation(params):
    """
    Translated from MATLAB runSimulation.m
    Runs the full dendritic growth simulation with visualization and GIF output.
    """

    '''# Initialize globals (same as MATLAB)
    from calculateForces import I_saved, t_saved
    I_saved.clear()
    t_saved.clear()'''

    np.random.seed(1)

    # ------------------------
    # Unpack parameters
    # ------------------------
    n = params["n"]
    To = params["T_0"]
    tspan = params["tspan"]

    Lx = params["L_x"]
    Ly = params["L_y"]
    electrode_width = params["electrode_width"]
    electrode_height = params["electrode_height"]

    # ------------------------
    # Randomize initial particle positions
    # ------------------------
    xo = 2.5 * np.random.rand(n)  - 2.5 # start behind the device - should prevent starting in voids
    yo = electrode_height * np.random.rand(n) - 0.5 * electrode_height

    # ------------------------
    # Zero all initial particle velocities
    # ------------------------
    vox = np.zeros(n)
    voy = np.zeros(n)

    # ------------------------
    # Initial state vector
    # ------------------------
    initial = np.concatenate([xo, vox, yo, voy, [To]]) # To is wrapped in a list so that concatenate resolves

    # ------------------------
    # Run ODE simulation
    # ------------------------
    print("Starting solver...")
    from calculateForces import tindex, V
    tindex = 1
    V = params["V"]

    if not params["Velocity_Verlet"]: #use scipy to integrate; this is now depreciated
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            fun=integrable_calcualte_forces,
            t_span=(tspan[0], tspan[-1]),
            y0=initial,
            t_eval=tspan, # recall that tspan is an np.arange array - gives us the integral at evenly spaced intervals of dt; these are used for the animation. Though the solver may take smaller steps to reduce error.
            method=params['scipy_tag'],
            vectorized=False,
            rtol=params['rtol'],
            atol=params['atol']
        )

        t = sol.t # times of the solution
        states = sol.y.T # the state vector at each time - should be an array; we transpose so that each row is a time and the columns are the states

        X_pos = states[:, 0:n] # all times, colums 0 to n-1 of the state vector
        Y_pos = states[:, (2*n):(3*n)] # all times, colums 2n-1 to 3n-1 of the state vector

        # ------------------------
        # Plot setup
        # ------------------------
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

        showPP = False

        if showPP:
            # ---- Pinning potential background ----
            x = np.linspace(0, Lx, 100)
            y = np.linspace(-Ly / 2, Ly / 2, 100)
            X, Y = np.meshgrid(x, y)
            U = np.zeros_like(X)
            for k in range(len(params["w_pin"])):
                dx = X - params["x_pin"][k]
                dy = Y - params["y_pin"][k]
                U += params["w_pin"][k] * np.exp(-(dx**2 + dy**2) / (params["R_pin"][k] ** 2))

            pcm = ax1.pcolormesh(X, Y, U, cmap="pink", shading="auto", alpha=0.3)
            cb = plt.colorbar(pcm, ax=ax1)
            cb.set_label("Pinning Potential U")
        else: #show voids
            from viscosity_field import make_globular_indicator
            ind_func = make_globular_indicator()

            x = np.linspace(0.,params["L_x"],1000)
            y = np.linspace(-params["L_y"]/2.,params["L_y"]/2.,1000)

            X, Y = np.meshgrid(x,y)

            Z = ind_func(X,Y)
            pcm = ax1.pcolormesh(X, Y, Z, cmap="pink", shading="auto", alpha=0.3)

            ax1.imshow(Z)
            plt.colorbar(pcm, ax=ax1)

        # ---- Initialize particle positions ----
        particles, = ax1.plot(X_pos[0, :], Y_pos[0, :], "b.", markersize=2)

        # ---- Bottom plot: current vs time ----
        ax2.set_xlim([t[0], t[-1]])
        #ax2.set_ylim([0, params["num_e"]])
        ax2.set_ylim([0, np.max(I_saved)])
        ax2.set_xlabel("time")
        ax2.set_ylabel("Current I")
        ax2.grid(True)
        current_line, = ax2.plot([], [], "r-", lw=1.5)

        # ------------------------
        # Animation update function
        # ------------------------
        def update(frame):
            particles.set_data(X_pos[frame, :], Y_pos[frame, :])

            from calculateForces import I_saved, t_saved
            if len(t_saved) > 0:
                mask = np.array(t_saved) <= t[frame]
                if np.any(mask):
                    current_line.set_data(np.array(t_saved)[mask], np.array(I_saved)[mask])

            ax1.set_title(f"t = {t[frame]:.2f}")
            return particles, current_line

        # ------------------------
        # Create and save animation
        # ------------------------
        anim = FuncAnimation(fig, update, frames=len(t), interval=100, blit=False)

        filename = params["filename"]
        print(f"Saving GIF to {filename}...")
        writer = PillowWriter(fps=60) #'imagemagick'
        anim.save(filename, writer=writer)#, fps=60)
        #plt.show()

        plt.close(fig)
        print("Simulation complete. GIF saved.")

    else: #Use Velocity-Verlet to integrate; Update: I'm changing this to RKF45 since the Verlet method had too small time steps. I think this is because the verlet integrator requires the use of the next force vector, but we can only approximate this in the current scheme because the force depends upon the velocities (drag force). I will not be changing the variable labels, so keep that in mind if something mentions the verlet alorithm.
        # we write the velocity verlet function here and include the ability to stream the simualtion directly into a file. Thus we don't need to wait at the end for the animation to compile
        error_tol = params['Verlet_error_per_unit_time']  
        min_step = params["min_step_size"]  
        from matplotlib.animation import FFMpegWriter # for streaming the frames into  a file
        # recall :
        #fun=integrable_calcualte_forces
        #t_span=(tspan[0], tspan[-1])
        #y0=initial
        #t_eval=tspan
        t_end = tspan[-1]
        times = np.zeros(1,float)
        time_step = tspan[1] - tspan[0]

        # set up the animation figrue and artists
        writer = FFMpegWriter(fps=30)
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

        from viscosity_field import make_globular_indicator
        ind_func = make_globular_indicator()

        x = np.linspace(0.,params["L_x"],1000)
        y = np.linspace(-params["L_y"]/2.,params["L_y"]/2.,1000)

        X, Y = np.meshgrid(x,y)

        Z = ind_func(X,Y)
        pcm = ax1.pcolormesh(X, Y, Z, cmap="pink", shading="auto", alpha=0.3)

        ax1.imshow(Z)
        plt.colorbar(pcm, ax=ax1)

        particles, = ax1.plot(initial[0:n], initial[2*n : 3*n], "b.", markersize=2) # this is the artist to be updated

        # deal with the current later - it would go here


        # now run the simulation
        t = tspan[0]
        current_time_indicator = 1 # the next time at which to capture the animation frame; index of the tspan array
        from scipy.integrate import RK45

        with writer.saving(fig, "test_simulation_output.mp4", dpi=100):
            """while t < t_end: # variable time step elocity verlet integrator
                print(f"{t}, {time_step}") # debug
                # actually accelerations, not forces, divided by eta
                if time_step > t_end - t: # do not exceed the total time
                    time_step = t_end - t
                
                '''#Full step
                initial_FS = (t,initial,n,time_step)

                #Two half steps
                initial_1HS = velocity_verlet_step(t,initial,n,time_step/2.)
                initial_2HS = velocity_verlet_step(t+time_step/2.,initial_1HS,n,time_step/2.)
                err = np.linalg.norm(initial_FS - initial_2HS) # error approximation'''#step variable verlet

                proposed, step_error, ideal_step = RKF45_step(t,initial,time_step, error_tol)

                if step_error < error_tol * time_step: # accept the result; error per unit time
                    times = np.append(times,t) # record time history
                    t += time_step # update time 
                    initial = proposed # update phase space coordiantes

                    # Update timestep
                    time_step = 2.*time_step if step_error < error_tol * time_step/4. else ideal_step # increase the step a lot if the error is much smaller than the ideal error
                    # now that everything is updated, we need to stream this data into an animation file
                    if t >= tspan[current_time_indicator]:# restrict the frames to grab - takes too long and files are too big if we capture every frame
                        current_time_indicator += 1 # increase the indicator - only record once in each window of the tspan array
                        particles.set_data(initial[0:n], initial[2*n : 3*n]) # update the artist - change particle positions in the image
                        writer.grab_frame() # stream the new image to the file

                else:
                    # Reject step -> reduce time step
                    time_step = max(0.5 * time_step, ideal_step)""" # manual RKF45 integrator - too slow
            # use scipy integrator instead
            solver = RK45(
            fun=integrable_calcualte_forces,
            t0=tspan[0],
            y0=initial,
            t_bound=tspan[-1],
            rtol=rtol,
            atol=atol,
            )
            current_time_indicator = 1
            while solver.status == "running": # run the solver
                solver.step() # one step
                t = solver.t # current time
                y = solver.y # current state vector
                print(f"{t}")
                # now that everything is updated, we need to stream this data into an animation file
                if t >= tspan[current_time_indicator]: 
                    current_time_indicator += 1
                    particles.set_data(y[0:n], y[2*n:3*n])
                    writer.grab_frame()
            states = solver.y
            t = solver.t

            # now we have completed the integration
        # now the file writer is closed
    print("Simulation complete.")

    return t, states

from defineParameters import params

atol = params['rtol']
rtol = params["atol"]


# these functions below are no longer in use
def RKF45_step(t,initial,dt,err_tol):

    # Change in positions
        initial_copy = np.copy(initial) # as to not mutate the original array
        k1 = dt*integrable_calcualte_forces(t,initial_copy)
        k2 = dt*integrable_calcualte_forces(t+0.25*dt, initial_copy + 0.25*k1)
        k3 = dt*integrable_calcualte_forces(t+(3./8.)*dt, initial_copy+(3./32.)*k1 + (9./32.)*k2)
        k4 = dt*integrable_calcualte_forces(t+(12./13.)*dt, initial_copy+(1932./2197.)*k1 + (7200./2197.)*k2 + (7296./2197.)*k3)
        k5 = dt*integrable_calcualte_forces(t+dt, initial_copy + (439./216.) * k1 - 8.*k2 + (5380./513.)*k3 - (845./4104.)*k4)
        k6 = dt*integrable_calcualte_forces(t + 0.5*dt, initial_copy - (8./27.)*k1 + 2.*k2 - (3544./2565.)*k3 + (1859./4104.)*k4 - (11./40.)*k5 )

        rk4 = initial_copy + (25./216.)*k1 + (1408./2565.)*k3 + (2197./4104.)*k4 - 0.2*k5 # RK4 approx
        rk5 =  initial_copy + (16./135.)*k1 + (6656./12825.)*k3 + (28561./56430.)*k4 - (9./50.)*k5 + (2./55.)*k6 # rk5 approx

        scale = atol + rtol * np.maximum(np.abs(rk4), np.abs(rk5))
        approx_err = np.linalg.norm((rk5 - rk4) / scale) / np.sqrt(len(initial)) # should help temper the size of the error 
        ideal_step_size = dt*(err_tol*dt/approx_err/2)**(0.25) * 0.9 # 0.9 is a safetey factor; this sets the ideal time step for the next iteration

        return rk5, approx_err, ideal_step_size

def velocity_verlet_step(t,initial,n,time_step):
    # Change in positions
        initial_copy = np.copy(initial)
        force_vec = integrable_calcualte_forces(t,initial)
        initial_copy[0:n] += initial[n:(2*n)]*time_step + 0.5 * time_step**2 * force_vec[n:(2*n)] # change in x coordinate
        initial_copy[(2*n):(3*n)] += initial[(3*n):(4*n)]*time_step + 0.5 * time_step**2 * force_vec[(3*n):(4*n)] # change in y coordinate

        # change in velocities: need to compute acceleration at the next step
        '''The below is not accurate because the force depends on velocity'''
        force_vec_next = integrable_calcualte_forces(t,initial_copy) # actually accelerations, not forces, divided by eta technically because of the inclusion of the drag force, this is not exact, but it's hopefully close enough
        initial_copy[n:(2*n)] += 0.5*(force_vec[n:(2*n)] + force_vec_next[n:(2*n)] )*time_step # change in x velocity
        initial_copy[(3*n):(4*n)] += 0.5*(force_vec[(3*n):(4*n)] + force_vec_next[(3*n):(4*n)] )*time_step  # change in y velocity
        initial_copy[4*n] += time_step * force_vec[4*n] # tempurature evolution

        return initial_copy
