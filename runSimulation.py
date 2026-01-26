import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from scipy.integrate import solve_ivp
#import imageio

from calculateForces import integrable_calcualte_forces


def run_simulation(params):
    """
    Translated from MATLAB runSimulation.m
    Runs the full dendritic growth simulation with visualization and GIF output.
    """

    # Initialize globals (same as MATLAB)
    from calculateForces import I_saved, t_saved
    I_saved.clear()
    t_saved.clear()

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

    else: #Use Velocity-Verlet to integrate
        # we write the velocity verlet function here and include the ability to stream the simualtion directly into a file. Thus we don't need to wait at the end for the animation to compile
        error_tol = params['Verlet_error_per_unit_time']    
        from matplotlib.animation import FFMpegWriter # for streaming the frames into  a file
        # recall :
        #fun=integrable_calcualte_forces
        #t_span=(tspan[0], tspan[-1])
        #y0=initial
        #t_eval=tspan
        t_end = tspan[-1]
        times = np.zeros(1,float)
        time_step = tspan[1] - tspan[0]

        while t < t_end: # ideallty make this variable time step at some point, for now this will work
             # actually accelerations, not forces, divided by eta
            if time_step > t_end - t: # do not exceed the total time
                time_step = t_end - t
            
            #Full step
            initial_FS = velocity_verlet_step(t,initial,n,time_step)

            #Two half steps
            initial_1HS = velocity_verlet_step(t,initial,n,time_step/2.)
            initial_2HS = velocity_verlet_step(t+time_step/2.,initial_1HS,n,time_step/2.)

            err = np.linalg.norm(initial_FS - initial_2HS) # error approximation

            if err < error_tol: # accept the result
                np.append(times,t) # record time history
                t += time_step # update time 
                initial = initial_2HS # update phase space coordiantes

                # Update timestep
                if err < error_tol/2.: # to increase efficiency, if the error is sugnifigantly less than the error tolerance then we can increase the time step
                    time_step *= 2.
            else:
                # Reject step -> reduce time step
                time_step *= 0.8 * (error_tol / err)**(1/3) # use error formula for the vel_verlet algorithm to choose a new time step to try
    
    return t, states

def velocity_verlet_step(t,initial,n,time_step):
    # Change in positions
        initial_copy = np.copy(initial)
        force_vec = integrable_calcualte_forces(t,initial)
        initial_copy[0:n] += initial[n:(2*n)]*time_step + 0.5 * time_step**2 * force_vec[0:n] # change in x coordinate
        initial_copy[(2*n):(3*n)] += initial[(3*n):(4*n)]*time_step + 0.5 * time_step**2 * force_vec[(2*n):(3*n)] # change in y coordinate

        # change in velocities: need to compute acceleration at the next step
        force_vec_next = integrable_calcualte_forces(t,initial) # actually accelerations, not forces, divided by eta technically because of the inclusion of the drag force, this is not exact, but it's hopefully close enough
        initial_copy[n:(2*n)] += 0.5*(force_vec[n:(2*n)] + force_vec_next[n:(2*n)] ) # change in x velocity
        initial_copy[(3*n):(4*n)] += 0.5*(force_vec[(3*n):(4*n)] + force_vec_next[(3*n):(4*n)] )  # change in y velocity
        initial_copy[4*n] = force_vec[4*n] # tempurature evolution

        return initial_copy
