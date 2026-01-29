from calculateForces import vdW_Force_AND_Dipole_Force
from scipy.integrate import RK45
from defineParameters import params
from numpy import array, column_stack, append,linspace
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter 

R = params["Average_particle_Radius"]

x = array([0.,4.*R])
y = array([0.,4.*R])
initial = append(x,y)

tspan = linspace(0.,10.,100)

def force(t,state):
    coord = column_stack((state[0:len(state)//2],state[len(state)//2:len(state)]))
    tree = cKDTree(coord)
    out=vdW_Force_AND_Dipole_Force(coord, tree, R=params["Average_particle_Radius"], A=params["Hamaker Constant"], Ex=params["V"]/params["L_x"], Ey=0., eps_r=150.)
    return append(out[0],out[1])

solver = RK45(
        fun=force,
        t0=0.,
        y0=initial,
        t_bound=tspan[-1],
        rtol=1e-9,
        atol=1e-9,
        )

current_time_indicator = 1
times = [0.]

fig = plt.figure()

plt.ylim(0.,4.*R)
plt.xlim(0.,4.*R)

particles, = plt.plot(x,y,ls='none',marker='.')
writer = FFMpegWriter(fps=5)

t = 0.
dt = 1e-3

with writer.saving(fig, "test-vdw.mp4", dpi=100):
    while solver.status == "running": # run the solver
        solver.step() # one step
        t = solver.t # current time
        y = solver.y # current state vector
        # now that everything is updated, we need to stream this data into an animation file
        if t >= tspan[current_time_indicator]: 
            current_time_indicator += 1
            particles.set_data(y[0:len(y)//2],y[len(y)//2:len(y)])
            # get the electric current data
            writer.grab_frame() # write the most recent frame to the file
        print(f"time: {t}") # debug tool

