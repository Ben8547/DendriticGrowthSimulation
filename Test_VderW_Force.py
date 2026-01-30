from calculateForces import vdW_Force_AND_Dipole_Force
from scipy.integrate import RK45, Radau, BDF
from defineParameters import params
from numpy import array, column_stack, append,linspace, random, zeros_like, copy
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter 

R = 0.002

#x = array([0.,4.*R])
#y = array([0.,4.*R])
#x = array([0.,6.*R,8.*R])
#y = array([0.,6.*R,-R])
random.seed(10)
x = 10. * (random.random(2)-0.5) * R
y = 10. * (random.random(2)-0.5) * R
vx = zeros_like(x)
vy = copy(vx)

initial = append(x,append(y,append(vx,vy)))

tspan = linspace(0.,10.,1000)

def force(t,state):
    coord = column_stack((state[0:len(state)//4],state[len(state)//4:len(state)//2]))
    tree = cKDTree(coord)
    out=vdW_Force_AND_Dipole_Force(coord, tree, R=R, A=0.00001, Ex=params["V"]/params["L_x"], Ey=0., eps_r=150.)
    return append(state[len(state)//2:3*len(state)//4],append(state[3*len(state)//4:len(state)],append(out[0],out[1])))

solver = RK45(
        fun=force,
        t0=0.,
        y0=initial,
        t_bound=tspan[-1],
        max_step= 10./1000.,
        rtol=1e-4,
        atol=1e-4,
        )

current_time_indicator = 1
times = [0.]

fig, ax = plt.subplots()

ax.set_ylim(min(y),max(y))
ax.set_xlim(min(x),max(x))

particles, = plt.plot(x,y,ls='none',marker='.')
writer = FFMpegWriter(fps=20)

with writer.saving(fig, "test-vdw.mp4", dpi=100):
    while solver.status == "running": # run the solver
        solver.step() # one step
        t = solver.t # current time
        y = solver.y # current state vector
        # now that everything is updated, we need to stream this data into an animation file
        if t >= tspan[current_time_indicator]: 
            current_time_indicator += 1
            particles.set_data(x1:=y[0:len(y)//4],y1:=y[len(y)//4:len(y)//2])
            # Calculate new limits with 10% padding
            x_min, x_max = min(x1), max(x1)
            y_min, y_max = min(y1), max(y1)
            
            # Add a small buffer so particles aren't on the edge
            pad_x = (x_max - x_min) * 0.1
            pad_y = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
            # get the electric current data
            writer.grab_frame() # write the most recent frame to the file
        print(f"time: {t}") # debug tool

