"""
Depreciated - replaced with velocity verlet
"""

import numpy as np

nmax = 8  # max number of steps in B-S before halving interval

# initialize lists to store the final lists of times and values
tpoints = []
xpoints = []


def step(ode, x, t, H, delta):
    """ adaptive Bulirsch-Stoer integration
     ode is the ODE function dxdt = f(x), x is the state vector,
      t is time,"""
    # Do one modified midpoint step to get things started
    x1 = x + 0.5*H*ode(t,x)
    x2 = x + H*ode(t+0.5*H,x1)

    if type(x) in (int,float,np.float32,np.float64):
        l = 1
    else:
        l = len(x)

    R1 = np.empty([1, l], float)
    R1[0] = 0.5*(x1 + x2 + 0.5*H*ode(t+0.5*H,x2))

    # Now increase n until the required accuracy is reached
    for q in range(2, nmax+1):
        h = H/float(q)
        x1 = x + 0.5*h*ode(t,x)  # Modified midpoint method 
        x2 = x + h*ode(t+0.5*h,x1)
        for i in range(q-1):
            x1 += h*ode(t+h*i,x2)
            x2 += h*ode(t+h*(i+0.5),x1)

        # Calculate extrapolation estimates.  Arrays R1 and R2
        # hold the two most recent lines of the table
        R2 = R1
        R1 = np.empty([q, l], float)
        R1[0] = 0.5*(x1 + x2 + 0.5*h*ode(t+H,x2))
        for m in range(1, q):
            epsilon = (R1[m-1]-R2[m-1])/((float(q)/float(q-1))**(2*m)-1)
            R1[m] = R1[m-1] + epsilon
        error = abs(epsilon[0])

        if error < H*delta:  # Check for convergence
            result = R1[q-1]
            tpoints.append(t+H)
            xpoints.append(result)
            return result, tpoints, xpoints
        
    # If convergence failed, do it again with smaller steps
    x1 = step(ode, x, t, H/2,delta)[0]
    x2 = step(ode, x1, t+H/2, H/2,delta)[0]

    return x2,tpoints,xpoints

if __name__ == "__main__": # fidelity test for integrator
    import matplotlib.pyplot as plt
    test_ode = lambda t,x: np.array([np.sin(x[1]*t),t])
    _, t,x = step(test_ode,[1,0],0,5,1e-5)
    print(t)
    print(x)
    plt.plot(t,x)
    plt.show()