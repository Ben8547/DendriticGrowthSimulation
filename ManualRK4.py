def rk45(f, t0, y0, t_end, dt_initial,
                  rtol=1e-6, atol=1e-9,
                  dt_min=1e-12, dt_max=1e6,
                  max_steps=10_000_000):
    """
    Pure Python, high-performance adaptive Dormand-Prince RK45.

    f(t, y, out) must write dy/dt into out (no allocations).
    """

    n = len(y0)
    y = y0[:] # copy of y0
    t = t0
    dt = dt_initial

    # Work buffers
    k1 = [0.0]*n; k2 = [0.0]*n; k3 = [0.0]*n
    k4 = [0.0]*n; k5 = [0.0]*n; k6 = [0.0]*n; k7 = [0.0]*n
    y_temp = [0.0]*n

    # Dormand–Prince coefficients
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a73, a74, a75, a76 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

    # Error coefficients
    e1, e3, e4 = 71/57600, -71/16695, 71/1920
    e5, e6, e7 = -17253/339200, 22/525, -1/40

    safety = 0.9

    prev_err = 1.0

    for _ in range(max_steps):
        if t >= t_end:
            break

        if dt < dt_min:
            raise RuntimeError("Step size underflow")

        if dt > dt_max:
            dt = dt_max

        if t + dt > t_end:
            dt = t_end - t

        # --- Compute stages ---
        f(t, y, k1) # this means output f(t,y) to the vector k1

        for i in range(n):
            y_temp[i] = y[i] + dt*a21*k1[i]
        f(t + dt*0.2, y_temp, k2)

        for i in range(n):
            y_temp[i] = y[i] + dt*(a31*k1[i] + a32*k2[i])
        f(t + dt*0.3, y_temp, k3)

        for i in range(n):
            y_temp[i] = y[i] + dt*(a41*k1[i] + a42*k2[i] + a43*k3[i])
        f(t + dt*0.8, y_temp, k4)

        for i in range(n):
            y_temp[i] = y[i] + dt*(a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i])
        f(t + dt*(8/9), y_temp, k5)

        for i in range(n):
            y_temp[i] = y[i] + dt*(a61*k1[i] + a62*k2[i] + a63*k3[i] +
                                   a64*k4[i] + a65*k5[i])
        f(t + dt, y_temp, k6)

        for i in range(n):
            y_temp[i] = y[i] + dt*(a71*k1[i] + a73*k3[i] +
                                   a74*k4[i] + a75*k5[i] + a76*k6[i])
        f(t + dt, y_temp, k7)

        # --- Error norm ---
        err_norm = 0.0
        for i in range(n):
            err = dt*(e1*k1[i] + e3*k3[i] + e4*k4[i] +
                      e5*k5[i] + e6*k6[i] + e7*k7[i])
            sc = atol + rtol * max(abs(y[i]), abs(y_temp[i]))
            r = err / sc
            err_norm += r*r

        err_norm = (err_norm / n) ** 0.5

        # --- Accept / Reject ---
        if err_norm <= 1.0:
            y[:] = y_temp
            t += dt

        # --- PI step controller (much better than basic scaling) ---
        if err_norm == 0.0: # prevent divide by zero
            factor = 5.0
        else:
            factor = safety * (err_norm ** (-0.2)) * (prev_err ** 0.04)

        # Clamp growth/shrink
        if factor < 0.2:
            factor = 0.2
        elif factor > 5.0:
            factor = 5.0

        dt *= factor
        prev_err = max(err_norm, 1e-16)

    return y, t # note; only returns the last iteration; this is an efficiency consideration since we are overwirting the inforation in each step of the loop.



if __name__ == "__main__": # test efficacy

    import matplotlib.pyplot as plt
    from numpy import linspace, array, exp
    def ode(t, x, out):
        out[0] = x[0] # overwite the out vector; saves allocation time
        out[1] = x[0]*x[1]-x[1]
    
    t_end = 4.
    y, t = rk45(ode,0,array([1.,0.]),t_end,0.00001,dt_max=0.01)
    print(y)
    print(t)
    plt.plot(t,y[0])
    plt.show()
    print(exp(4))
