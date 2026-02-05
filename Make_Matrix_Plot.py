
from defineParameters import params
from runSimulation import run_simulation
from numpy import array


matrix_Ham = array([]) # 3
matrix_viscMult = array([]) # 3

for i in range(3):
    for j in range(3):
        # Change paramaters
        params["Hamaker Constant"] = matrix_Ham[i]
        params["viscosity_multiplier"] = matrix_viscMult[j]
        # Run simulation
        t, states = run_simulation(params)
        print(f"Simulation finished successfully")
        print(f"Final time: {t:.3f}")
        print(f"State vector shape: {states.shape}")