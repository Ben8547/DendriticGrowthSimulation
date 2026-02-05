
from defineParameters import params
from runSimulation import run_simulation
from numpy import array


matrix_Ham = array([]) # 3
matrix_viscMult = array([]) # 3

for i in range(3):
    with open("Hamaker.txt",'w') as Ham:
            Ham.write(matrix_Ham[i])
    for j in range(3):
        with open("Visc.txt",'w') as Visc:
            Visc.write(matrix_viscMult[j])
        # Change paramaters
        # Run simulation
        t, states = run_simulation(params)
        print(f"Simulation finished successfully")
        print(f"Final time: {t:.3f}")
        print(f"State vector shape: {states.shape}")