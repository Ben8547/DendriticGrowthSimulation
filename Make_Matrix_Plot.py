from launchSimulation import run
from numpy import array


matrix_Ham = array([float(2**(2*i)) for i in range (3)],dtype=str) # 3
matrix_viscMult = array([2.5, 8, 16],dtype=str) # 3

for i in range(3):
    with open("Hamaker.txt",'w') as Ham: # Change paramaters
            Ham.write(matrix_Ham[i])
    for j in range(3):
        with open("Visc.txt",'w') as Visc: # Change paramaters
            Visc.write(matrix_viscMult[j])
        # Run simulation
        run()

# Now extract the last frame and stich into one PNG file in a matrix

