from calculateForces import vdW_Force_AND_Dipole_Force
from scipy.integrate import solve_ivp
from defineParameters import params
from numpy import array, column_stack
from scipy.spatial import cKDTree

R = params["Average_particle_Radius"]

x = array([0.,4.*R])
y = array([0.,4.*R])

coords = column_stack(x,y)
Tree = cKDTree(coords)

force = lambda coords,tree: vdW_Force_AND_Dipole_Force(coords, tree, R=params["Average_particle_Radius"], A=params["Hamaker Constant"], Ex=params["V"]/params["L_x"], Ey=0., eps_r=150.) # input parameters

sol = solve_ivp(coords,Tree)

