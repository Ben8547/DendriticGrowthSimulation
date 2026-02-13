#%%
import sympy as sp

sp.init_printing()

A, R, d = sp.symbols("A,R,d")

F_vdW_mag =  A/6 * ( (4*R**2*d)/(d**2 -4*R**2)**2 + (4*R**2 * d)/(d**4) + (2*d)/(d**2-4*R**2) - 2/(d)  )

#sp.pretty_print(F_vdW_mag)
print(sp.latex(F_vdW_mag))
F_vdW_mag
# %% Taylor
expansion = F_vdW_mag.series(d,10*R,6)
print(sp.latex(expansion))
expansion

#%% Try a Fourier Series
'''F_expansion = sp.fourier_series(F_vdW_mag,(d,0,10*R),finite=True)
print(sp.latex(F_expansion))
F_expansion''' # useless


# THis did not work well no matter where we took the expansion about.
