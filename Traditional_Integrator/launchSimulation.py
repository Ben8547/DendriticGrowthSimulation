#####################################################################
##################### RUN DENDRITIC SIMULATION ######################
#####################################################################

from time import time
from defineParameters import params
import matplotlib.pyplot as plt

def run(Hamaker, Visc, lissajous = False):
    from runSimulation import run_simulation
    start = time()
    t, states, t_hist, c_hist, v_hist = run_simulation(params, Hamaker, Visc)
    ellapsed = time() - start
    print(f"Simulation finished successfully in {ellapsed} seconds!")
    #print(f"Final time: {t[-1]:.3f}")

    # make a lissajous figure is this was asked for
    if lissajous:
        fig_lissajous, ax_lissajous = plt.subplots(figsize=(6, 6))
        ax_lissajous.plot(v_hist, c_hist, color='purple', linewidth=1.5, ls='none',marker='.')
        ax_lissajous.set_title("I-V Lissajous Figure", fontsize=14, fontweight='bold')
        ax_lissajous.set_xlabel("Voltage (V)", fontsize=12)
        ax_lissajous.set_ylabel("Current (I)", fontsize=12)
        ax_lissajous.grid(True, linestyle='--', alpha=0.7)
        ax_lissajous.axhline(0, color='black', linewidth=0.8)
        ax_lissajous.axvline(0, color='black', linewidth=0.8)

        plt.tight_layout()
        plt.savefig(f"lissajous_IV_curve_{params['num_e']}_electrons.png", dpi=300, bbox_inches='tight')


    print(f"Final time: {t:.3f}")
    print(f"State vector shape: {states.shape}")

if __name__ == "__main__": # Only run the following code if this file is being run directly, not when it’s imported by another script
    #run(params['Hamaker Constant'],params['viscosity_multiplier'])
    run(1e0,1.,lissajous=False)