# Dendritic Growth Simulation

A Python-based simulator for studying dendritic filament formation in a metal-oxide memristor. The model evolves hundreds of mobile ions across a 2D domain with pinning sites, computes the induced current, and exports an animation illustrating how filaments bridge the electrodes over time. This is based off of Sam's MATLAB simulation from before I took over the project. The MATLAB version is since depreciated.

---
The below information is not yet updated from the MATLAB version
---

## Simulation Overview
- **Time integration:** `ode45` integrates coupled position/velocity ODEs for each particle plus a bulk temperature state.
- **Forces:** Applied electric field, drag, interfacial confinement, random thermal kicks, residual stress, and position-dependent pinning potentials.
- **Current estimation:** A kinetic Monte Carlo style hop model (`calculateCurrent.m`) tracks electron motion from the left electrode to the right electrode as dendrites grow.
- **Visualization:** `runSimulation.m` animates particle positions, pinning potentials, and the evolving current trace, exporting frames to an animated GIF.

## Quick Start
1. **Environment:** MATLAB R2020a or newer (tested in the standard desktop environment). No toolboxes beyond base MATLAB are required.
2. **Open the project:** Launch MATLAB and set the working directory to the repository root (`DendriticGrowthSimulation`).
3. **Run the entry script:** Execute `defineParameters` from the command window or editor. This script seeds the RNG, assembles the parameter struct, and calls `runSimulation`.

```matlab
>> defineParameters
```

The solver prints progress updates as the ODE integration proceeds. When it finishes, an animation such as `dendrite_growthNEW1.gif` appears in the project folder.

## Repository Structure
- `defineParameters.m` – master setup script; constructs the `params` struct and starts the simulation.
- `runSimulation.m` – integrates the ODEs, updates plots in real time, and writes the GIF.
- `calculateForces.m` – core force-balance model evaluated by `ode45`.
- `calculateCurrent.m` – electron hopping model that translates dendrite geometry into a current estimate.
- `getNextIndex.m` – helper for stochastic electron stepping.
- `plots/`, `dendrite_*.gif` – sample output and saved animations.

Older experiments live in the `old/` directory for reference.

## Key Parameters to Tune
All parameters sit in `defineParameters.m`; adjust them before re-running.

- **Domain:** `params.Lx`, `params.Ly` control the device size; electrode width/height derive from these values.
- **Population:** `params.n` sets the number of mobile ions; denser systems require smaller ODE step sizes for stability.
- **Electric bias:** `params.V` determines the applied voltage; the script automatically drops the voltage to zero after 1000 time steps to illustrate reset behavior.
- **Pinning landscape:** `params.w_pin`, `params.R_pin`, and the random pin coordinates (`params.x_pin`, `params.y_pin`) shape nucleation hot spots.
- **Thermal noise:** Modify `params.T_coeff` and `params.To` to explore temperature-driven diffusion.
- **Current probe:** `params.num_e`, `params.lambda`, `params.Rt`, `params.steps` alter the electron hopping statistics.

Tip: keep changes modest and re-run to observe qualitative trends in filament formation and current.

## Output Files
- **Animation:** `dendrite_growthNEW<i>.gif` shows particle motion, pinning potential, and the current curve.
- **Console logs:** Integration progress and current samples print to the MATLAB command window via `disp`.

Delete or rename existing GIFs before re-running if you want to preserve prior results; the script currently overwrites files with matching names.

## Troubleshooting
- **Integration warnings:** If MATLAB reports convergence issues, reduce `params.n`, shorten `params.tspan`, or tighten `ode45` tolerances inside `runSimulation.m`.
- **Static particles:** Confirm `params.V > 0` and that the voltage is not being zeroed too early (see the block in `calculateForces.m` that toggles `V`).
- **Zero current:** The hopping model only computes current when ions approach the right electrode (`threshold_dist` in `calculateCurrent.m`). Increase the run time or lower the threshold to force a calculation.

## Next Steps
Consider adding parameter sweeps, exporting data (`X_pos`, `Y_pos`, `I_saved`) for analysis, or integrating the solver with experimental datasets for calibration.
