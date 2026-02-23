# nonlinear-heat

bayesian inference for thermal wave propagation experiments using peltier heat pumps.

four thermal wave models with:
1. **SimpleForward**: single propagating wave (high-frequency, heavily-damped limit)
2. **FullSolution**: finite rod with adiabatic boundary at x=L
3. **FullSolutionNewton**: finite rod with newton cooling (heat loss)
4. **FullSolutionRobin**: finite rod with robin boundary conditions (convective loss at tip)

shares **global parameters** (diffusivity, newton's cooling constant, ambient temp.) across all frequencies. learns **frequency-specific** parameters (ampltiude, phase) for each dataset. using all eight sensors simultaneously.

experiments at different driving frequencies; `al_5s.csv`, `al_10s.csv`, ..., `al_70s.csv` (period in seconds). 8 thermistors at positions: [3, 8, 13, 18, 23, 28, 33, 43] mm