# Changelog

## 2.4.0 (2026-11-21)

### Added
- Geometric yaw bias of the reeving: single continuous rope on parallel axles (5.3 deg) versus mirror-symmetric reeving with two ropes (none).
- Secant slopes of the wind-tunnel data near the steepest slope; critical speeds of the box over the fits and the secant slopes.
- Swell at the periods of the coupled roll-pendulum and pendulum-pitch modes (all paths and horizontal path); paired differences with and without vessel motion; check of the wind synthesis.

### Changed
- Rope torque scaled with the vertical boom-tip acceleration in the time-domain model.
- Figures 1(b), 2-4, 6-9 and graphical abstract revised.

## 2.3.0 (2026-11-21)

### Added
- Uncorrected data of the 20 ft container including negative yaw angles (NASA TM-X-62169, Table II-1) and an uncertainty envelope of the measured box moment: four choices of the data, fit orders 4-7.
- Bifurcation of the box moment for an oblique wind (transcritical, with a fold slightly below the critical speed).

### Changed
- Time-domain runs: burn-in of at least three and records of at least ten correlation times of the yaw response, and wind synthesised with a frequency spacing that resolves the yaw resonance (low-damping cases).
- Two-way coupled boom-tip motion evaluated directly at the synthesis frequencies, with the pendulum damping of each case; mass matrix of the vessel without the load taken about its own centre of gravity.
- Pendulum angle from the sway relative to the boom tip; interpolated wind-speed limits of the expected largest yaw.
- Figures 1-8 and graphical abstract revised.

## 2.2.0 (2026-11-20)

### Added
- Measured static yawing moments of rectangular boxes (NASA TM-X-62169: cargo containers and smooth-sided boxes, aspect ratios 1, 2.5 and 5), digitised in `wind_tunnel_data.py`.
- Sharp-edged-box moment model fitted to the 2.5:1 box data (sine series, orders 4-7); it replaces the Greenwell fit in all analyses. A wind along the axis is stabilising; the steepest destabilising slope is about 18 deg off the axis.
- Critical speed for the most destabilising wind direction; primary equilibrium followed exactly with the yaw as parameter.
- Two-way coupling of the crane vessel with the pendulum of the suspended load: coupled RAOs, coupled natural periods, comparison with the one-way model. The time-domain simulations use the coupled boom-tip motion.

### Changed
- Vessel BEM on a finer frequency grid (0.005 rad/s).
- Time factor of the BEM RAOs taken into account in the synthesis of the vessel motion.
- Figures 3-7 revised.

## 2.1.0 (2026-11-20)

Revision after an independent review of the paper.

### Added
- Gust-gradient moment: the aerodynamic damping acts on the rotation rate relative to the wind direction (strip theory, frozen turbulence) in the spectral and time-domain models.
- Pendulum damping (baseline 2 % of critical) and a linear aerodynamic side force on the payload; sensitivity to the pendulum damping.
- Lift-off curves of the turbulent response (height-dependent turbulence length scale).
- Expected largest yaw in a 20-min window (narrow-band peak factor).
- Worst wind direction of the bluff-body fit (fold of the primary equilibrium) and zero crossings of the aerodynamic stiffness.
- Principal parametric resonance (Floquet) and time-domain swell test at twice the yaw frequency.
- First-order actuator lag of the tugger loop.
- Finer frequency grid of the vessel BEM computation.

### Changed
- Figures 3(b), 4(b), 5(b), 6 and 7(c) revised.

## 2.0.0 (2026-11-06)

Code for the revised paper. The analysis was rewritten around a reduced yaw model and closed-form results; the earlier six-degree-of-freedom plant is kept for verification.

### Added
- Exact restoring moment of a multi-fall suspension for arbitrary fall arrangement and tension distribution, with an independent three-dimensional equilibrium check and sheave friction (`s1_suspension.py`).
- Boundary-element added masses in an unbounded fluid for the actual payload shape, with verification against Lamb's exact values (`s2_aero_bem.py`).
- Sharp-edged-box yaw-moment model based on wind-tunnel data, and a suspension-axis offset term (`aero_models.py`).
- Equilibrium branches with the complete moment balance (rope torque, wind misalignment) and their stability (`s3_statics.py`).
- Crane-vessel hydrodynamics with the main dimensions and crane data of a real heavy-lift crane vessel, and boom-tip motions (`s4_vessel.py`).
- Spectral and time-domain yaw response; vessel coupling paths; Floquet analysis (`s5_dynamics.py`, `s5b_checks.py`).
- Closed-loop threshold from eigenvalues and tugger saturation (`s6_control.py`).
- Comparison of the reduced model with the six-degree-of-freedom model (`s7_verify_6dof.py`).
- Verification tests (`tests/test_key_results.py`) and a single entry point (`run_all.py`).

### Changed
- The fall length at mating is derived from the published lift height of the crane (13 m instead of 100 m).
- The hoist rope is 76 mm with 12 falls (safety factor 7.6) instead of 120 mm.
- Tugger lines react against the crane boom.
- The yaw excitation uses the lateral turbulence spectrum; the turbulence intensity is stated explicitly (0.06 and 0.12).

### Removed
- State estimation (MEKF) and Sobol sensitivity studies, which are not part of the revised paper.

## 1.0.1

Earlier version of the solver (six-degree-of-freedom plant, Monte Carlo campaign, MEKF and Sobol studies).
