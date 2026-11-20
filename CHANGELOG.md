# Changelog

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
