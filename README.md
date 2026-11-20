# nacelle-yaw-stability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24403652.svg)](https://doi.org/10.5281/zenodo.24403652)

Code, data and figures for the paper

> S. Podliesnyi, O. Sheremet. *Yaw stability of a nacelle suspended from a multi-fall offshore crane hoist: gravitational torsional stiffness and wind-induced static divergence.*

Every number, table and figure of the paper is produced by the scripts in `src/`. The computed results are stored in `results/`, and the figures in `figures/` (PNG and LZW-compressed TIFF, 600 dpi).

## What the code computes

| Script | Paper section | Content | Output |
|---|---|---|---|
| `src/common.py` | 2.2, Table 1 | baseline parameters (IEA 15 MW nacelle, hoist reeving, crane vessel), derived suspension quantities, figure style | – |
| `src/s1_suspension.py` | 3, Table 2, Fig. 2 | closed-form restoring moment of a multi-fall suspension; independent 3D equilibrium of the block (Levenberg–Marquardt); sheave friction; wire-rope cabling benchmark; elastic-torsion bound | `results/s1_suspension.json` |
| `src/s2_aero_bem.py` | 4.1, Fig. 3, Fig. 9(b,c) | added masses in an unbounded fluid with Capytaine: prolate spheroid vs Lamb, superellipsoids, rectangular box; aerodynamic yaw-stiffness coefficients | `results/s2_aero_bem.json` |
| `src/aero_models.py` | 4.1 | potential-flow (Munk) and sharp-edged-box yaw-moment models; divergence speed | – |
| `src/s3_statics.py` | 4.2–4.4, Table 3, Figs. 4–5 | divergence speed along the lift; wind-heading dependence; all equilibria and their stability with the complete moment balance | `results/s3_statics.json` |
| `src/s4_vessel.py` | 5.3, Table 4, Fig. 7(a–c) | crane-vessel hydrodynamics (Capytaine), RAOs, boom-tip motions | `results/s4_vessel.json` |
| `src/s5_dynamics.py` | 5, Figs. 6, 7(d) | spectral and time-domain yaw response to turbulent wind; vessel coupling; Floquet analysis of heave-induced parametric excitation | `results/s5_dynamics.json` |
| `src/s5b_checks.py` | 5.3, 7.2 | time-step convergence; attribution of the vessel coupling paths | `results/s5b_checks.json` |
| `src/s6_control.py` | 6, Fig. 8 | closed-loop divergence speed (eigenvalues), tugger saturation, closed-loop response | `results/s6_control.json` |
| `src/s7_verify_6dof.py` | 7.2, Fig. 9(a) | reduced yaw model vs six-degree-of-freedom rigid-body model (`src/legacy_6dof/`) | `results/s7_verify_6dof.json` |
| `src/make_figures.py`, `src/fig1_schematic.py`, `src/graphical_abstract.py` | all figures | figures from the stored results | `figures/` |

## Installation

Python ≥ 3.11 is required. Install the dependencies with

```
pip install -r requirements.txt
```

The versions used for the paper are pinned in `requirements.txt`.

## Reproducing the results

```
python run_all.py            # all analyses and figures, about 10 minutes on a 24-core workstation
python run_all.py --figures  # figures only, from the stored results
pytest                       # verification tests of the key results (about 1 minute)
```

`run_all.py` runs the scripts in the order of the table above. The six-degree-of-freedom comparison (`s7_verify_6dof.py`) is the slowest step, at about 7 minutes.

## Verification tests

`tests/test_key_results.py` checks, independently of the stored results:

- the closed-form restoring moment against the three-dimensional equilibrium solution;
- the wire-rope cabling benchmark;
- the boundary-element added masses against Lamb's exact values;
- the divergence speeds quoted in the paper;
- the post-critical equilibrium of the potential-flow model;
- the closed-loop threshold against the eigenvalues of the closed loop;
- the consistency of the stored results with the values quoted in the paper.

## Parameters and their sources

All baseline values are in `src/common.py`, with their sources in comments, and are listed in Table 1 of the paper. The main dimensions of the crane vessel and the lift height of the auxiliary block are taken from the published datasheet of the heavy-lift crane vessel *Les Alizés* (Jan De Nul). The operating draught, the centre of gravity, the crane position, the hoist reeving and the rigging height are assumptions. To analyse another configuration, change these values and rerun `run_all.py`.

## Structure

```
src/                 analysis scripts
src/legacy_6dof/     six-degree-of-freedom model of the payload (earlier version of this code, used for verification)
tests/               verification tests
results/             computed results (JSON)
figures/png, tiff    figures of the paper and graphical abstract
```

## Version history

See `CHANGELOG.md`.

## Licence

MIT, see `LICENSE`. If you use this code, please cite the paper and this repository (`CITATION.cff`). Archived version 2.0.0: https://doi.org/10.5281/zenodo.24403653 (all versions: https://doi.org/10.5281/zenodo.24403652).
