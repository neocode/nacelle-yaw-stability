# Legacy six-degree-of-freedom model

Rigid-body model of the payload on an elastic multi-fall hoist (quaternion attitude, NED frame), from version 1 of this code. In version 2 it is used only to verify the reduced yaw model (`src/s7_verify_6dof.py`, Section 7.2 of the paper). Its parameter set (120 mm rope, idealised fall circle, L = 100 m) is the version-1 baseline and differs from the baseline of the paper; some docstrings reflect interpretations of version 1 that the paper revises.
