"""
Section 7: verification of the reduced yaw model against the full 6-DOF
rigid-body plant of the original study (legacy_6dof/, same authors, MIT licence).

Both models are run with the ORIGINAL parameter set (d = 120 mm, idealised fall
circle S = 2.4 m, L = 100 m, Munk moment with the box volume) and identical
wind time histories:
  (i)   free yaw decay in still air from 5 deg,
  (ii)  steady wind (static equilibrium with aerodynamic amplification),
  (iii) turbulent wind, 1500 s.
"""
import sys
import os
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "legacy_6dof"))
import params as P6               # noqa: E402  (legacy)
import payload as PL              # noqa: E402
import frames as FR               # noqa: E402
import aero as AE6                # noqa: E402
sys.path.insert(0, HERE)
import common as cm               # noqa: E402
from s5_dynamics import turbulence, integrate, NP   # noqa: E402


def reduced_params(pl, zeta, U):
    """Parameter vector of the reduced model replicating the legacy plant."""
    dv = P6.derive(L=pl.L)
    CK = 2 * dv.V_box * dv.k2mk1 / (cm.A_side * cm.L_n)
    c_s = 2 * zeta * np.sqrt(pl.k_psi * float(P6.PAYLOAD["I_zz"]))
    m = pl.m
    P = np.zeros(NP)
    P[:20] = [pl.k_psi, float(P6.PAYLOAD["I_zz"]), c_s, 0.0, pl.M_rope, m,
              m * cm.g, pl.L + pl.h_att, 0.5 * cm.rho_air * cm.A_side * cm.L_n,
              0.0, 0, 0, 0, 0, CK, 0, 0, 1, 0, 0]
    P[23] = 0.5 * CK            # Munk moment c = (C_K/2) sin 2 beta (sine-series slot)
    return P


def run():
    out = {}
    L = 100.0
    zeta = 0.01
    # (i) free decay ------------------------------------------------------
    pl = PL.Payload(L=L, wind=None, zeta_struct=zeta, use_crossflow_damping=False)
    x0 = pl.equilibrium(psi=np.radians(5.0))
    t0 = time.time()
    t, X = pl.simulate(400.0, dt=0.5, x0=x0, rtol=1e-9, atol=1e-11)
    psi6 = np.degrees(pl.yaw_of(X))
    Pv = reduced_params(pl, zeta, 0.0)
    n = len(t)
    z = np.zeros(int(round(400.0 / 0.05)) + 1)
    Xr = integrate(np.array([np.radians(5.0), 0, 0, 0, 0, 0]), 0.05, 0.0, z, z, z, z, z, z, z, Pv)
    psir = np.degrees(Xr[::10, 0])[:n]
    out["free_decay"] = dict(t=t.tolist(), psi_6dof=psi6.tolist(), psi_red=psir.tolist(),
                             max_diff=float(np.max(np.abs(psi6 - psir))),
                             psi0_analytic=float(np.degrees(np.arcsin(pl.M_rope / pl.k_psi))),
                             wall=time.time() - t0)
    print("free decay max diff", out["free_decay"]["max_diff"], time.time() - t0)

    # (iii) turbulent wind ---------------------------------------------------
    U, TI, T, dt = 6.0, 0.06, 1500.0, 0.05
    u, v = turbulence(3, T, dt, U, TI, 70.0)
    tt = np.arange(len(u)) * dt

    def wind(time_, p_I):
        return np.array([U + np.interp(time_, tt, u), np.interp(time_, tt, v), 0.0])

    pl = PL.Payload(L=L, wind=wind, zeta_struct=zeta, use_crossflow_damping=False)
    # start both from the steady-wind equilibrium of the reduced model
    from scipy.optimize import brentq
    Pv = reduced_params(pl, zeta, U)
    f = lambda p: -Pv[0] * np.sin(p) + Pv[8] * U ** 2 * 0.5 * Pv[14] * np.sin(2 * p) + Pv[4]
    psi_eq = brentq(f, -0.5, 0.5)
    x0 = pl.equilibrium(psi=psi_eq)
    t0 = time.time()
    t6, X6 = pl.simulate(T - 1.0, dt=0.5, x0=x0, rtol=1e-8, atol=1e-10)
    psi6 = np.degrees(pl.yaw_of(X6))
    zz = np.zeros(len(u))
    Xr = integrate(np.array([psi_eq, 0, 0, 0, 0, 0]), dt, U, u, v, zz, zz, zz, zz, zz, Pv)
    psir = np.degrees(Xr[::10, 0])[:len(t6)]
    out["turbulent"] = dict(t=t6.tolist(), psi_6dof=psi6.tolist(), psi_red=psir.tolist(),
                            psi_eq_static=float(np.degrees(psi_eq)),
                            std_6dof=float(psi6[200:].std()), std_red=float(psir[200:].std()),
                            mean_6dof=float(psi6[200:].mean()), mean_red=float(psir[200:].mean()),
                            corr=float(np.corrcoef(psi6[200:], psir[200:])[0, 1]),
                            max_diff=float(np.max(np.abs(psi6 - psir))), wall=time.time() - t0)
    print("turbulent", {k: v for k, v in out["turbulent"].items() if not isinstance(v, list)})
    with open(f"{cm.RES}/s7_verify_6dof.json", "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    run()
