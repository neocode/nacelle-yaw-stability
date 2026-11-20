"""Supplementary checks for Section 5/7:
(1) time-step convergence with ONE wind realisation (generated at the finest step);
(2) attribution of the vessel effect on yaw to its three coupling paths."""
import json, numpy as np
import common as cm
from aero_models import Munk, BluffBox, U_div
import s5_dynamics as d
from s3_statics import z_payload

out = {}
m = Munk(json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"])
L = cm.L_ref; z = z_payload(L)
Ud = U_div(cm.suspension(L).k_psi, m.CK); U = 0.5 * Ud
P = d.params_vector(L, m, 0.01, U, rope=False)
T = 4200.0; dtf = 0.0125
rows = {}
for sd in range(4):
    u, v = d.turbulence(sd, T, dtf, U, 0.06, z)
    for stride in (16, 8, 4, 2, 1):
        dt = dtf * stride
        uu, vv = u[::stride], v[::stride]
        zz = np.zeros(len(uu))
        X = d.integrate(np.zeros(6), dt, U, uu, vv, zz, zz, zz, zz, zz, P)
        i0 = int(600 / dt)
        rows.setdefault(dt, []).append(float(np.degrees(X[i0:, 0].std())))
ref = np.array(rows[dtf])
out["dt_convergence"] = {str(k): dict(std=v, max_rel_err_vs_finest=float(np.max(np.abs(np.array(v) - ref) / ref)))
                         for k, v in rows.items()}
print(json.dumps(out["dt_convergence"], indent=0))

# attribution of the vessel effect (mating, heading 120 deg, box and Munk, U = 8 m/s)
L = cm.L_mating; z = z_payload(L)
att = {}
for mname, mm in (("munk", m), ("box", BluffBox(0.09, 0.04))):
    P = d.params_vector(L, mm, 0.01, 8.0)
    psib = d.static_equilibrium(P, 8.0)
    res = {k: [] for k in ("none", "all", "yaw_only", "horiz_only", "heave_only")}
    for sd in range(10):
        n = int(round(T / 0.05))
        u, v = d.turbulence(sd, T, 0.05, 8.0, 0.06, z)
        xt, yt, azt, psv, dpsv = d.vessel_inputs(sd, T, 0.05, 120.0)
        Z = np.zeros(n)
        combos = {"none": (Z, Z, Z, Z, Z), "all": (xt, yt, azt, psv, dpsv),
                  "yaw_only": (Z, Z, Z, psv, dpsv), "horiz_only": (xt, yt, Z, Z, Z),
                  "heave_only": (Z, Z, azt, Z, Z)}
        for k, (a1, a2, a3, a4, a5) in combos.items():
            X = d.integrate(np.array([psib, 0, 0, 0, 0, 0]), 0.05, 8.0, u, v, a1, a2, a3, a4, a5, P)
            res[k].append(float(np.degrees(X[12000:, 0].std())))
    att[mname] = {k: d.summarize([{"s": x} for x in vv], "s") for k, vv in res.items()}
out["vessel_attribution"] = att
print(json.dumps(att, indent=0))
json.dump(out, open(f"{cm.RES}/s5b_checks.json", "w"), indent=1)
