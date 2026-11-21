"""Supplementary checks for Sections 5 and 7:
(1) time-step convergence with ONE wind realisation (generated at the finest step);
(2) attribution of the vessel effect on yaw to its three coupling paths;
(3) parametric excitation by swell at twice the yaw frequency (time domain)."""
import json
import numpy as np
import common as cm
from aero_models import Munk, U_div
import s5_dynamics as d
from s3_statics import z_payload

out = {}
m = Munk(json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"])
box = d.box_model()

# (1) time-step convergence --------------------------------------------------
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
print(json.dumps({k: v["max_rel_err_vs_finest"] for k, v in out["dt_convergence"].items()}))

# (2) attribution of the vessel effect (mating, heading 120 deg, U = 8 m/s) ----
L = cm.L_mating; z = z_payload(L)
att = {}
for mname, mm in (("munk", m), ("box", box)):
    r = {}
    r["none"] = d.summarize(d.simulate(L, mm, 0.01, 8.0, 0.06, z, range(10)), "std")
    r["all"] = d.summarize(d.simulate(L, mm, 0.01, 8.0, 0.06, z, range(10), vessel=True,
                                      heading=120.0), "std")
    for only in ("yaw", "horiz", "heave"):
        r[only + "_only"] = d.summarize(d.simulate(L, mm, 0.01, 8.0, 0.06, z, range(10), vessel=True,
                                                   heading=120.0, only=only), "std")
    att[mname] = r
out["vessel_attribution"] = att
print(json.dumps(att))

# (3) parametric excitation by swell at twice the yaw frequency ---------------
s = cm.suspension(L)
par = {}
for U in (8.0,):
    u_r = U / U_div(s.k_psi, m.CK)
    f_eff = s.f_psi * np.sqrt(1 - u_r ** 2)
    Tsw = 1.0 / (2 * f_eff)
    for zeta in (0.002, 0.01):
        for Hs in (1.0, 2.0):
            for hd in (120.0, 150.0):
                base = d.simulate(L, m, zeta, U, 0.06, z, range(10))
                heave = d.simulate(L, m, zeta, U, 0.06, z, range(10), vessel=True, heading=hd,
                                   Hs=Hs, Tp=Tsw, only="heave")
                par[f"zeta={zeta}|Hs={Hs}|hd={hd}"] = dict(
                    T_swell=float(Tsw), none=d.summarize(base, "std"),
                    heave_only=d.summarize(heave, "std"),
                    max_abs_heave=d.summarize(heave, "max_abs"))
out["parametric_swell"] = par
print(json.dumps(par, indent=0))

# (4) swell near the coupled roll-pendulum modes at mating (two-way coupled boom-tip motion)
sw = {}
for mname, mm in (("munk", m), ("box", box)):
    base = d.simulate(L, mm, 0.01, 8.0, 0.06, z, range(10))
    for Tsw in (18.6, 11.3):
        for Hs in (1.0, 2.0):
            for hd in (90.0, 120.0):
                st_all = d.simulate(L, mm, 0.01, 8.0, 0.06, z, range(10), vessel=True, heading=hd,
                                    Hs=Hs, Tp=Tsw)
                st_h = d.simulate(L, mm, 0.01, 8.0, 0.06, z, range(10), vessel=True, heading=hd,
                                  Hs=Hs, Tp=Tsw, only="horiz")
                sw[f"{mname}|Tp={Tsw}|Hs={Hs}|hd={hd}"] = dict(
                    none=d.summarize(base, "std"), all=d.summarize(st_all, "std"),
                    horiz=d.summarize(st_h, "std"), diff_all_pct=d.paired(base, st_all),
                    diff_horiz_pct=d.paired(base, st_h), sway_rel=d.summarize(st_all, "sway_rel_std"))
out["swell_coupled_modes"] = sw
print(json.dumps(sw, indent=0))

# (5) wind synthesis: standard deviations of one realisation against the target
U, TI, zz = 10.0, 0.06, 150.0
u, v = d.turbulence(0, 4200.0, 0.05, U, TI, zz)
out["wind_synthesis"] = dict(sigma_u_ratio=float(u.std() / (TI * U)),
                             sigma_v_ratio=float(v.std() / (cm.r_sigma_v * TI * U)))
print(out["wind_synthesis"])
json.dump(out, open(f"{cm.RES}/s5b_checks.json", "w"), indent=1)
