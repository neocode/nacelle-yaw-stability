"""
Section 4.4-4.6: static divergence along the lift, wind-heading dependence and
equilibrium branches with the complete moment balance (rope torque included).

Complete static balance (relative to the boom-tip/deck reference):
    M(psi) = -k_psi sin(psi) + q A_s L c(psi - theta) + M_r = 0
with theta the mean wind direction relative to the reference.
Stability of an equilibrium: dM/dpsi < 0.
"""
import json
import numpy as np
from scipy.optimize import brentq
import common as cm
from aero_models import Munk, BluffBox, U_div, qASL

H_TIP = cm.BOOM_TIP_HEIGHT
Z_BLOCK_CG = cm.Z_CG_BELOW_BLOCK
z_payload = cm.z_payload


def equilibria(k, Mr, U, model, theta=0.0, npts=7201):
    """All equilibria psi in (-pi, pi] and their stability."""
    psi = np.linspace(-np.pi, np.pi, npts)
    f = lambda p: -k * np.sin(p) + qASL(U) * model.c(p - theta) + Mr
    F = f(psi)
    roots = []
    for i in range(npts - 1):
        if F[i] == 0.0:
            roots.append(psi[i])
        elif F[i] * F[i + 1] < 0:
            roots.append(brentq(f, psi[i], psi[i + 1], xtol=1e-14))
    out = []
    for r in roots:
        dM = -k * np.cos(r) + qASL(U) * model.dc(r - theta)
        out.append((float(r), bool(dM < 0)))
    return out


def branches(k, Mr, model, u_grid, theta=0.0):
    """Equilibrium yaw vs U/U_div (U_div of the perfect, aligned system)."""
    Ud = U_div(k, model.CK)
    rows = []
    for u in u_grid:
        for r, st in equilibria(k, Mr, u * Ud, model, theta):
            rows.append((float(u), np.degrees(r), st))
    return rows


def primary_branch(k, Mr, model, u_grid, theta=0.0):
    """Follow the equilibrium connected to the still-air state by continuation;
    stops at a fold (returns NaN beyond it)."""
    Ud = U_div(k, model.CK)
    psi_prev = np.arcsin(Mr / k)
    out = []
    folded = False
    for u in u_grid:
        if folded:
            out.append(np.nan)
            continue
        U = u * Ud
        cand = [r for r, st in equilibria(k, Mr, U, model, theta) if st]
        if not cand:
            folded = True
            out.append(np.nan)
            continue
        r = min(cand, key=lambda x: abs(x - psi_prev))
        if abs(r - psi_prev) > np.radians(8.0) and u > 0.3:
            folded = True        # jump to a remote branch = fold of the primary branch
            out.append(np.nan)
            continue
        psi_prev = r
        out.append(np.degrees(r))
    return np.array(out)


def run():
    res = {}
    with open(f"{cm.RES}/s2_aero_bem.json") as fh:
        CKd = json.load(fh)["CK"]
    models = {
        "Munk (BEM box)": Munk(CKd["bem_box"]),
        "Munk (spheroid)": Munk(CKd["munk_spheroid_own_volume"]),
        "box, high": BluffBox(0.09, 0.03),
        "box, mid": BluffBox(0.09, 0.04),
        "box, low": BluffBox(0.075, 0.04),
        "box, mid, x_s=-1 m": BluffBox(0.09, 0.04, xs=-1.0),
        "box, mid, x_s=+1 m": BluffBox(0.09, 0.04, xs=+1.0),
    }
    res["CK"] = {n: m.CK for n, m in models.items()}
    res["K3_over_K"] = {n: m.K3_over_K() for n, m in models.items()}

    # ---------------------------------------------------- U_div along the lift
    Ls = np.linspace(cm.L_mating, cm.L_liftoff, 53)
    lift = {"L": Ls.tolist(), "z_payload": [z_payload(L) for L in Ls]}
    for n, m in models.items():
        Ud = [U_div(cm.suspension(L).k_psi, m.CK) for L in Ls]
        lift[n] = Ud
        # equivalent hub-height wind at which divergence is reached
        lift[n + " | hub"] = [u * (cm.z_hub / z_payload(L)) ** cm.alpha_shear
                              for u, L in zip(Ud, Ls)]
    res["along_lift"] = lift

    # ------------------------------------------- table: U_div for key configurations
    tab = {}
    for L in (cm.L_mating, 60.0, cm.L_ref, cm.L_liftoff):
        s = cm.suspension(L)
        tab[str(int(L))] = {n: U_div(s.k_psi, m.CK) for n, m in models.items()}
        tab[str(int(L))]["k_psi"] = s.k_psi
        tab[str(int(L))]["z_payload"] = z_payload(L)
    res["table"] = tab

    # ---------------------------------------------- heading dependence (linear)
    th = np.radians(np.linspace(0, 180, 181))
    head = {"theta_deg": np.degrees(th).tolist()}
    for n in ("Munk (BEM box)", "box, mid", "box, mid, x_s=+1 m"):
        m = models[n]
        # effective aerodynamic stiffness at psi = 0 for wind direction theta
        head[n] = [float(m.dc(-t)) for t in th]
    res["heading"] = head
    # zero crossings of the aerodynamic stiffness versus wind direction
    zc = {}
    for n in ("Munk (BEM box)", "box, mid", "box, high", "box, low"):
        v = np.array([models[n].dc(-t) for t in th])
        idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        zc[n] = [float(np.degrees(th[i])) for i in idx]
        zc[n + " | max"] = [float(np.degrees(th[int(np.argmax(v))])), float(v.max())]
    res["heading_zero_crossings"] = zc

    # worst heading for the bluff-body fit: lowest fold speed of the primary branch
    # (rope torque included) over wind directions 0..60 deg
    worst = {}
    for n in ("box, mid", "box, high", "box, low"):
        m = models[n]
        for L in (cm.L_mating, 60.0, cm.L_ref, cm.L_liftoff):
            s = cm.suspension(L)
            Ud = U_div(s.k_psi, m.CK)
            best = (np.inf, None)
            for thd in np.arange(0.0, 61.0, 2.5):
                ug = np.linspace(0.0, 1.5, 301)
                pb = primary_branch(s.k_psi, s.M_r, m, ug, np.radians(thd))
                ok = np.where(~np.isnan(pb))[0]
                uf = ug[ok[-1]] if len(ok) < len(ug) else np.inf
                if uf * Ud < best[0]:
                    best = (uf * Ud, thd)
            worst[f"{n} | L={int(L)}"] = dict(U_fold_min=float(best[0]), theta_deg=best[1],
                                              U_div_aligned=float(Ud))
    res["worst_heading"] = worst

    # ---------------------------------------------- equilibrium branches
    u = np.linspace(0.0, 1.6, 321)
    br = {}
    for L in (cm.L_ref, cm.L_mating):
        s = cm.suspension(L)
        for n in ("Munk (BEM box)", "box, mid"):
            m = models[n]
            key = f"{n} | L={int(L)}"
            br[key] = dict(
                perfect=branches(s.k_psi, 0.0, m, u),
                imperfect=branches(s.k_psi, s.M_r, m, u),
                primary=primary_branch(s.k_psi, s.M_r, m, u).tolist(),
                primary_theta5=primary_branch(s.k_psi, 0.0, m, u, np.radians(5)).tolist(),
                amplification_linear=[float(np.degrees(s.M_r / (s.k_psi * (1 - x ** 2))))
                                      if x < 1 else None for x in u],
                psi0_deg=float(np.degrees(s.psi0)))
    res["branches"] = br
    res["u_grid"] = u.tolist()

    # Munk post-critical lock angle check: cos psi* = (U_div/U)^2
    s = cm.suspension(cm.L_ref)
    m = models["Munk (BEM box)"]
    chk = []
    for uu in (1.05, 1.2, 1.5):
        eq = [np.degrees(r) for r, st in equilibria(s.k_psi, 0.0, uu * U_div(s.k_psi, m.CK), m)
              if st and r > 0]
        chk.append(dict(u=uu, numeric=eq, closed=float(np.degrees(np.arccos(uu ** -2))),
                        onset_2sqrt=float(np.degrees(2 * np.sqrt(uu - 1)))))
    res["munk_lock_check"] = chk
    with open(f"{cm.RES}/s3_statics.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    r = run()
    print("C_K:", json.dumps(r["CK"], indent=0))
    print("K3/K:", json.dumps(r["K3_over_K"], indent=0))
    for L, row in r["table"].items():
        print(L, {k: (round(v, 1) if isinstance(v, float) and v < 1e4 else v) for k, v in row.items()})
    print(json.dumps(r["munk_lock_check"], indent=0))
    for k, v in r["branches"].items():
        p = np.array(v["primary"], float)
        ug = np.array(r["u_grid"])
        last = np.where(~np.isnan(p))[0][-1]
        print(k, "primary branch exists up to u=", ug[last], " psi there=", p[last])
