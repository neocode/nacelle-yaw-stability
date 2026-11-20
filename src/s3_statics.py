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
import wind_tunnel_data as wt
from aero_models import Munk, BluffBox, MeasuredBox, U_div, qASL, max_slope

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


def U_crit(k, model):
    """Lowest linear divergence speed over wind directions: the aerodynamic
    stiffness at psi = 0 is q A_s L c'(-theta), largest for theta = -beta_max."""
    return U_div(k, max_slope(model)[0])


def branches(k, Mr, model, u_grid, theta=0.0, Ud=None):
    """Equilibrium yaw vs U/Ud (default: U_div of the perfect, aligned system)."""
    Ud = U_div(k, model.CK) if Ud is None else Ud
    rows = []
    for u in u_grid:
        for r, st in equilibria(k, Mr, u * Ud, model, theta):
            rows.append((float(u), np.degrees(r), st))
    return rows


def primary_path(k, Mr, model, theta=0.0, ds=2e-4):
    """Equilibrium connected to the still-air twist, parametrised by the yaw:
    q(psi) = (k sin psi - M_r) / (A_s L c(psi - theta)).  Starting from
    psi_0 = asin(M_r/k) in the direction in which q grows, the branch is stable
    while q increases; its first maximum is a fold.  If c changes sign first,
    q grows without bound (the payload turns towards a zero of c) and there is
    no fold.  Returns (U, psi) along the stable part and the fold speed."""
    psi0 = np.arcsin(Mr / k)
    c0 = float(model.c(psi0 - theta))
    d = 1.0 if c0 >= 0 else -1.0
    s_ = np.arange(ds, np.pi, ds)
    psi = psi0 + d * s_
    c = model.c(psi - theta)
    ch = np.where(np.sign(c) != d)[0]
    n = ch[0] if len(ch) else len(s_)
    psi, c = psi[:n], c[:n]
    q = (k * np.sin(psi) - Mr) / (cm.A_side * cm.L_n * c)
    dq = np.diff(q)
    dec = np.where(dq <= 0)[0]
    if len(dec):
        i = dec[0] + 1
        U_f = float(np.sqrt(2 * q[i - 1] / cm.rho_air))
        psi, q = psi[:i], q[:i]
    else:
        U_f = np.inf
    U = np.sqrt(2 * np.maximum(q, 0) / cm.rho_air)
    return np.r_[0.0, U], np.r_[psi0, psi], U_f


def primary_branch(k, Mr, model, u_grid, theta=0.0, Ud=None):
    """Primary equilibrium [deg] on a grid of U/Ud; NaN beyond a fold."""
    Ud = U_div(k, model.CK) if Ud is None else Ud
    U, psi, U_f = primary_path(k, Mr, model, theta)
    Ug = np.asarray(u_grid) * Ud
    out = np.interp(Ug, U, np.degrees(psi), right=np.nan)
    out[Ug > min(U_f, U[-1])] = np.nan
    return out


def run():
    res = {}
    with open(f"{cm.RES}/s2_aero_bem.json") as fh:
        CKd = json.load(fh)["CK"]
    models = {
        "Munk (BEM box)": Munk(CKd["bem_box"]),
        "Munk (spheroid)": Munk(CKd["munk_spheroid_own_volume"]),
        "box": MeasuredBox(),
        **{f"box, K={K}": MeasuredBox(K) for K in (4, 5, 6, 7) if K != wt.K_NOMINAL},
        "box, x_s=-1 m": MeasuredBox(xs=-1.0),
        "box, x_s=+1 m": MeasuredBox(xs=+1.0),
        "box, Eq. (12)": BluffBox(0.09, 0.04),
    }
    BOX_FITS = ("box",) + tuple(f"box, K={K}" for K in (4, 5, 6, 7) if K != wt.K_NOMINAL)
    res["CK"] = {n: m.CK for n, m in models.items()}
    res["max_slope"] = {n: [max_slope(m)[0], float(np.degrees(max_slope(m)[1]))]
                        for n, m in models.items()}
    res["K3_over_K"] = {n: m.K3_over_K() for n, m in models.items() if hasattr(m, "K3_over_K")}

    # ---------------------------------------------------- U_div along the lift
    Ls = np.linspace(cm.L_mating, cm.L_liftoff, 53)
    lift = {"L": Ls.tolist(), "z_payload": [z_payload(L) for L in Ls]}
    for n, m in models.items():
        # aligned wind for potential flow; worst wind direction for the box
        Ud = [U_div(cm.suspension(L).k_psi, m.CK) if isinstance(m, Munk) else
              U_crit(cm.suspension(L).k_psi, m) for L in Ls]
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
        tab[str(int(L))].update({n + " | worst direction": U_crit(s.k_psi, m)
                                 for n, m in models.items() if not isinstance(m, Munk)})
        tab[str(int(L))]["k_psi"] = s.k_psi
        tab[str(int(L))]["z_payload"] = z_payload(L)
    res["table"] = tab

    # ---------------------------------------------- heading dependence (linear)
    th = np.radians(np.linspace(0, 180, 181))
    head = {"theta_deg": np.degrees(th).tolist()}
    for n in ("Munk (BEM box)", "box", "box, x_s=+1 m", "box, Eq. (12)"):
        m = models[n]
        # effective aerodynamic stiffness at psi = 0 for wind direction theta
        head[n] = [float(m.dc(-t)) for t in th]
    res["heading"] = head
    # zero crossings of the aerodynamic stiffness versus wind direction
    zc = {}
    for n in ("Munk (BEM box)",) + BOX_FITS:
        v = np.array([models[n].dc(-t) for t in th])
        idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        zc[n] = [float(np.degrees(th[i])) for i in idx]
        zc[n + " | max"] = [float(np.degrees(th[int(np.argmax(v))])), float(v.max())]
    res["heading_zero_crossings"] = zc

    # nonlinear check for the box: fold of the primary branch (rope torque included)
    # over wind directions -90..90 deg, and its yaw at the linear critical speed
    worst = {}
    ths = np.radians(np.arange(-90.0, 90.01, 0.5))
    for n in BOX_FITS:
        m = models[n]
        for L in (cm.L_mating, 60.0, cm.L_ref, cm.L_liftoff):
            s = cm.suspension(L)
            Uc = U_crit(s.k_psi, m)
            folds = np.array([primary_path(s.k_psi, s.M_r, m, t)[2] for t in ths])
            i = int(np.argmin(folds))
            bm = max_slope(m)[1]
            psi_at = {}
            for sgn in (+1, -1):
                U, psi, Uf = primary_path(s.k_psi, s.M_r, m, sgn * bm)
                psi_at[f"theta={sgn * np.degrees(bm):.1f}"] = [
                    float(np.degrees(np.interp(f * Uc, U, psi, right=np.nan)))
                    for f in (0.5, 0.8, 0.9, 1.0, 1.1, 1.2)]
            worst[f"{n} | L={int(L)}"] = dict(
                U_crit=float(Uc), beta_max_deg=float(np.degrees(bm)),
                U_fold_min=float(folds[i]), theta_fold_deg=float(np.degrees(ths[i])),
                psi_at_u=psi_at)
    res["worst_heading"] = worst

    # ------------------- uncertainty envelope of the measured box: data choice x fit order
    env = {}
    Lkeys = (cm.L_mating, 60.0, cm.L_ref, cm.L_liftoff)
    ths_e = np.radians(np.arange(-60.0, 60.01, 1.0))
    U_all = []
    from aero_models import zero_near_max_slope
    for ch, sets in wt.DATA_CHOICES.items():
        for K in wt.K_RANGE:
            m = MeasuredBox(K, sets=sets)
            cmax, bmax = max_slope(m)
            row = dict(CK=m.CK, cmax=cmax, beta_max_deg=float(np.degrees(bmax)),
                       zero_deg=float(np.degrees(zero_near_max_slope(m))), rms=wt.fit(K, sets)[2],
                       U_crit={}, fold_min={})
            for L in Lkeys:
                sL = cm.suspension(L)
                row["U_crit"][str(int(L))] = U_crit(sL.k_psi, m)
                folds = np.array([primary_path(sL.k_psi, sL.M_r, m, t)[2] for t in ths_e])
                i = int(np.argmin(folds))
                row["fold_min"][str(int(L))] = [float(folds[i]), float(np.degrees(ths_e[i])),
                                                float(folds[i] / U_crit(sL.k_psi, m))]
            U_all.append([U_crit(cm.suspension(L).k_psi, m) for L in Ls])
            env[f"{ch} | K={K}"] = row
    res["envelope"] = env
    U_all = np.array(U_all)
    res["along_lift"]["box envelope min"] = U_all.min(0).tolist()
    res["along_lift"]["box envelope max"] = U_all.max(0).tolist()

    # ---------------------------------------------- equilibrium branches
    u = np.linspace(0.0, 1.6, 321)
    br = {}
    for L in (cm.L_ref, cm.L_mating):
        s = cm.suspension(L)
        for n, th in (("Munk (BEM box)", 0.0), ("box", None)):
            m = models[n]
            if th is None:
                # wind at the zero of c nearest to the steepest slope: psi = 0 is then an
                # equilibrium of the perfect system.  The side is chosen so that the
                # rope-torque twist (psi > 0) moves the incidence towards the steepest slope.
                bm = max_slope(m)[1]
                bz = brentq(lambda b: float(m.c(b)), bm - 0.2, bm + 0.2)
                th = +bz
                Ud = U_div(s.k_psi, float(m.dc(bz)))
            else:
                Ud = U_div(s.k_psi, m.CK)
            key = f"{n} | L={int(L)}"
            br[key] = dict(
                theta_deg=float(np.degrees(th)), U_ref=float(Ud),
                perfect=branches(s.k_psi, 0.0, m, u, th, Ud),
                imperfect=branches(s.k_psi, s.M_r, m, u, th, Ud),
                primary=primary_branch(s.k_psi, s.M_r, m, u, th, Ud).tolist(),
                primary_mirror=primary_branch(s.k_psi, s.M_r, m, u, -th, Ud).tolist(),
                amplification_linear=[float(np.degrees(s.M_r / (s.k_psi * (1 - x ** 2))))
                                      if x < 1 else None for x in u],
                psi0_deg=float(np.degrees(s.psi0)))
        # aligned wind on the box: the still-air twist is reduced (weathervane)
        m = models["box"]
        br[f"box aligned | L={int(L)}"] = dict(
            primary=primary_branch(s.k_psi, s.M_r, m, u, 0.0, Ud).tolist())
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
    print("max slope:", json.dumps(r["max_slope"], indent=0))
    for L, row in r["table"].items():
        print(L, {k: (round(v, 1) if isinstance(v, float) and v < 1e4 else v) for k, v in row.items()})
    for k, v in r["worst_heading"].items():
        print(k, json.dumps(v))
    for k, v in r["envelope"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print(json.dumps(r["munk_lock_check"], indent=0))
    for k, v in r["branches"].items():
        p = np.array(v["primary"], float)
        ug = np.array(r["u_grid"])
        last = np.where(~np.isnan(p))[0][-1]
        print(k, "primary branch up to u=", ug[last], " psi there=", p[last],
              " theta=", v.get("theta_deg"))
