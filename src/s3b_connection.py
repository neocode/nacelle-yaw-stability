"""
Section 4.4: yaw stiffness of the connection between the hook block and the payload.

The falls act on the lower (hook) block; the payload is connected to it through the
hook, the yoke and the slings, whose yaw stiffness k_c acts in series with k_psi:
    k_eq = k_psi k_c / (k_psi + k_c),   U_div,c = U_div sqrt(r / (1 + r)),  r = k_c / k_psi.
(1) Series reduction of the divergence speed along the lift for both aerodynamic
    scenarios; gravitational (bifilar) stiffness of slings from a twin hook.
(2) Two-mass model (block + payload): eigenvalue threshold against the series formula
    and the natural periods of the two modes.
(3) Linear spectral response of the two-mass model: wind speed at which the expected
    largest yaw in 20 min reaches the tolerance, versus r and versus the aerodynamic slope.
"""
import json
import numpy as np
from scipy.optimize import brentq
import common as cm
from aero_models import Munk, U_div, ref_slope
import s5_dynamics as d
from s3_statics import z_payload

R_LIST = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf)
I_B_FRAC = 0.01          # yaw inertia of the hook block / I_zz (yoke counted with the payload)


def factor(r):
    """U_div,c / U_div for the connection-to-falls stiffness ratio r."""
    return 1.0 if np.isinf(r) else float(np.sqrt(r / (1.0 + r)))


def k_twin_hook(a, b, h, W):
    """Gravitational yaw stiffness of two slings from the horns of a twin hook
    (half-spacing a) to the yoke (half-spacing b), vertical extent h, weight W."""
    return W * a * b / h


def two_mass(L, model, U, r, zeta_s=0.01, control=False, kp=cm.k_p_nom, kd=cm.k_d_nom,
             I_b=None):
    """Mass, damping and stiffness matrices of the linearised block-payload model about
    the mean equilibrium.  The structural damping acts on the payload with the damping
    ratio zeta_s of the yaw mode of the series system (k_eq), so that r changes only
    the stiffness.  Returns M, C, K, the mean payload yaw and the aerodynamic terms."""
    s = cm.suspension(L)
    P = d.params_vector(L, model, zeta_s, U, control, kp, kd)
    k, Mr = s.k_psi, s.M_r
    kc = 1e6 * k if np.isinf(r) else r * k
    I_b = I_B_FRAC * cm.I_zz if I_b is None else I_b
    qasl = P[8] * U ** 2
    th0 = getattr(model, "theta0", 0.0)
    kpp = kp if control else 0.0

    def block(psi):
        # block yaw for a given payload yaw: k sin(pb) + kc pb = Mr + kc psi (monotonic)
        return brentq(lambda pb: k * np.sin(pb) + kc * pb - Mr - kc * psi, -1.5, 1.5)

    def G(psi, Ui):
        return -kc * (psi - block(psi)) + P[8] * Ui ** 2 * float(model.c(psi - th0)) - kpp * psi

    psi = np.arcsin(min(Mr / k, 1.0))          # still air: payload and block at the rope-torque twist
    for Ui in np.linspace(0.0, U, 21)[1:]:
        grid = np.linspace(psi - 0.1, psi + 0.1, 201)
        Gv = np.array([G(x, Ui) for x in grid])
        idx = [i for i in range(200) if Gv[i] > 0 >= Gv[i + 1]]
        if not idx:
            return None
        i = min(idx, key=lambda j: abs(grid[j] - psi))
        psi = brentq(lambda p: G(p, Ui), grid[i], grid[i + 1])
    pb = block(psi)
    Ka = qasl * float(model.dc(psi - th0))
    cval = float(model.c(psi - th0))
    keq = 1.0 / (1.0 / (k * np.cos(pb)) + 1.0 / kc)
    c_s = 2 * zeta_s * np.sqrt(keq * cm.I_zz)
    M = np.diag([I_b, cm.I_zz])
    K = np.array([[k * np.cos(pb) + kc, -kc], [-kc, kc - Ka + kpp]])
    C = np.diag([0.0, c_s + P[3] + (kd if control else 0.0)])
    return dict(M=M, C=C, K=K, psib=psi, Ka=Ka, c_a=P[3], qasl=qasl, cval=cval)


def threshold_two_mass(L, model, r, zeta_s=0.01, Umax=80.0):
    """Smallest U at which an eigenvalue of the linearised two-mass model about psi = 0
    (no rope torque, wind at the model direction) has a non-negative real part."""
    s = cm.suspension(L)
    k = s.k_psi
    kc = 1e6 * k if np.isinf(r) else r * k
    I_b = I_B_FRAC * cm.I_zz
    slope = ref_slope(model)
    lo, hi = 0.0, Umax

    def unstable(U):
        Ka = 0.5 * cm.rho_air * U ** 2 * cm.A_side * cm.L_n * slope
        keq = 1.0 / (1.0 / k + 1.0 / kc)
        c_s = 2 * zeta_s * np.sqrt(keq * cm.I_zz)
        M = np.diag([I_b, cm.I_zz])
        K = np.array([[k + kc, -kc], [-kc, kc - Ka]])
        C = np.diag([0.0, c_s])
        A = np.block([[np.zeros((2, 2)), np.eye(2)],
                      [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
        return np.max(np.linalg.eigvals(A).real) >= -1e-12

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if unstable(mid):
            hi = mid
        else:
            lo = mid
    return hi


def modes(L, r):
    """Undamped natural periods of the two-mass model in still air [s]."""
    s = cm.suspension(L)
    kc = r * s.k_psi
    M = np.diag([I_B_FRAC * cm.I_zz, cm.I_zz])
    K = np.array([[s.k_psi + kc, -kc], [-kc, kc]])
    lam = np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real)
    return [float(2 * np.pi / np.sqrt(x)) for x in lam[::-1]]


def spectral2(L, model, U, r, TI=0.06, zeta_s=0.01, control=False, f=None):
    """Linear spectral response of the payload yaw (two-mass model, wind only):
    standard deviation and expected largest |psi| in 20 min [rad]."""
    sysm = two_mass(L, model, U, r, zeta_s, control)
    if sysm is None:
        return None
    M, C, K = sysm["M"], sysm["C"], sysm["K"]
    if np.max(np.linalg.eigvals(np.linalg.solve(M, K)).real) <= 0 or \
            np.linalg.det(K) <= 0:
        return None
    f = np.logspace(-4, 0.3, 4000) if f is None else f
    w = 2 * np.pi * f
    z = z_payload(L)
    lam = cm.Lambda1(z)
    Su = d.kaimal(f, TI * U, 8.1 * lam, U)
    Sv = d.kaimal(f, cm.r_sigma_v * TI * U, 2.7 * lam, U)
    Ka, c_a, qasl, cval = sysm["Ka"], sysm["c_a"], sysm["qasl"], sysm["cval"]
    SF = ((Ka ** 2 + (w * c_a) ** 2) / U ** 2) * Sv + (2 * qasl * cval / U) ** 2 * Su
    det = ((K[0, 0] - w ** 2 * M[0, 0] + 1j * w * C[0, 0]) * (K[1, 1] - w ** 2 * M[1, 1] + 1j * w * C[1, 1])
           - K[0, 1] * K[1, 0])
    H22 = (K[0, 0] - w ** 2 * M[0, 0] + 1j * w * C[0, 0]) / det
    S = np.abs(H22) ** 2 * SF
    m0 = np.trapezoid(S, f)
    nu = np.sqrt(np.trapezoid(f ** 2 * S, f) / max(m0, 1e-30))
    nT = max(nu * 1200.0, 1.0001)
    gpk = np.sqrt(2 * np.log(nT)) + 0.5772 / np.sqrt(2 * np.log(nT))
    sig = np.sqrt(m0)
    return dict(sigma=float(sig), expected_max=float(abs(sysm["psib"]) + gpk * sig),
                psib=float(sysm["psib"]))


def U_tol(L, model, r, tol_deg=1.0, TI=0.06, zeta_s=0.01, Umax=40.0, control=False):
    """Lowest wind speed at which the expected largest yaw in 20 min reaches tol."""
    tol = np.radians(tol_deg)
    Us = np.arange(1.0, Umax + 1e-9, 0.5)
    prev = None
    for U in Us:
        o = spectral2(L, model, U, r, TI, zeta_s, control)
        if o is None:
            return float(U) if prev is None else float(prev[0] + 0.25)
        if o["expected_max"] >= tol:
            if prev is None:
                return float(U)
            # refine by bisection
            lo, hi = prev[0], U
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                om = spectral2(L, model, mid, r, TI, zeta_s, control)
                if om is None or om["expected_max"] >= tol:
                    hi = mid
                else:
                    lo = mid
            return float(0.5 * (lo + hi))
        prev = (U, o)
    return np.inf


def run():
    out = {}
    CK = json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"]
    models = {"munk": Munk(CK), "box": d.box_model()}
    stages = {"mating": cm.L_mating, "upper": 60.0, "mid": cm.L_ref, "liftoff": cm.L_liftoff}

    # (1) series reduction of the divergence speed -------------------------------
    tab = {}
    for sn, L in stages.items():
        s = cm.suspension(L)
        for mn, m in models.items():
            Ud = U_div(s.k_psi, ref_slope(m))
            tab[f"{sn}|{mn}"] = {str(r): Ud * factor(r) for r in R_LIST}
    out["U_div_series"] = tab
    # twin-hook slings: gravitational stiffness of the connection
    W_below = (cm.m_p + 0.5 * cm.m_hb) * cm.g          # nacelle + yoke (half of block + yoke mass)
    tw = {}
    for a, b, h in ((0.4, 1.0, 10.0), (0.5, 2.0, 8.0), (0.8, 3.0, 5.0)):
        kc = k_twin_hook(a, b, h, W_below)
        row = dict(a=a, b=b, h=h, k_c=kc)
        for sn, L in stages.items():
            s = cm.suspension(L)
            r = kc / s.k_psi
            row[sn] = dict(r=r, factor=factor(r),
                           U_munk=U_div(s.k_psi, ref_slope(models["munk"])) * factor(r),
                           U_box=U_div(s.k_psi, ref_slope(models["box"])) * factor(r))
        tw[f"a={a}|b={b}|h={h}"] = row
    out["twin_hook"] = tw
    print(json.dumps({k: {kk: round(vv["r"], 2) if isinstance(vv, dict) else vv for kk, vv in v.items()}
                      for k, v in tw.items()}, indent=0))

    # (2) two-mass model: threshold and modes ------------------------------------
    tm = {}
    for sn in ("mating", "liftoff"):
        L = stages[sn]
        s = cm.suspension(L)
        for r in (0.1, 0.5, 1.0, 5.0):
            Ud = U_div(s.k_psi, ref_slope(models["munk"]))
            tm[f"{sn}|r={r}"] = dict(eig=threshold_two_mass(L, models["munk"], r),
                                     series=Ud * factor(r), periods=modes(L, r))
    out["two_mass"] = tm
    print(json.dumps(tm, indent=0))

    # (3) operational threshold (expected largest yaw in 20 min = 1 deg) vs r ----------
    rs = np.r_[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
    ut = {"r": rs.tolist()}
    for mn, m in models.items():
        for TI in cm.TI_values:
            ut[f"mating|{mn}|TI={TI}"] = [U_tol(cm.L_mating, m, r, 1.0, TI) for r in rs]
            ut[f"mating|{mn}|TI={TI}|Udiv"] = [U_div(cm.suspension(cm.L_mating).k_psi, ref_slope(m)) * factor(r)
                                               for r in rs]
        print(mn, [round(x, 1) for x in ut[f"mating|{mn}|TI=0.06"]])
    out["U_tol_vs_r"] = ut

    # map over the aerodynamic slope (potential-flow shape, wind along the axis) and r
    cps = np.r_[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
    rr = np.r_[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
    mp = {"cprime": cps.tolist(), "r": rr.tolist(), "U_tol": [], "U_div": []}
    s = cm.suspension(cm.L_mating)
    for cp in cps:
        m = Munk(cp)
        mp["U_tol"].append([U_tol(cm.L_mating, m, r, 1.0, 0.06) for r in rr])
        mp["U_div"].append([U_div(s.k_psi, cp) * factor(r) for r in rr])
    out["map_mating"] = mp
    out["I_b_frac"] = I_B_FRAC
    with open(f"{cm.RES}/s3b_connection.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


if __name__ == "__main__":
    o = run()
    for k, v in o["U_div_series"].items():
        print(k, {kk: round(vv, 1) for kk, vv in v.items()})
