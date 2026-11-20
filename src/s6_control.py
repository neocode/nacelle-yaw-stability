"""
Section 6: yaw control with tugger lines - stiffness versus damping.

(1) Closed-loop divergence speed from the eigenvalues of the linearised
    closed loop, for several derivative gains (numerical check of Eq. 27).
(2) Static closed-loop equilibrium with tugger saturation and a misaligned wind.
(3) Time-domain closed-loop response at mating (wind + vessel), open vs closed.
"""
import json
import numpy as np
import common as cm
from aero_models import Munk, BluffBox, U_div, qASL
import s5_dynamics as d
from s3_statics import z_payload


def eig_threshold(k, I, c, kp, kd, CK, Umax=120.0):
    """Smallest U at which the linear closed loop has an eigenvalue with Re >= 0."""
    Us = np.linspace(0.0, Umax, 4801)
    for U in Us:
        Ka = qASL(U) * CK
        A = np.array([[0.0, 1.0], [-(k + kp - Ka) / I, -(c + kd) / I]])
        if np.max(np.linalg.eigvals(A).real) >= 0:
            return float(U)
    return np.inf


def run():
    out = {}
    CK = json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"]
    m_munk = Munk(CK)
    m_box = BluffBox(0.09, 0.04)
    # (1) closed-loop threshold vs kp for several kd -------------------------
    kps = np.r_[0, np.logspace(4, 7, 25)]
    thr = {"kp": kps.tolist()}
    for L in (cm.L_mating, cm.L_ref):
        s = cm.suspension(L)
        c = 2 * 0.01 * np.sqrt(s.k_psi * cm.I_zz)
        for kd in (0.0, 6e6, 6e7):
            thr[f"L={int(L)}|kd={kd:g}"] = [eig_threshold(s.k_psi, cm.I_zz, c, kp, kd, CK) for kp in kps]
        thr[f"L={int(L)}|closed_form"] = [U_div(s.k_psi, CK) * np.sqrt(1 + kp / s.k_psi) for kp in kps]
    out["threshold"] = thr

    # (2) static equilibrium with saturation, misaligned wind -------------------
    sat = {}
    for L in (cm.L_mating, cm.L_ref):
        s = cm.suspension(L)
        Us = np.linspace(0, 60, 241)
        for th_deg in (0.0, 10.0, 20.0):
            th = np.radians(th_deg)
            psi_eq, Mt_eq = [], []
            prev = 0.0
            for U in Us:
                def M(p):
                    Mt = np.clip(-cm.k_p_nom * p, -cm.M_t_max, cm.M_t_max)
                    return -s.k_psi * np.sin(p) + qASL(U) * m_munk.c(p - th) + s.M_r + Mt
                ps = np.linspace(prev - 0.5, prev + 0.5, 4001)
                F = M(ps)
                idx = np.where(np.sign(F[:-1]) != np.sign(F[1:]))[0]
                stable = [i for i in idx if F[i] > 0 > F[i + 1]]
                if not stable:
                    psi_eq.append(np.nan); Mt_eq.append(np.nan)
                    prev = np.nan
                    break
                i = min(stable, key=lambda j: abs(ps[j] - prev))
                p = 0.5 * (ps[i] + ps[i + 1])
                prev = p
                psi_eq.append(float(np.degrees(p)))
                Mt_eq.append(float(np.clip(-cm.k_p_nom * p, -cm.M_t_max, cm.M_t_max)))
            sat[f"L={int(L)}|theta={th_deg:g}"] = dict(U=Us[:len(psi_eq)].tolist(), psi=psi_eq, Mt=Mt_eq)
    out["saturation"] = sat

    # (3) time domain at mating: open vs closed loop, Munk and box -------------
    L = cm.L_mating
    z = z_payload(L)
    td = {}
    seeds = list(range(10))
    for mname, mm in (("munk", m_munk), ("box", m_box)):
        for U in (8.0, 12.0, 16.0, 20.0):
            for ctrl in (False, True):
                st = d.simulate(L, mm, 0.01, U, 0.06, z, seeds, vessel=True, heading=150.0,
                                control=ctrl)
                td[f"{mname}|U={U}|ctrl={ctrl}"] = dict(
                    std=d.summarize(st, "std"), mean=d.summarize(st, "mean"),
                    frac=d.summarize(st, "frac_in_tol"), maxabs=d.summarize(st, "max_abs"))
            print(mname, U, td[f"{mname}|U={U}|ctrl=False"]["std"], td[f"{mname}|U={U}|ctrl=True"]["std"])
    out["time_domain"] = td
    # closed-loop sensitivity to structural damping (spectral)
    sens = {}
    for mname, mm in (("munk", m_munk), ("box", m_box)):
        for zeta in cm.zeta_values:
            for ctrl in (False, True):
                spx = d.spectral(L, mm, zeta, 12.0, 0.06, z, control=ctrl)
                sens[f"{mname}|zeta={zeta}|ctrl={ctrl}"] = float(np.degrees(spx["sigma"]))
    out["zeta_sensitivity_U12"] = sens
    with open(f"{cm.RES}/s6_control.json", "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    o = run()
    t = o["threshold"]
    for k in t:
        if k != "kp":
            print(k, [round(x, 1) for x in np.array(t[k])[[0, 8, 16, 25]]])
    print(json.dumps(o["zeta_sensitivity_U12"], indent=0))
    for k, v in o["saturation"].items():
        print(k, "last U with equilibrium:", v["U"][-1], "psi:", v["psi"][-1], "Mt:", v["Mt"][-1])
