"""
Section 6: feasibility of the tugger controller with two physical lines, and its
robustness to actuator lag and measurement delay.

(1) Time domain at mating and at L = 100 m (wind and vessel motion, heading 150 deg):
    two lines attached at +-r_t = +-10 m on the payload axis, pulling towards anchors on
    the boom 30 m away, pretension 50 kN, tension limits 5-100 kN; the yaw couple of the
    PD law is produced by the tension difference.  Compared with the ideal couple.
(2) The same lines with a common-mode (tension-sum) damping of the transverse sway.
(3) Critical first-order actuator lag and pure-delay margin versus the gains.
"""
import json
import time
import numpy as np
import common as cm
import s5_dynamics as d
from aero_models import Munk, U_div, qASL
from s3_statics import z_payload


def tau_crit(k, I, c, kp, kd, Ka):
    """Smallest first-order lag tau at which M_t = -(kp psi + kd psi')/(1 + tau s)
    destabilises the loop (bisection on the eigenvalues)."""
    def maxre(tau):
        A = np.array([[0.0, 1.0, 0.0],
                      [-(k - Ka) / I, -c / I, 1.0 / I],
                      [-kp / tau, -kd / tau, -1.0 / tau]])
        return float(np.max(np.linalg.eigvals(A).real))
    lo, hi = 1e-3, 200.0
    if maxre(hi) < 0:
        return np.inf
    if maxre(lo) >= 0:
        return 0.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if maxre(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float(hi)


def delay_margin(k, I, c, kp, kd, Ka):
    """Smallest pure delay T for which I s^2 + c s + (k - Ka) + (kp + kd s) e^{-sT} = 0
    has a root on the imaginary axis (closed loop stable for T = 0 assumed)."""
    w = np.logspace(-4, 2, 200001)
    P = (k - Ka) - I * w ** 2 + 1j * c * w
    Cc = kp + 1j * kd * w
    g = np.abs(P) - np.abs(Cc)
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    best = np.inf
    for i in idx:
        wc = w[i]
        ph = np.angle(-P[i] / Cc[i])              # e^{-i w T} = -P/C
        T = (-ph) % (2 * np.pi) / wc
        best = min(best, T)
    return float(best)


def run(seeds=range(10)):
    t0 = time.time()
    out = {}
    CK = json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"]
    models = {"munk": Munk(CK), "box": d.box_model()}

    # (1)+(2) two physical lines ---------------------------------------------------
    td = {}
    for cname, L, Us in (("mating", cm.L_mating, (12.0, 16.0, 20.0)), ("long", cm.L_ref, (12.0, 16.0))):
        z = z_payload(L)
        for mn, m in models.items():
            for U in Us:
                row = {}
                for tag, ctrl, tg in (("ideal", True, None), ("lines", "lines", None),
                                      ("lines+sway", "lines",
                                       dict(T0=100e3, T_max=250e3, c_sway=1.0e5))):
                    st = d.simulate(L, m, 0.01, U, 0.06, z, seeds, vessel=True, heading=150.0,
                                    control=ctrl, tugger=tg)
                    r = dict(std=d.summarize(st, "std"), maxabs=d.summarize(st, "max_abs"),
                             sway_rel=d.summarize(st, "sway_rel_std"))
                    if ctrl == "lines":
                        for key in ("T_min_kN", "T_max_kN", "dT_max_kN_s", "reel_std", "reel_max",
                                    "frac_saturated", "sway_rel_y_std", "sway_rel_y_mean"):
                            vals = [s_[key] for s_ in st]
                            r[key] = float(np.min(vals) if key == "T_min_kN" else
                                           np.max(vals) if key in ("T_max_kN", "dT_max_kN_s", "reel_max",
                                                                   "frac_saturated") else np.mean(vals))
                    row[tag] = r
                td[f"{cname}|{mn}|U={U}"] = row
                print(cname, mn, U, {k: (round(v["std"][0], 3), round(v.get("T_min_kN", 0), 1),
                                         round(v.get("T_max_kN", 0), 1), round(v["sway_rel"][0], 2))
                                     for k, v in row.items()}, round(time.time() - t0))
    out["two_lines"] = td
    # open loop sway for reference (same wind and waves)
    L = cm.L_mating
    st = d.simulate(L, models["munk"], 0.01, 12.0, 0.06, z_payload(L), seeds, vessel=True, heading=150.0)
    out["open_loop_sway_rel_mating_U12"] = d.summarize(st, "sway_rel_std")

    # (3) lag and delay margins versus the gains -----------------------------------------
    kps = (0.5e6, 1e6, 2e6, 4e6)
    kds = (1e6, 2e6, 3e6, 6e6, 12e6)
    mg = {}
    for cname, L in (("mating", cm.L_mating), ("liftoff", cm.L_liftoff)):
        s = cm.suspension(L)
        c = 2 * 0.01 * np.sqrt(s.k_psi * cm.I_zz)
        z = z_payload(L)
        for U in (8.0, 12.0):
            Ka = qASL(U) * CK
            for kp in kps:
                for kd in kds:
                    key = f"{cname}|U={U}|kp={kp:g}|kd={kd:g}"
                    row = dict(tau_crit=tau_crit(s.k_psi, cm.I_zz, c, kp, kd, Ka),
                               delay_margin=delay_margin(s.k_psi, cm.I_zz, c, kp, kd, Ka),
                               U_div_cl=U_div(s.k_psi, CK) * np.sqrt(1 + kp / s.k_psi))
                    if cname == "mating":
                        row["sigma_cl_deg"] = float(np.degrees(
                            d.spectral(L, models["munk"], 0.01, U, 0.06, z, control=True, kp=kp, kd=kd)["sigma"]))
                    mg[key] = row
    out["margins"] = mg
    for k, v in mg.items():
        if "|U=12.0|" in k:
            print(k, {kk: round(vv, 3) for kk, vv in v.items()})
    out["wall_s"] = time.time() - t0
    with open(f"{cm.RES}/s6b_tuggers.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
