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

KI = 6.7e4      # N m/(rad s): integral gain, integral time k_p/k_i = 30 s


def tau_crit(k, I, c, kp, kd, Ka, ki=0.0):
    """Smallest first-order lag tau at which M_t = -(kp psi + kd psi' + ki int psi)/(1 + tau s)
    destabilises the loop, and the upper end of the unstable window (a very slow actuator
    removes the control and leaves the stable open loop).  Scan in log(tau), then bisection."""
    def maxre(tau):
        if ki > 0:
            A = np.array([[0.0, 1.0, 0.0, 0.0],
                          [-(k - Ka) / I, -c / I, 1.0 / I, 0.0],
                          [-kp / tau, -kd / tau, -1.0 / tau, -ki / tau],
                          [1.0, 0.0, 0.0, 0.0]])
        else:
            A = np.array([[0.0, 1.0, 0.0],
                          [-(k - Ka) / I, -c / I, 1.0 / I],
                          [-kp / tau, -kd / tau, -1.0 / tau]])
        return float(np.max(np.linalg.eigvals(A).real))
    taus = np.logspace(-2, 3, 1001)
    un = np.array([maxre(t) >= 0 for t in taus])
    if not un.any():
        return np.inf, np.inf
    if un[0]:
        return 0.0, float(taus[np.argmin(un)]) if not un.all() else np.inf

    def refine(lo, hi, want_unstable_at_hi=True):
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            if (maxre(mid) >= 0) == want_unstable_at_hi:
                hi = mid
            else:
                lo = mid
        return hi
    i = int(np.argmax(un))
    t1 = refine(taus[i - 1], taus[i])
    j = len(taus) - 1 - int(np.argmax(un[::-1]))
    t2 = np.inf if j == len(taus) - 1 else refine(taus[j], taus[j + 1], want_unstable_at_hi=False)
    return float(t1), float(t2)


def delay_margin(k, I, c, kp, kd, Ka, ki=0.0):
    """Smallest pure delay T for which I s^2 + c s + (k - Ka) + (kp + kd s + ki/s) e^{-sT} = 0
    has a root on the imaginary axis (closed loop stable for T = 0 assumed)."""
    w = np.logspace(-4, 2, 200001)
    P = (k - Ka) - I * w ** 2 + 1j * c * w
    Cc = kp + 1j * kd * w - 1j * ki / w
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
                    tc = tau_crit(s.k_psi, cm.I_zz, c, kp, kd, Ka)
                    row = dict(tau_crit=tc[0], tau_unstable_upper=tc[1],
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

    # (4) integral action: misaligned wind, closed-loop tolerance speed (spectral) -------
    ia = {"ki": KI}
    for cname, L in (("mating", cm.L_mating), ("liftoff", cm.L_liftoff)):
        s = cm.suspension(L)
        c = 2 * 0.01 * np.sqrt(s.k_psi * cm.I_zz)
        Ka = qASL(12.0) * CK
        tc = tau_crit(s.k_psi, cm.I_zz, c, cm.k_p_nom, cm.k_d_nom, Ka, KI)
        ia[f"{cname}|margins_U12"] = dict(tau_crit=tc[0], tau_upper=tc[1],
                                         delay_margin=delay_margin(s.k_psi, cm.I_zz, c, cm.k_p_nom,
                                                                   cm.k_d_nom, Ka, KI))
    L = cm.L_mating
    z = z_payload(L)
    Ug = np.arange(4.0, 40.01, 0.5)
    for th_deg in (0.0, 10.0, 20.0):
        m = Munk(CK)
        m.theta0 = np.radians(th_deg)
        for TI in cm.TI_values:
            for tag, ki in (("PD", 0.0), ("PID", KI)):
                em, mean = [], []
                for U in Ug:
                    spx = d.spectral(L, m, 0.01, U, TI, z, control=True, ki=ki)
                    if not np.isfinite(spx["psib"]) or spx["keff"] <= 0:
                        break
                    em.append(float(np.degrees(spx["expected_max"])))
                    mean.append(float(np.degrees(spx["psib"])))
                em = np.array(em)
                lim = None
                if np.any(em >= 1.0):
                    i = int(np.argmax(em >= 1.0))
                    lim = float(Ug[0]) if i == 0 else float(np.interp(1.0, em[i - 1:i + 1], Ug[i - 1:i + 1]))
                ia[f"theta={th_deg:g}|TI={TI}|{tag}"] = dict(U=Ug[:len(em)].tolist(), expected_max=em.tolist(),
                                                           mean=mean, U_tol_spectral=lim)
    # time-domain check at 12 and 16 m/s
    for th_deg in (10.0, 20.0):
        m = Munk(CK)
        m.theta0 = np.radians(th_deg)
        for U in (12.0, 16.0):
            for tag, ki in (("PD", 0.0), ("PID", KI)):
                st = d.simulate(L, m, 0.01, U, 0.06, z, seeds, vessel=True, heading=150.0, control=True, ki=ki)
                ia[f"td|theta={th_deg:g}|U={U}|{tag}"] = dict(mean=d.summarize(st, "mean"), std=d.summarize(st, "std"),
                                                            maxabs=d.summarize(st, "max_abs"))
    out["integral_action"] = ia
    print("integral action", {k: v.get("U_tol_spectral") for k, v in ia.items() if isinstance(v, dict)})

    # (5) passive constant-tension tugger lines (radial): divergence and tolerance speeds ---
    pl = {}
    for Tt in (20e3, 50e3):
        kt = Tt * 10.0 * (1 + 10.0 / 30.0)
        row = dict(k_t=kt)
        for cname, L in (("mating", cm.L_mating), ("upper", 60.0), ("mid", cm.L_ref), ("liftoff", cm.L_liftoff)):
            s = cm.suspension(L)
            row[cname] = dict(PF=U_div(s.k_psi, CK) * np.sqrt(1 + kt / s.k_psi),
                              SB=U_div(s.k_psi, 0.3347) * np.sqrt(1 + kt / s.k_psi))
        L = cm.L_mating
        z = z_payload(L)
        for mn, m in models.items():
            em = []
            for U in Ug:
                spx = d.spectral(L, m, 0.01, U, 0.06, z, control=True, kp=kt, kd=0.0)
                if not np.isfinite(spx["psib"]) or spx["keff"] <= 0:
                    break
                em.append(float(np.degrees(spx["expected_max"])))
            em = np.array(em)
            i = int(np.argmax(em >= 1.0))
            row[f"U_tol_mating|{mn}"] = float(np.interp(1.0, em[i - 1:i + 1], Ug[i - 1:i + 1])) if i > 0 else None
        pl[f"T_t={Tt / 1e3:g}kN"] = row
    out["passive_lines"] = pl
    print("passive lines", json.dumps(pl, indent=0, default=float))

    # (6) swell near the coupled roll-pendulum mode with damped sway (tugger common mode,
    #     modelled as pendulum damping in the two-way coupled vessel solution) ---------------
    sw = {}
    L = cm.L_mating
    z = z_payload(L)
    for mn, m in models.items():
        base = d.simulate(L, m, 0.01, 8.0, 0.06, z, seeds)
        for zp in (0.02, 0.1, 0.2):
            for hd in (90.0, 120.0):
                st = d.simulate(L, m, 0.01, 8.0, 0.06, z, seeds, vessel=True, heading=hd, Hs=2.0, Tp=18.6,
                                zeta_p=zp)
                mL = cm.m_p + cm.m_hb
                c_p = 2 * zp * mL * np.sqrt(cm.g / (L + cm.Z_CG_BELOW_BLOCK))
                sw[f"{mn}|zeta_p={zp}|hd={hd}"] = dict(
                    diff_pct=d.paired(base, st), sway_rel=d.summarize(st, "sway_rel_std"),
                    damping_force_std_kN=float(c_p * np.mean([x["sway_rel_vel_std"] for x in st]) / 1e3))
    out["swell_damped_sway"] = sw
    print("swell damped sway", {k: (round(v["diff_pct"][0]), round(v["sway_rel"][0], 2),
                                   round(v["damping_force_std_kN"])) for k, v in sw.items()})
    out["wall_s"] = time.time() - t0
    with open(f"{cm.RES}/s6b_tuggers.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
