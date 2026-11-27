"""
Section 5.2: operability of the open-loop yaw at mating.

(1) Probability that the largest |psi| in a 20-min window exceeds the tolerance
    (0.5, 1 and 2 deg), from time-domain records split into 20-min windows; mean and
    95 % confidence interval over independent seeds.  U_tol: wind speed at which the
    probability reaches 10 % (and 50 %).
(2) Spectral thresholds (expected largest yaw) for the same tolerances.
(3) Froya instead of Kaimal spectral shape (same variances), and the native Froya
    turbulence intensity.
(4) Aerodynamic damping of the sharp-edged box halved and doubled.
(5) Reduced frequency k = pi f L_n / U of the yaw mode.
"""
import json
import time
import numpy as np
from scipy.stats import t as tdist
import common as cm
import s5_dynamics as d
from aero_models import Munk, U_div, ref_slope
from s3_statics import z_payload

TOLS = (0.5, 1.0, 2.0)
WINDOW = 1200.0
N_WIN = 20


def prob(stats, tol, key="win_max"):
    """Mean and 95 % half-width over seeds of the fraction of windows above tol."""
    p = np.array([np.mean(np.array(s[key]) > tol) for s in stats])
    hw = tdist.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p))
    return float(p.mean()), float(hw)


def crossing(U, P, level):
    U, P = np.asarray(U), np.asarray(P)
    i = np.argmax(P >= level)
    if not np.any(P >= level):
        return None
    if i == 0:
        return float(U[0])
    return float(np.interp(level, [P[i - 1], P[i]], [U[i - 1], U[i]]))


def run(seeds=range(10)):
    t0 = time.time()
    out = {}
    L = cm.L_mating
    z = z_payload(L)
    CK = json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"]
    models = {"munk": Munk(CK), "box": d.box_model()}

    # (1) exceedance probability from 20-min windows ----------------------------------
    grids = {0.06: list(np.arange(2.5, 10.01, 0.5)) + [12.0, 14.0],
             0.12: list(np.arange(2.0, 8.01, 0.5)) + [10.0]}
    ex = {}
    for mn, m in models.items():
        for TI, Us in grids.items():
            rows = []
            for U in Us:
                st = d.simulate(L, m, 0.01, U, TI, z, seeds, record=WINDOW * N_WIN, windows=WINDOW)
                row = dict(U=U, std=d.summarize(st, "std"), mean=d.summarize(st, "mean"))
                for tol in TOLS:
                    row[f"P|{tol}"] = prob(st, tol)
                    row[f"Pdyn|{tol}"] = prob(st, tol, "win_max_dyn")
                row["median_max"] = float(np.median(np.concatenate([s["win_max"] for s in st])))
                row["frac_in_tol"] = d.summarize(st, "frac_in_tol")
                rows.append(row)
            lim = {}
            for tol in TOLS:
                for key in ("P", "Pdyn"):
                    Pm = [r[f"{key}|{tol}"][0] for r in rows]
                    lim[f"{key}|{tol}|10%"] = crossing(Us, Pm, 0.10)
                    lim[f"{key}|{tol}|50%"] = crossing(Us, Pm, 0.50)
                Pm = np.array([r[f"P|{tol}"][0] for r in rows])
                Ph = np.array([r[f"P|{tol}"][1] for r in rows])
                lim[f"P|{tol}|10%|range"] = [crossing(Us, Pm + Ph, 0.10), crossing(Us, np.maximum(Pm - Ph, 0), 0.10)]
            # alternative criterion: fraction of time within 1 deg falls to 90 %
            ft = np.array([r["frac_in_tol"][0] for r in rows])
            lim["time_in_tol_90%"] = crossing(Us, 1.0 - ft, 0.10)
            ex[f"{mn}|TI={TI}"] = dict(rows=rows, U_tol=lim)
            print(mn, TI, {k: (v if v is None or isinstance(v, list) else round(v, 2)) for k, v in lim.items()},
                  round(time.time() - t0))
    out["exceedance"] = ex

    # (1b) U_tol (P_e = 10 %, 1 deg) for other structural damping ratios -----------------
    zt = {}
    for mn, m in models.items():
        for zeta in (0.002, 0.05):
            for TI in cm.TI_values:
                Us_ = None
                for U in np.linspace(1.0, 20.0, 77):
                    if d.spectral(L, m, zeta, U, TI, z)["expected_max"] >= np.radians(1.0):
                        Us_ = U
                        break
                Ug = np.arange(max(1.5, np.floor(1.2 * Us_) / 2), 1.5 * Us_ + 0.01, 0.5)
                Pm, Ph = [], []
                for U in Ug:
                    st = d.simulate(L, m, zeta, U, TI, z, seeds, record=WINDOW * N_WIN, windows=WINDOW)
                    pm, ph = prob(st, 1.0)
                    Pm.append(pm); Ph.append(ph)
                Pm, Ph = np.array(Pm), np.array(Ph)
                zt[f"{mn}|zeta={zeta}|TI={TI}"] = dict(
                    U=Ug.tolist(), P=Pm.tolist(), hw=Ph.tolist(), U_spectral=float(Us_),
                    U_tol=crossing(Ug, Pm, 0.10),
                    U_tol_range=[crossing(Ug, Pm + Ph, 0.10), crossing(Ug, np.maximum(Pm - Ph, 0), 0.10)])
                print(mn, zeta, TI, zt[f"{mn}|zeta={zeta}|TI={TI}"]["U_tol"], round(time.time() - t0))
    out["U_tol_damping"] = zt

    # (1c) compliant connection, k_c/k_psi = 1: time-domain check of the spectral estimate --
    # (block quasi-static: series stiffness, damping ratio of the payload mode zeta_s,
    #  rope torque transmitted in the ratio k_c/(k_psi + k_c))
    cc = {}
    s_ = cm.suspension(L)
    keq = 0.5 * s_.k_psi
    for mn, m in models.items():
        Pm = []
        Ug = np.arange(2.5, 6.51, 0.5)
        for U in Ug:
            ov = {0: keq, 2: 2 * 0.01 * np.sqrt(keq * cm.I_zz), 4: 0.5 * s_.M_r}
            st = d.simulate(L, m, 0.01, U, 0.06, z, seeds, record=WINDOW * N_WIN, windows=WINDOW,
                            burn=3000.0, P_override=ov)
            Pm.append(prob(st, 1.0)[0])
        cc[mn] = dict(U=Ug.tolist(), P=Pm, U_tol=crossing(Ug, Pm, 0.10))
    out["connection_r1"] = cc
    print("connection r=1", {k: v["U_tol"] for k, v in cc.items()})

    # (2) spectral thresholds for the same tolerances --------------------------------
    spl = {}
    Ug = np.linspace(1.0, 30.0, 117)
    for mn, m in models.items():
        for TI in cm.TI_values:
            em = []
            for U in Ug:
                spx = d.spectral(L, m, 0.01, U, TI, z)
                if not np.isfinite(spx["psib"]) or spx["keff"] <= 0:
                    break
                em.append(np.degrees(spx["expected_max"]))
            spl[f"{mn}|TI={TI}"] = {str(tol): crossing(Ug[:len(em)], em, tol) for tol in TOLS}
    out["spectral_limits"] = spl
    print("spectral", spl)

    # (3) Froya spectral shape ----------------------------------------------------------
    fr = {}
    for cname, Lc in (("mating", cm.L_mating), ("long", cm.L_ref), ("liftoff", cm.L_liftoff)):
        zc = z_payload(Lc)
        for mn, m in models.items():
            Ud = U_div(cm.suspension(Lc).k_psi, ref_slope(m))
            for U in ((8.0, 12.0) if cname == "mating" else (0.5 * Ud,)):
                a = d.spectral(Lc, m, 0.01, U, 0.06, zc)
                b = d.spectral(Lc, m, 0.01, U, 0.06, zc, spectrum="froya")
                fr[f"{cname}|{mn}|U={U:.2f}"] = dict(kaimal=float(np.degrees(a["sigma"])),
                                                    froya=float(np.degrees(b["sigma"])),
                                                    ratio=float(b["sigma"] / a["sigma"]))
    # turbulence intensity implied by the Froya model at the payload height (mating)
    fr["TI_froya_150m"] = {str(U): float(0.06 * (1 + 0.043 * U * (10 / 150) ** cm.alpha_shear)
                                         * (150 / 10) ** -0.22) for U in (8.0, 12.0)}
    out["froya"] = fr
    print("froya", json.dumps(fr, indent=0))

    # (4) aerodynamic damping of the box ---------------------------------------------
    ad = {}
    for U in (8.0, 12.0):
        a = d.spectral(L, models["box"], 0.01, U, 0.06, z)
        for fac in (0.0, 0.5, 2.0):
            b = d.spectral(L, models["box"], 0.01, U, 0.06, z, ca_factor=fac)
            ad[f"U={U}|ca x{fac}"] = float(b["sigma"] / a["sigma"])
        ad[f"U={U}|zeta_aero"] = float(d.params_vector(L, models["box"], 0.0, U)[3] /
                                      (2 * np.sqrt(cm.suspension(L).k_psi * cm.I_zz)))
    out["aero_damping_box"] = ad
    print("aero damping", ad)

    # (5) reduced frequency ----------------------------------------------------------
    rf = {}
    for cname, Lc in (("mating", cm.L_mating), ("liftoff", cm.L_liftoff)):
        s = cm.suspension(Lc)
        Ud = U_div(s.k_psi, CK)
        for Uh in (5.0, 8.0, 12.0):
            U = cm.wind_at(z_payload(Lc), Uh)
            f_eff = s.f_psi * np.sqrt(max(1 - (U / Ud) ** 2, 0.0))
            rf[f"{cname}|Uhub={Uh}"] = dict(U=float(U), k_still_air=float(np.pi * s.f_psi * cm.L_n / U),
                                            k_softened=float(np.pi * f_eff * cm.L_n / U))
    out["reduced_frequency"] = rf
    print("reduced frequency", json.dumps(rf, indent=0))
    out["wall_s"] = time.time() - t0
    with open(f"{cm.RES}/s5c_operability.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
