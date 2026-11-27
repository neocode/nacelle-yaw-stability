"""
Section 5.3: sensitivity of the coupled roll-pendulum modes and of the swell-induced
yaw response at mating to the assumed vessel properties.

Variants (s4_vessel.apply_variant): roll radius of gyration x0.8-1.2, centre of gravity
+-2 m, viscous roll damping 2.5 % and 10 %, crane amidships; pendulum damping
0.002 and 0.05.  For each: one-way roll period, coupled mode periods, and the change of
sigma_psi at 8 m/s in a swell (Hs = 2 m, beam and 120 deg seas) with the peak period
of the baseline coupled mode (18.6 s) and of the variant's own coupled mode.
"""
import json
import time
import numpy as np
import common as cm
import s4_vessel as v
import s5_dynamics as d
from aero_models import Munk
from s3_statics import z_payload

VARIANTS = {
    "baseline": {},
    "rxx=0.8": {"rxx": 0.8}, "rxx=0.9": {"rxx": 0.9}, "rxx=1.1": {"rxx": 1.1}, "rxx=1.2": {"rxx": 1.2},
    "KG-2": {"dKG": -2.0}, "KG+2": {"dKG": 2.0},
    "zeta_roll=0.025": {"zeta_roll": 0.025}, "zeta_roll=0.10": {"zeta_roll": 0.10},
    "crane amidships": {"tip_dx": 85.0},
}
ZETA_P_CASES = (0.002, 0.05)


def roll_period(bem):
    M, C, A, om = bem["M"], bem["C"], bem["A"], bem["omega"]
    dKG = bem.get("rg", np.zeros(3))[2]
    w = 0.4
    for _ in range(80):
        w = np.sqrt(C[3, 3] / (M[3, 3] + M[0, 0] * dKG ** 2 + np.interp(w, om, A[:, 3, 3])))
    return float(2 * np.pi / w)


def run(seeds=range(10)):
    t0 = time.time()
    L = cm.L_mating
    z = z_payload(L)
    CK = json.load(open(f"{cm.RES}/s2_aero_bem.json"))["CK"]["bem_box"]
    models = {"munk": Munk(CK), "box": d.box_model()}
    bem0 = dict(np.load(f"{cm.RES}/s4_bem.npz"))
    base = {mn: d.simulate(L, m, 0.01, 8.0, 0.06, z, seeds) for mn, m in models.items()}
    out = {}
    cases = [(k, var, d.ZETA_P) for k, var in VARIANTS.items()] + \
            [(f"zeta_p={zp}", {}, zp) for zp in ZETA_P_CASES]
    T_base = None
    for name, var, zp in cases:
        bem, r_tip = v.apply_variant(bem0, var)
        md = v.coupled_modes(bem, L, r_tip)
        roll_modes = sorted([x for x in md if x["roll"] > 0.1], key=lambda x: -x["T"])
        T1 = roll_modes[0]["T"]
        if name == "baseline":
            T_base = T1
        row = dict(T_roll_one_way=roll_period(bem), T_coupled=[x["T"] for x in roll_modes],
                   shares=[(x["roll"], x["pendulum"]) for x in roll_modes])
        for Tsw_name, Tsw in (("Tp=base", T_base), ("Tp=own", T1)):
            if Tsw_name == "Tp=own" and name in ("baseline",) + tuple(f"zeta_p={zp}" for zp in ZETA_P_CASES):
                continue
            for hd in (90.0, 120.0):
                for mn, m in models.items():
                    st = d.simulate(L, m, 0.01, 8.0, 0.06, z, seeds, vessel=True, heading=hd, Hs=2.0,
                                    Tp=Tsw, zeta_p=zp, vessel_variant=var or None)
                    row[f"{Tsw_name}|hd={hd}|{mn}"] = dict(
                        Tp=Tsw, diff_pct=d.paired(base[mn], st), std=d.summarize(st, "std"),
                        sway_rel=d.summarize(st, "sway_rel_std"))
        out[name] = row
        print(name, round(row["T_roll_one_way"], 2), [round(x, 2) for x in row["T_coupled"]],
              {k: round(val["diff_pct"][0], 1) for k, val in row.items() if isinstance(val, dict)},
              round(time.time() - t0))
    out["base_sigma"] = {mn: d.summarize(st, "std") for mn, st in base.items()}
    with open(f"{cm.RES}/s4b_vessel_sensitivity.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
