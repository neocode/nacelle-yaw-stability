"""
Section 5.3: floating crane vessel - hydrodynamics (Capytaine BEM), RAOs and
boom-tip motions.

Box-shaped hull with the main dimensions of the heavy-lift crane vessel Les Alizes
(236.8 m x 52 m, operating draught 9 m).
Rotation centre = centre of gravity.  Extra linear roll damping 5 % of
critical (viscous) and soft horizontal restoring (DP/mooring) giving 150 s
surge/sway/yaw natural periods are added outside the BEM.

Two-way coupling with the suspended load (Section 5.3): the load (payload and
hook block) hangs from the boom tip as a pendulum of length l_p = L + z_cg.  It
follows the tip vertically (inextensible hoist) and acts on it horizontally
through the rope, F = (W/l_p - i w c_p)(x_L - x_tip).  The weight of the load is
taken at the boom tip in the hydrostatics (KG includes it), so the coupled and
the one-way models have the same statics.  In the one-way model the load is
part of the vessel mass and its pendulum reaction on the vessel is ignored.
"""
import json
import time
import numpy as np
import xarray as xr
import capytaine as cpt
import common as cm

cpt.set_logging("ERROR")

LV, BV, TV = cm.V_LOA, cm.V_B, cm.V_T
KG = cm.V_KG                    # m  vertical CG above keel incl. crane and suspended load
ZG = KG - TV                    # m  CG above the waterline
# crane slew centre aft on the starboard side (ASSUMED), boom radius 29.5 m to starboard
R_TIP = np.array([-85.0, 14.0 + cm.AUX_RADIUS, cm.BOOM_TIP_HEIGHT - ZG])   # boom tip relative to the CG [m]
RXX, RYY, RZZ = 0.38 * BV, 0.25 * LV, 0.26 * LV
ZETA_ROLL = 0.05
T_HORIZ = 150.0


def build_body(res=(56, 14, 8)):
    hull = cpt.mesh_parallelepiped(size=(LV, BV, 2 * TV), resolution=res,
                                   center=(0.0, 0.0, 0.0), name="hull")
    hull = hull.immersed_part()
    lid = cpt.mesh_rectangle(size=(LV, BV), resolution=(res[0], res[1]),
                             center=(0.0, 0.0, -0.01), normal=(0.0, 0.0, 1.0))
    body = cpt.FloatingBody(mesh=hull, lid_mesh=lid,
                            dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, ZG)),
                            center_of_mass=(0, 0, ZG), name="vessel")
    return body


def skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


J_TIP = np.hstack([np.eye(3), -skew(R_TIP)])     # boom-tip displacement = J_TIP @ xi


def jonswap(w, Hs=cm.Hs, Tp=cm.Tp):
    wp = 2 * np.pi / Tp
    sig = np.where(w <= wp, 0.07, 0.09)
    S_pm = (5 / 16) * Hs ** 2 * wp ** 4 / w ** 5 * np.exp(-1.25 * (wp / w) ** 4)
    return (1 - 0.287 * np.log(cm.gamma_J)) * S_pm * cm.gamma_J ** np.exp(-0.5 * ((w - wp) / (sig * wp)) ** 2)


def pendulum(L, zeta_p=None):
    from s5_dynamics import ZETA_P
    zeta_p = ZETA_P if zeta_p is None else zeta_p
    mL = cm.m_p + cm.m_hb
    lp = L + cm.Z_CG_BELOW_BLOCK
    return mL, mL * cm.g / lp, 2 * zeta_p * mL * np.sqrt(cm.g / lp)


def coupled_rao(bem, L, zeta_p=None, rigid=False):
    """Vessel + pendulum (8 unknowns per frequency).  Returns the vessel RAO,
    the boom-tip RAO and the horizontal RAO of the load (absolute).
    rigid=True: load fixed to the boom tip (low-frequency limit)."""
    om, M, A, Bd, Bext, C, Kh, Fex = (bem[k] for k in ("omega", "M", "A", "B", "Bext", "C", "Kh", "Fex"))
    mL, Kp, cp = pendulum(L, zeta_p)
    Jh, Jv = J_TIP[:2], J_TIP[2:3]
    Mv = M - np.diag([mL, mL, mL, 0.0, 0.0, 0.0])
    nh = Fex.shape[1]
    rao = np.zeros((len(om), nh, 6), complex)
    xL = np.zeros((len(om), nh, 2), complex)
    for i, w in enumerate(om):
        Hv = -w ** 2 * (Mv + A[i]) - 1j * w * (Bd[i] + Bext) + C + Kh
        if rigid:
            H = Hv - w ** 2 * mL * J_TIP.T @ J_TIP
            for j in range(nh):
                rao[i, j] = np.linalg.solve(H, Fex[i, j])
                xL[i, j] = Jh @ rao[i, j]
            continue
        Kc = Kp - 1j * w * cp
        H = np.zeros((8, 8), complex)
        H[:6, :6] = Hv - w ** 2 * mL * Jv.T @ Jv + Kc * Jh.T @ Jh
        H[:6, 6:] = -Kc * Jh.T
        H[6:, :6] = -Kc * Jh
        H[6:, 6:] = (Kc - w ** 2 * mL) * np.eye(2)
        for j in range(nh):
            x = np.linalg.solve(H, np.r_[Fex[i, j], 0.0, 0.0])
            rao[i, j], xL[i, j] = x[:6], x[6:]
    tip = np.einsum("ij,fhj->fhi", J_TIP, rao)
    return rao, tip, xL


def coupled_modes(bem, L):
    """Undamped natural periods of the coupled vessel-pendulum system with the
    added mass at the natural frequency (fixed point), and the share of the
    kinetic energy in vessel roll and in the pendulum for each mode."""
    om, M, A, C, Kh = (bem[k] for k in ("omega", "M", "A", "C", "Kh"))
    mL, Kp, _ = pendulum(L)
    Jh, Jv = J_TIP[:2], J_TIP[2:3]
    Mv = M - np.diag([mL, mL, mL, 0.0, 0.0, 0.0])
    K8 = np.zeros((8, 8))
    K8[:6, :6] = C + Kh + Kp * Jh.T @ Jh
    K8[:6, 6:] = -Kp * Jh.T
    K8[6:, :6] = -Kp * Jh
    K8[6:, 6:] = Kp * np.eye(2)

    def eig(w):
        Ai = np.array([[np.interp(w, om, A[:, a, b]) for b in range(6)] for a in range(6)])
        M8 = np.zeros((8, 8))
        M8[:6, :6] = Mv + Ai + mL * Jv.T @ Jv
        M8[6:, 6:] = mL * np.eye(2)
        lam, V = np.linalg.eig(np.linalg.solve(M8, K8))
        o = np.argsort(lam.real)
        return np.sqrt(np.abs(lam.real[o])), V[:, o].real, M8

    out = []
    for j in range(8):
        w = 0.5
        for _ in range(60):
            wn, V, M8 = eig(w)
            w = 0.5 * w + 0.5 * max(wn[j], om[0])
        wn, V, M8 = eig(w)
        v = V[:, j]
        e = v * (M8 @ v)
        e = e / e.sum()
        out.append(dict(T=float(2 * np.pi / wn[j]), roll=float(e[3]), pendulum=float(e[6] + e[7]),
                        heave=float(e[2]), pitch=float(e[4]), sway=float(e[1]), surge=float(e[0]),
                        yaw=float(e[5])))
    return out


def run():
    t0 = time.time()
    body = build_body()
    mass = cm.rho_w * LV * BV * TV
    body.mass = mass
    M = np.diag([mass, mass, mass, mass * RXX ** 2, mass * RYY ** 2, mass * RZZ ** 2])
    C = body.compute_hydrostatic_stiffness(rho=cm.rho_w, g=cm.g)
    C = np.array(C.values)
    omegas = np.round(np.arange(0.15, 2.0001, 0.005), 4)
    headings = np.radians([90.0, 120.0, 150.0, 180.0])
    test_matrix = xr.Dataset(coords={"omega": omegas, "wave_direction": headings,
                                     "radiating_dof": list(body.dofs),
                                     "water_depth": [np.inf], "rho": [cm.rho_w]})
    ds = cpt.BEMSolver().fill_dataset(test_matrix, body, n_threads=16)
    ds = ds.squeeze()
    A = ds["added_mass"].transpose("omega", "influenced_dof", "radiating_dof").values
    Bd = ds["radiation_damping"].transpose("omega", "influenced_dof", "radiating_dof").values
    Fex = ds["excitation_force"].transpose("omega", "wave_direction", "influenced_dof").values
    dofs = list(body.dofs)
    # external terms
    w_low = 0
    Mtot_low = M + A[w_low]
    Kh = np.zeros((6, 6))
    for i in (0, 1, 5):
        Kh[i, i] = Mtot_low[i, i] * (2 * np.pi / T_HORIZ) ** 2
    Bext = np.zeros((6, 6))
    # roll: 5 % of critical at the roll natural frequency
    wn_roll = np.sqrt(C[3, 3] / (M[3, 3] + A[:, 3, 3].mean()))
    Bext[3, 3] = 2 * ZETA_ROLL * np.sqrt(C[3, 3] * (M[3, 3] + A[:, 3, 3].mean()))
    rao = np.zeros((len(omegas), len(headings), 6), complex)
    for i, w in enumerate(omegas):
        H = -w ** 2 * (M + A[i]) - 1j * w * (Bd[i] + Bext) + C + Kh
        for j in range(len(headings)):
            rao[i, j] = np.linalg.solve(H, Fex[i, j])
    bem = dict(omega=omegas, M=M, A=A, B=Bd, Bext=Bext, C=C, Kh=Kh, Fex=Fex)
    np.savez(f"{cm.RES}/s4_bem.npz", **bem)
    # natural periods
    Tn = {}
    for idx, nm in ((2, "heave"), (3, "roll"), (4, "pitch")):
        # solve C = w^2 (M+A(w)) by fixed point
        w = 0.6
        for _ in range(50):
            Ai = np.interp(w, omegas, A[:, idx, idx])
            w = np.sqrt(C[idx, idx] / (M[idx, idx] + Ai))
        Tn[nm] = 2 * np.pi / w
    # boom-tip motion: translation + rotation x r_tip ; block yaw = vessel yaw
    tip = rao[:, :, 0:3] + np.cross(rao[:, :, 3:6], R_TIP[None, None, :])
    # JONSWAP spectrum
    w = omegas
    S = jonswap(w)
    dw = np.gradient(w)
    rms = {}
    for j, h in enumerate(np.degrees(headings)):
        r = {}
        for k, nm in enumerate(["surge", "sway", "heave", "roll", "pitch", "yaw"]):
            v = np.sqrt(np.sum(np.abs(rao[:, j, k]) ** 2 * S * dw))
            r[nm] = float(np.degrees(v)) if k >= 3 else float(v)
        for k, nm in enumerate(["tip_x", "tip_y", "tip_z"]):
            r[nm] = float(np.sqrt(np.sum(np.abs(tip[:, j, k]) ** 2 * S * dw)))
        r["tip_az"] = float(np.sqrt(np.sum(np.abs(w ** 2 * tip[:, j, 2]) ** 2 * S * dw)))
        r["tip_yaw_rate"] = float(np.degrees(np.sqrt(np.sum(np.abs(w * rao[:, j, 5]) ** 2 * S * dw))))
        rms[str(int(round(h)))] = r
    out = dict(
        particulars=dict(L=LV, B=BV, T=TV, KG=KG, displacement_t=mass / 1e3,
                         GM_T=float(C[3, 3] / (mass * cm.g)), GM_L=float(C[4, 4] / (mass * cm.g)),
                         r_tip=R_TIP.tolist(), Tn=Tn, nfaces=body.mesh.nb_faces,
                         zeta_roll=ZETA_ROLL, T_horiz=T_HORIZ),
        omega=omegas.tolist(), headings_deg=np.degrees(headings).tolist(),
        rao_abs=np.abs(rao).tolist(), rao_re=rao.real.tolist(), rao_im=rao.imag.tolist(),
        tip_abs=np.abs(tip).tolist(), tip_re=tip.real.tolist(), tip_im=tip.imag.tolist(),
        S_wave=S.tolist(), rms=rms)
    out["coupled"] = coupling(bem, rao, tip)
    out["wall_s"] = time.time() - t0
    with open(f"{cm.RES}/s4_vessel.json", "w") as fh:
        json.dump(out, fh)
    return out


def coupling(bem, rao1, tip1):
    """One-way versus two-way coupled responses for the stages of the lift."""
    w = bem["omega"]
    S = jonswap(w)
    hd = [90, 120, 150, 180]
    sd = lambda Z: float(np.sqrt(np.trapezoid(np.abs(Z) ** 2 * S, w)))
    res = {}
    for L in (cm.L_mating, cm.L_ref, cm.L_liftoff):
        mL, Kp, cp = pendulum(L)
        G = (Kp - 1j * w * cp) / (Kp - 1j * w * cp - w ** 2 * mL)          # x_L / x_tip
        rao2, tip2, xL2 = coupled_rao(bem, L)
        rao3, tip3, _ = coupled_rao(bem, L, rigid=True)
        r = dict(T_pend=float(2 * np.pi * np.sqrt((L + cm.Z_CG_BELOW_BLOCK) / cm.g)),
                 modes=coupled_modes(bem, L),
                 tip_re=tip2.real.tolist(), tip_im=tip2.imag.tolist(),
                 rao_re=rao2.real.tolist(), rao_im=rao2.imag.tolist(),
                 xL_abs=np.abs(xL2).tolist(), tip_rigid_abs=np.abs(tip3).tolist(), rms={})
        for j, h in enumerate(hd):
            one_rel = np.hypot(sd((G - 1) * tip1[:, j, 0]), sd((G - 1) * tip1[:, j, 1]))
            two_rel = np.hypot(sd(xL2[:, j, 0] - tip2[:, j, 0]), sd(xL2[:, j, 1] - tip2[:, j, 1]))
            r["rms"][str(h)] = dict(
                roll_one=float(np.degrees(sd(rao1[:, j, 3]))), roll_two=float(np.degrees(sd(rao2[:, j, 3]))),
                roll_rigid=float(np.degrees(sd(rao3[:, j, 3]))),
                yaw_one=float(np.degrees(sd(rao1[:, j, 5]))), yaw_two=float(np.degrees(sd(rao2[:, j, 5]))),
                tip_one=float(np.hypot(sd(tip1[:, j, 0]), sd(tip1[:, j, 1]))),
                tip_two=float(np.hypot(sd(tip2[:, j, 0]), sd(tip2[:, j, 1]))),
                sway_rel_one=float(one_rel), sway_rel_two=float(two_rel),
                pend_angle_one=float(np.degrees(one_rel / (L + cm.Z_CG_BELOW_BLOCK))),
                pend_angle_two=float(np.degrees(two_rel / (L + cm.Z_CG_BELOW_BLOCK))))
        res[str(int(L))] = r
    return res


def coupling_from_cache():
    """Recompute only the coupled part from the saved BEM coefficients."""
    bem = dict(np.load(f"{cm.RES}/s4_bem.npz"))
    with open(f"{cm.RES}/s4_vessel.json") as fh:
        out = json.load(fh)
    rao1 = np.array(out["rao_re"]) + 1j * np.array(out["rao_im"])
    tip1 = np.array(out["tip_re"]) + 1j * np.array(out["tip_im"])
    out["coupled"] = coupling(bem, rao1, tip1)
    with open(f"{cm.RES}/s4_vessel.json", "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    import sys
    o = coupling_from_cache() if "coupled" in sys.argv else run()
    print(json.dumps(o["particulars"], indent=1))
    print(json.dumps(o["rms"], indent=1))
    for L, r in o["coupled"].items():
        print("L =", L, "T_pend =", round(r["T_pend"], 2))
        for md in r["modes"]:
            print("   ", {k: round(v, 3) for k, v in md.items()})
        print(json.dumps(r["rms"], indent=0))
    print("wall", o.get("wall_s"))
