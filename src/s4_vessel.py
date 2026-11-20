"""
Section 5.3: floating crane vessel - hydrodynamics (Capytaine BEM), RAOs and
boom-tip motions.

Box-shaped hull with the main dimensions of the heavy-lift crane vessel Les Alizes
(236.8 m x 52 m, operating draught 9 m).
Rotation centre = centre of gravity.  Extra linear roll damping 5 % of
critical (viscous) and soft horizontal restoring (DP/mooring) giving 150 s
surge/sway/yaw natural periods are added outside the BEM.
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


def run():
    t0 = time.time()
    body = build_body()
    mass = cm.rho_w * LV * BV * TV
    body.mass = mass
    M = np.diag([mass, mass, mass, mass * RXX ** 2, mass * RYY ** 2, mass * RZZ ** 2])
    C = body.compute_hydrostatic_stiffness(rho=cm.rho_w, g=cm.g)
    C = np.array(C.values)
    omegas = np.linspace(0.15, 2.0, 75)
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
    wp = 2 * np.pi / cm.Tp
    w = omegas
    sig = np.where(w <= wp, 0.07, 0.09)
    S_pm = (5 / 16) * cm.Hs ** 2 * wp ** 4 / w ** 5 * np.exp(-1.25 * (wp / w) ** 4)
    Ag = 1 - 0.287 * np.log(cm.gamma_J)
    S = Ag * S_pm * cm.gamma_J ** np.exp(-0.5 * ((w - wp) / (sig * wp)) ** 2)
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
        S_wave=S.tolist(), rms=rms, wall_s=time.time() - t0)
    with open(f"{cm.RES}/s4_vessel.json", "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps(o["particulars"], indent=1))
    print(json.dumps(o["rms"], indent=1))
    print("wall", o["wall_s"])
