"""
Section 4.2: potential-flow (Munk) yaw stiffness of the payload.

Added masses in an unbounded fluid are computed with the boundary-element
code Capytaine (free_surface = inf) for
  * a prolate spheroid a/b = 2 (verification against Lamb's exact values),
  * superellipsoids |x/a|^n+|y/b|^n+|z/c|^n = 1, n = 2, 4, 8 (edge rounding),
  * the rectangular envelope 20 x 10 x 10 m (n -> infinity).
The Munk yaw stiffness coefficient is
      C_K = K_a / (q A_s L) = 2 (m22 - m11) / (rho A_s L),
with A_s = L H the side-face area and L the length.
"""
import json
import time
import numpy as np
import capytaine as cpt
import common as cm

cpt.set_logging("ERROR")


def lamb_prolate(ar):
    """Exact Lamb (1932, Art. 115) added-mass coefficients of a prolate spheroid."""
    e = np.sqrt(1 - 1 / ar ** 2)
    lg = np.log((1 + e) / (1 - e))
    a0 = 2 * (1 - e ** 2) / e ** 3 * (0.5 * lg - e)
    b0 = 1 / e ** 2 - (1 - e ** 2) / (2 * e ** 3) * lg
    return a0 / (2 - a0), b0 / (2 - b0)


def mesh_superellipsoid(a, b, c, n, res):
    """Radial projection of a sphere mesh onto a superellipsoid surface."""
    sph = cpt.mesh_sphere(radius=1.0, center=(0, 0, 0), resolution=res)
    v = sph.vertices.copy()
    u = v / np.linalg.norm(v, axis=1)[:, None]
    s = (np.abs(u[:, 0] / a) ** n + np.abs(u[:, 1] / b) ** n
         + np.abs(u[:, 2] / c) ** n) ** (-1.0 / n)
    return cpt.Mesh(vertices=u * s[:, None], faces=sph.faces, name=f"se{n}")


def added_masses(mesh):
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))
    solver = cpt.BEMSolver()
    out = {}
    for dof in ("Surge", "Sway"):
        pb = cpt.RadiationProblem(body=body, radiating_dof=dof, free_surface=np.inf,
                                  water_depth=np.inf, omega=1.0, rho=1.0)
        r = solver.solve(pb, keep_details=False)
        out[dof] = float(r.added_masses[dof])
    return out["Surge"], out["Sway"], float(mesh.volume), mesh.nb_faces


def run():
    L, W, H = cm.L_n, cm.W_n, cm.H_n
    As = L * H
    res = {}
    k1, k2 = lamb_prolate(2.0)
    res["lamb"] = dict(k1=k1, k2=k2, dk=k2 - k1)
    # ------------------------------------------------ spheroid convergence
    sph = []
    for rr in ((16, 24), (24, 36), (36, 54), (52, 78)):
        t = time.time()
        m = mesh_superellipsoid(L / 2, W / 2, H / 2, 2.0, rr)
        m11, m22, Vm, nf = added_masses(m)
        Vex = np.pi / 6 * L * W * H
        sph.append(dict(nf=nf, k1=m11 / Vex, k2=m22 / Vex, k1_mv=m11 / Vm, k2_mv=m22 / Vm,
                        t=time.time() - t))
        print("spheroid", sph[-1])
    res["spheroid"] = sph
    # ------------------------------------------------ superellipsoids n = 4, 8
    se = {}
    for nexp in (4.0, 8.0):
        rows = []
        for rr in ((24, 36), (36, 54), (52, 78)):
            m = mesh_superellipsoid(L / 2, W / 2, H / 2, nexp, rr)
            m11, m22, Vm, nf = added_masses(m)
            rows.append(dict(nf=nf, m11=m11, m22=m22, V=Vm,
                             CK=2 * (m22 - m11) / (As * L)))
            print("superellipsoid", nexp, rows[-1])
        se[str(nexp)] = rows
    res["superellipsoid"] = se
    # ------------------------------------------------ rectangular box
    box = []
    for rr in ((10, 5, 5), (16, 8, 8), (24, 12, 12), (32, 16, 16)):
        m = cpt.mesh_parallelepiped(size=(L, W, H), resolution=rr, center=(0, 0, 0))
        m11, m22, Vm, nf = added_masses(m)
        box.append(dict(nf=nf, m11=m11, m22=m22, V=Vm, k1=m11 / Vm, k2=m22 / Vm,
                        CK=2 * (m22 - m11) / (As * L)))
        print("box", box[-1])
    res["box"] = box
    # Richardson extrapolation (panel size h ~ nf^-1/2, first order assumed for edges)
    nf = np.array([b["nf"] for b in box], float)
    ck = np.array([b["CK"] for b in box])
    hh = nf ** -0.5
    A = np.c_[np.ones_like(hh), hh]
    coef = np.linalg.lstsq(A[-3:], ck[-3:], rcond=None)[0]
    res["box_CK_extrap"] = float(coef[0])
    # ------------------------------------------------ summary of C_K estimates
    CK = {}
    CK["munk_spheroid_own_volume"] = 2 * (np.pi / 6) * (W / L) * (k2 - k1)
    CK["munk_spheroid_dk_box_volume"] = 2 * (W / L) * (k2 - k1)   # original manuscript
    CK["bem_box"] = float(coef[0])
    CK["bem_superellipsoid_n4"] = se["4.0"][-1]["CK"]
    CK["bem_superellipsoid_n8"] = se["8.0"][-1]["CK"]
    # Empirical sharp-edged box, Greenwell (2008) Eq. 7, 3D containers:
    #   C*_YM = c_s (1-(W/L)^2) sin 2psi - c_a (W/L) sin 4psi, side-face area and length
    r = W / L
    for tag, cs, ca in (("box_empirical_low", 0.075, 0.04), ("box_empirical_mid", 0.09, 0.04),
                        ("box_empirical_high", 0.09, 0.03)):
        CK[tag] = 2 * cs * (1 - r ** 2) - 4 * ca * r
    res["CK"] = CK
    with open(f"{cm.RES}/s2_aero_bem.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["CK"], indent=1))
    print("lamb", r["lamb"])
