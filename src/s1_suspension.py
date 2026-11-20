"""
Section 3: torsional restoring moment of a multi-fall suspension.

(1) Closed-form exact moment for an arbitrary fall arrangement and tension
    distribution (Eqs. (3)-(5) of the paper).
(2) Independent numerical check: full static equilibrium of a rigid lower
    block hanging on n straight falls (6 unknowns: block position, roll,
    pitch and rope tension; yaw prescribed), solved with Newton iterations.
(3) Sheave friction: geometric tension distribution T_i ~ eta^(i-1).
(4) External check against the wire-rope cabling formula (Bridon worked example).
(5) Bound on the neglected elastic torsional stiffness of the ropes.
"""
import json
import numpy as np
from scipy.optimize import least_squares
import common as cm


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Rxyz(phi, theta, psi):
    cx, sx, cy, sy = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz(psi) @ Ry @ Rx


# ---------------------------------------------------------------- closed form
def moment_closed_form(psi, a, b, w, W, ell_tot):
    """Exact yaw moment of the falls on a level block, Eq. (9).

    a, b : (n,2) upper / lower fall positions;  w: tension weights (sum 1);
    W: suspended weight; ell_tot: total length of the falls (inextensible).
    h(psi) follows from sum_i sqrt(h^2 + |a_i - R b_i|^2) = ell_tot.
    Returns (M_z, h)."""
    from scipy.optimize import brentq
    Rb = (Rz(psi)[:2, :2] @ b.T).T
    d2 = np.sum((a - Rb) ** 2, axis=1)
    h = brentq(lambda hh: np.sum(np.sqrt(hh ** 2 + d2)) - ell_tot, 1e-6, ell_tot)
    ab = np.einsum("ij,ij->i", a, b)
    bxa = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]
    # vertical tension components T_v,i = w_i W; exact moment of straight falls
    M = (W / h) * np.sum(w * (bxa * np.cos(psi) - ab * np.sin(psi)))
    return M, h


# ---------------------------------------------------- numerical 3D equilibrium
class BlockEquilibrium:
    """Rigid lower block + payload on n straight falls of one continuous rope.

    Upper fall ends a_i are fixed at z = 0 (z positive DOWN).  Lower ends are
    b_i in the block frame (at the block reference point).  The weight W acts at
    z_cg below the block reference.  Tension ratios between consecutive falls are
    eta (eta = 1: frictionless sheaves, equal tension).  The rope is inextensible:
    the sum of the fall lengths equals ell_tot.
    Unknowns q = [x, y, z, phi, theta, T1] for a prescribed yaw psi."""

    def __init__(self, a, b, W, h0, z_cg, eta=1.0):
        self.a3 = np.c_[a, np.zeros(len(a))]
        self.b3 = np.c_[b, np.zeros(len(b))]
        self.W, self.z_cg = W, z_cg
        n = len(a)
        self.ratio = eta ** np.arange(n)
        self.ratio = self.ratio / self.ratio.sum()
        # reference state: block centred, level, at depth h0
        self.ell_tot = float(np.sum(np.linalg.norm(
            self.a3 - (self.b3 + [0, 0, h0]), axis=1)))
        self.h0 = h0

    def residual(self, q, psi):
        x, y, z, phi, th, T1 = q
        R = Rxyz(phi, th, psi)
        p0 = np.array([x, y, z])
        bw = (R @ self.b3.T).T + p0                     # lower ends, world
        vec = self.a3 - bw                               # lower -> upper
        ell = np.linalg.norm(vec, axis=1)
        T = T1 * self.ratio / self.ratio[0]
        F = (T[:, None] * vec / ell[:, None])            # forces on block
        rcg = R @ np.array([0.0, 0.0, self.z_cg])        # CG offset (down)
        Fsum = F.sum(axis=0) + np.array([0.0, 0.0, self.W])
        Msum = np.sum(np.cross(bw - p0, F), axis=0) + np.cross(rcg, [0.0, 0.0, self.W])
        con = (ell.sum() - self.ell_tot) / self.ell_tot
        s = self.W
        return np.r_[Fsum / s, Msum[:2] / (s * 1.0), con * 1e3], Msum[2]

    def solve(self, psi, q0=None):
        if q0 is None:
            q0 = np.array([0.0, 0.0, self.h0, 0.0, 0.0, self.W / len(self.a3)])
        f = lambda q: self.residual(q, psi)[0]
        sol = least_squares(f, q0, xtol=1e-15, ftol=1e-15, gtol=1e-15, method="lm")
        res, Mz = self.residual(sol.x, psi)
        return sol.x, Mz, float(np.max(np.abs(res)))


def run():
    out = {}
    W_load = (cm.m_p + cm.m_hb) * cm.g
    z_cg = 17.0                                     # m, CG below the block (yoke + half nacelle)
    a_axle = cm.fall_positions()
    n = len(a_axle)
    ang = 2 * np.pi * np.arange(n) / n
    a_circ = 1.2 * np.c_[np.cos(ang), np.sin(ang)]  # idealised fall circle S = 2.4 m

    # ---------------- (1)+(2) moment curve, closed form vs 3D equilibrium
    psis = np.radians(np.r_[0.5, 1, 2, 5, 10, 20, 30, 45, 60, 75, 90, 105, 120, 150])
    cases = {}
    for name, a in (("circle", a_circ), ("axle", a_axle)):
        for L in (cm.L_mating, cm.L_ref):
            for eta in (1.0, 0.98, 0.96):
                eq = BlockEquilibrium(a, a, W_load, L, z_cg, eta)
                w = eq.ratio
                Mnum, Mcf, Mlin, hs, resmax = [], [], [], [], []
                q = None
                for p in psis:
                    q, Mz, r = eq.solve(p, q)
                    Mnum.append(-Mz)                    # restoring moment (applied = -Mz)
                    M, h = moment_closed_form(p, a, a, w, W_load, eq.ell_tot)
                    Mcf.append(M)
                    hs.append(h)
                    resmax.append(r)
                Mnum = np.array(Mnum)
                # Note: Mz returned is the moment of the fall forces on the block;
                # the restoring moment is Mz itself (sign check below)
                Mnum = -Mnum
                k_cf = W_load * cm.r_eq2(a, a, w) / L
                cases[f"{name}_L{int(L)}_eta{eta}"] = dict(
                    psi_deg=np.degrees(psis).tolist(), M_num=Mnum.tolist(),
                    M_cf=np.array(Mcf).tolist(), h=hs, res=max(resmax),
                    k_cf=k_cf, r_eq=float(np.sqrt(cm.r_eq2(a, a, w))),
                    k_num_small=float(-Mnum[0] / np.sin(psis[0])))
    out["moment_cases"] = cases

    # ---------------- arc approximation vs chord (exact) restoring moment
    L = cm.L_ref
    r = 1.2
    ps = np.radians(np.linspace(0, 120, 121))
    k = W_load * r ** 2 / L
    M_exact = -k * np.sin(ps) / np.sqrt(1 - 2 * (r / L) ** 2 * (1 - np.cos(ps)))
    with np.errstate(invalid="ignore"):
        M_arc = -W_load * r ** 2 * ps / np.sqrt(L ** 2 - (r * ps) ** 2)
    out["arc_vs_chord"] = dict(psi_deg=np.degrees(ps).tolist(),
                               M_exact=(M_exact / k).tolist(),
                               M_arc=(M_arc / k).tolist(),
                               M_lin=(-ps).tolist())

    # ---------------- (4) external check: Bridon worked example (two falls)
    C, d, S, Lb = 0.0076, 0.020, 0.360, 30.0
    psi_b = np.degrees(np.arcsin(4 * C * d * Lb / S ** 2))
    Lcab = S ** 2 / (4 * C * d)
    # same with the 3D equilibrium solver: two falls, find psi where restoring = rope torque
    a2 = np.array([[+S / 2, 0.0], [-S / 2, 0.0]])
    Wt = 1.0e5
    eq2 = BlockEquilibrium(a2, a2, Wt, Lb, 1.0, 1.0)
    from scipy.optimize import brentq
    Mr = C * Wt * d
    f = lambda p: -(-eq2.solve(p)[1]) - Mr    # restoring magnitude minus rope torque
    psi_num = np.degrees(brentq(lambda p: eq2.solve(p)[1] * -1 * -1 - Mr if False else
                                (-eq2.solve(p)[1]) - Mr, 1e-4, np.radians(80)))
    out["bridon"] = dict(psi_formula_deg=psi_b, psi_3d_deg=float(psi_num),
                         psi_published="8 deg 10 min", L_cab=Lcab, L_cab_published=213.0)

    # ---------------- (5) elastic torsional stiffness of the ropes (bound)
    G = 79.3e9
    J = np.pi * cm.d_rope ** 4 / 32
    GJ_solid = G * J
    res5 = {}
    for L in (cm.L_mating, cm.L_ref, cm.L_liftoff):
        kpsi = cm.suspension(L).k_psi
        res5[str(int(L))] = dict(
            k_psi=kpsi,
            upper_solid_LeqL=n * GJ_solid / L / kpsi,               # solid steel, twist confined to falls
            realistic_confined=n * 0.15 * GJ_solid / L / kpsi,       # G_eff ~ 0.15 G
            realistic_spread=0.15 * GJ_solid / (L) / kpsi)           # twist spreads over n L of rope
    out["elastic_bound"] = res5

    # ---------------- along the lift: k, f, T, psi0
    Ls = np.linspace(cm.L_mating, cm.L_liftoff, 27)
    out["along_lift"] = dict(
        L=Ls.tolist(),
        k=[cm.suspension(L).k_psi for L in Ls],
        f=[cm.suspension(L).f_psi for L in Ls],
        psi0_deg=[np.degrees(cm.suspension(L).psi0) for L in Ls],
        f_pend=[cm.suspension(L).f_pend for L in Ls])

    # ---------------- equivalent spacing and tension-distribution sensitivity
    out["r_eq"] = dict(
        circle=float(np.sqrt(cm.r_eq2(a_circ))),
        axle=float(np.sqrt(cm.r_eq2(a_axle))),
        axle_eta098=float(np.sqrt(cm.r_eq2(a_axle, a_axle, 0.98 ** np.arange(n)))),
        axle_eta096=float(np.sqrt(cm.r_eq2(a_axle, a_axle, 0.96 ** np.arange(n)))))
    with open(f"{cm.RES}/s1_suspension.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    o = run()
    for k_, v in o["moment_cases"].items():
        Mn, Mc = np.array(v["M_num"]), np.array(v["M_cf"])
        rel = np.max(np.abs(Mn - Mc) / np.maximum(np.abs(Mc), 1e-9))
        print(f"{k_:24s} k_cf={v['k_cf']:.5g} k_num={v['k_num_small']:.5g} "
              f"max rel diff={rel:.2e} res={v['res']:.1e}")
    print("Bridon:", o["bridon"])
    print("elastic:", json.dumps(o["elastic_bound"], indent=0))
    print("r_eq:", o["r_eq"])
