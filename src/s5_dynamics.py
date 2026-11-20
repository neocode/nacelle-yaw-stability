"""
Section 5: dynamic yaw response under turbulent wind and vessel motion.

Reduced-order model (3 DOF): payload yaw psi + horizontal pendulum motion (x, y)
    I psi''  = -k_psi (1 + a_z/g) sin(psi - psi_v) - c_s (psi' - psi_v') - c_a psi'
               + q_rel A_s L c(psi - theta_rel) + M_r + M_t
    m x''    = -(W (1 + a_z/g)/l_p) (x - x_t) + F_drag,x     (same for y)
with relative wind (U + u - x', v - y'), theta_rel its direction, a_z the
boom-tip vertical acceleration, (x_t, y_t) the boom-tip horizontal motion and
psi_v the vessel (upper block) yaw.

Frequency-domain counterpart (linearised about the mean equilibrium) gives the
response spectrum, used both as the prediction and as a verification of the
time-domain integrator.
"""
import json
import time
import numpy as np
from numba import njit
import common as cm
from aero_models import Munk, BluffBox, U_div
from s3_statics import Z_BLOCK_CG, z_payload

CDA = 1.05 * cm.W_n * cm.H_n        # m^2  end-on drag area of the payload
CYB = 1.35                          # side-force slope (Greenwell 2008), for strip damping


def kaimal(f, sigma, Lk, U):
    """IEC 61400-1 Kaimal one-sided spectrum [ (m/s)^2 / Hz ]."""
    x = f * Lk / U
    return sigma ** 2 * 4 * Lk / U / (1 + 6 * x) ** (5 / 3)


def turbulence(seed, T, dt, U, TI, z):
    """Point time series u(t), v(t) by spectral synthesis (random phases, FFT)."""
    rng = np.random.default_rng(seed)
    n = int(round(T / dt))
    f = np.fft.rfftfreq(n, dt)
    df = f[1]
    lam = cm.Lambda1(z)
    out = []
    for sig, Lk in ((TI * U, 8.1 * lam), (cm.r_sigma_v * TI * U, 2.7 * lam)):
        S = np.zeros_like(f)
        S[1:] = kaimal(f[1:], sig, Lk, U)
        S[-1] = 0.0                                   # drop the Nyquist bin
        amp = np.sqrt(2 * S * df)
        ph = rng.uniform(0, 2 * np.pi, len(f))
        X = amp * np.exp(1j * ph) * n / 2
        out.append(np.fft.irfft(X, n))
    return out[0], out[1]


def vessel_inputs(seed, T, dt, heading_deg=150.0, Hs=cm.Hs):
    """Boom-tip horizontal motion, vertical acceleration and vessel yaw [rad]
    from the BEM RAOs and a JONSWAP sea (random phases)."""
    with open(f"{cm.RES}/s4_vessel.json") as fh:
        V = json.load(fh)
    w_r = np.array(V["omega"])
    j = V["headings_deg"].index(min(V["headings_deg"], key=lambda h: abs(h - heading_deg)))
    rao = np.array(V["rao_re"])[:, j, :] + 1j * np.array(V["rao_im"])[:, j, :]
    tip = np.array(V["tip_re"])[:, j, :] + 1j * np.array(V["tip_im"])[:, j, :]
    rng = np.random.default_rng(10_000 + seed)
    n = int(round(T / dt))
    dw = 2 * np.pi / T
    w = np.fft.rfftfreq(n, dt) * 2 * np.pi            # multiples of dw
    act = (w >= w_r[0]) & (w <= w_r[-1])
    wa = np.where(act, w, 1.0)
    wp = 2 * np.pi / cm.Tp
    sig = np.where(wa <= wp, 0.07, 0.09)
    S = ((1 - 0.287 * np.log(cm.gamma_J)) * (5 / 16) * Hs ** 2 * wp ** 4 / wa ** 5
         * np.exp(-1.25 * (wp / wa) ** 4) * cm.gamma_J ** np.exp(-0.5 * ((wa - wp) / (sig * wp)) ** 2))
    S = np.where(act, S, 0.0)
    a = np.sqrt(2 * S * dw)
    ph = rng.uniform(0, 2 * np.pi, len(w))

    def interp_c(Z):
        return np.interp(wa, w_r, Z.real) + 1j * np.interp(wa, w_r, Z.imag)

    def synth(Z):
        # x(t) = Re sum a_k Z_k exp(i(w_k t + ph_k))
        return np.fft.irfft(a * Z * np.exp(1j * ph) * n / 2, n)

    xt = synth(interp_c(tip[:, 0]))
    yt = synth(interp_c(tip[:, 1]))
    azt = synth(-w ** 2 * interp_c(tip[:, 2]))
    psv = synth(interp_c(rao[:, 5]))
    dpsv = synth(1j * w * interp_c(rao[:, 5]))
    return xt, yt, azt, psv, dpsv


# --------------------------------------------------------------- integrator
@njit(cache=True)
def _aero_c(beta, kind, cs, ca, cyx, CK):
    """kind 0: Munk  c = CK/2 sin2b ; kind 1: box fit."""
    if kind == 0:
        return 0.5 * CK * np.sin(2 * beta)
    r = 0.5
    return cs * (1 - r * r) * np.sin(2 * beta) - ca * r * np.sin(4 * beta) - cyx * np.sin(beta)


@njit(cache=True)
def _rhs(s, U, u, v, xt, yt, azt, psv, dpsv, P):
    # P: [k, I, cs_damp, ca_damp, Mr, m, W, lp, rhoAsL_half, CDA_half_rho, kind, cs, ca, cyx, CK,
    #     kp, kd, Mt_max, psi_ref, ctrl]
    psi, dpsi, x, dx, y, dy = s[0], s[1], s[2], s[3], s[4], s[5]
    k = P[0]; I = P[1]
    ux = U + u - dx
    uy = v - dy
    sp2 = ux * ux + uy * uy
    th = np.arctan2(uy, ux)
    beta = psi - th
    Ma = P[8] * sp2 * _aero_c(beta, int(P[10]), P[11], P[12], P[13], P[14])
    fac = 1.0 + azt / 9.80665
    Mt = 0.0
    if P[19] > 0.5:
        Mt = -P[15] * (psi - P[18]) - P[16] * dpsi
        if Mt > P[17]:
            Mt = P[17]
        elif Mt < -P[17]:
            Mt = -P[17]
    ddpsi = (-k * fac * np.sin(psi - psv) - P[2] * (dpsi - dpsv) - P[3] * dpsi
             + Ma + P[4] + Mt) / I
    sp = np.sqrt(sp2)
    kx = P[6] * fac / P[7]
    ddx = (-kx * (x - xt) + P[9] * sp * ux) / P[5]
    ddy = (-kx * (y - yt) + P[9] * sp * uy) / P[5]
    return np.array([dpsi, ddpsi, dx, ddx, dy, ddy])


@njit(cache=True)
def integrate(s0, dt, U, u, v, xt, yt, azt, psv, dpsv, P):
    """RK4 with inputs sampled on the integration grid (midpoints averaged)."""
    n = len(u)
    out = np.empty((n, 6))
    s = s0.copy()
    out[0] = s
    for i in range(n - 1):
        um = 0.5 * (u[i] + u[i + 1]); vm = 0.5 * (v[i] + v[i + 1])
        xm = 0.5 * (xt[i] + xt[i + 1]); ym = 0.5 * (yt[i] + yt[i + 1])
        am = 0.5 * (azt[i] + azt[i + 1]); pm = 0.5 * (psv[i] + psv[i + 1])
        dm = 0.5 * (dpsv[i] + dpsv[i + 1])
        k1 = _rhs(s, U, u[i], v[i], xt[i], yt[i], azt[i], psv[i], dpsv[i], P)
        k2 = _rhs(s + 0.5 * dt * k1, U, um, vm, xm, ym, am, pm, dm, P)
        k3 = _rhs(s + 0.5 * dt * k2, U, um, vm, xm, ym, am, pm, dm, P)
        k4 = _rhs(s + dt * k3, U, u[i + 1], v[i + 1], xt[i + 1], yt[i + 1], azt[i + 1],
                  psv[i + 1], dpsv[i + 1], P)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


def params_vector(L, model, zeta_s, U, control=False, kp=cm.k_p_nom, kd=cm.k_d_nom,
                  rope=True, aero_damping=True):
    s = cm.suspension(L)
    c_s = 2 * zeta_s * np.sqrt(s.k_psi * cm.I_zz)
    kind = 0 if isinstance(model, Munk) else 1
    c_a = 0.0
    if kind == 1 and aero_damping:
        # quasi-steady strip estimate of the yaw damping of a box with linear side force
        c_a = cm.rho_air * U * cm.H_n * cm.L_n ** 3 * CYB / 24.0
    cs_ = getattr(model, "cs", 0.0); ca_ = getattr(model, "ca", 0.0)
    cyx = getattr(model, "CYb", 0.0) * getattr(model, "xs", 0.0) / cm.L_n
    m = cm.m_p + cm.m_hb
    return np.array([s.k_psi, cm.I_zz, c_s, c_a, s.M_r if rope else 0.0, m, s.W_load,
                     L + Z_BLOCK_CG, 0.5 * cm.rho_air * cm.A_side * cm.L_n,
                     0.5 * cm.rho_air * CDA, kind, cs_, ca_, cyx, model.CK,
                     kp, kd, cm.M_t_max, 0.0, 1.0 if control else 0.0])


def static_equilibrium(P, U):
    """Mean yaw from the full nonlinear static balance (continuation from 0)."""
    from scipy.optimize import brentq
    k, Mr = P[0], P[4]
    f = lambda p: (-k * np.sin(p) + P[8] * U ** 2 * float(_aero_c(p, int(P[10]), P[11], P[12],
                                                                  P[13], P[14])) + Mr
                   + (-(P[15]) * p if P[19] > 0.5 else 0.0))
    # primary branch: the stable root closest to the still-air twist, found by
    # continuation in U (a fold of the primary branch returns NaN)
    psi = np.arcsin(min(Mr / k, 1.0))
    for Ui in np.linspace(0.0, U, 41)[1:]:
        g = lambda p: (-k * np.sin(p) + P[8] * Ui ** 2 * float(_aero_c(p, int(P[10]), P[11], P[12],
                                                                       P[13], P[14])) + Mr
                       + (-(P[15]) * p if P[19] > 0.5 else 0.0))
        grid = np.linspace(psi - 0.15, psi + 0.15, 301)
        G = np.array([g(x) for x in grid])
        idx = [i for i in range(300) if G[i] > 0 >= G[i + 1]]     # stable crossings (dM/dpsi < 0)
        if not idx:
            return np.nan
        i = min(idx, key=lambda j: abs(grid[j] - psi))
        psi = brentq(g, grid[i], grid[i + 1])
    return psi


# ------------------------------------------------------------ spectral model
def spectral(L, model, zeta_s, U, TI, z, control=False, kp=cm.k_p_nom, kd=cm.k_d_nom,
             vessel=None, rope=True, f=None):
    """Linearised yaw response spectrum and its decomposition."""
    P = params_vector(L, model, zeta_s, U, control, kp, kd, rope)
    k, I, c_s, c_a = P[0], P[1], P[2], P[3]
    psib = static_equilibrium(P, U)
    qasl = P[8] * U ** 2
    kind = int(P[10])
    dc = model.dc(psib)
    cval = model.c(psib)
    Ka = qasl * dc
    keff = k * np.cos(psib) - Ka + (kp if control else 0.0)
    ctot = c_s + c_a + (kd if control else 0.0)
    if f is None:
        f = np.logspace(-4, 0.3, 6000)
    w = 2 * np.pi * f
    H = 1.0 / (keff - I * w ** 2 + 1j * ctot * w)
    lam = cm.Lambda1(z)
    Su = kaimal(f, TI * U, 8.1 * lam, U)
    Sv = kaimal(f, cm.r_sigma_v * TI * U, 2.7 * lam, U)
    # forcing: lateral gust rotates the flow by v/U  -> -Ka v/U ; along-wind gust
    # changes q -> 2 q A_s L c(psib) u/U
    SF = (Ka / U) ** 2 * Sv + (2 * qasl * cval / U) ** 2 * Su
    out = dict(psib=psib, keff=keff, Ka=Ka, ctot=ctot, f=f)
    Spsi = np.abs(H) ** 2 * SF
    if vessel is not None:
        fv, Spv = vessel   # vessel yaw spectrum (one-sided, rad^2/Hz)
        Spsi = Spsi + np.abs(H) ** 2 * np.interp(f, fv, Spv, left=0, right=0) * (k ** 2 + (c_s * w) ** 2)
    out["Spsi"] = Spsi
    out["sigma"] = float(np.sqrt(np.trapezoid(Spsi, f)))
    out["sigma_qs"] = float(np.sqrt(np.trapezoid(SF, f)) / abs(keff))
    out["zeta_tot"] = ctot / (2 * np.sqrt(abs(keff) * I))
    out["f_n"] = np.sqrt(max(keff, 0) / I) / (2 * np.pi)
    return out


def simulate(L, model, zeta_s, U, TI, z, seeds, T=4200.0, burn=600.0, dt=0.05,
             vessel=False, heading=150.0, control=False, kp=cm.k_p_nom, kd=cm.k_d_nom,
             rope=True, Hs=cm.Hs):
    P = params_vector(L, model, zeta_s, U, control, kp, kd, rope)
    psib = static_equilibrium(P, U)
    stats = []
    n = int(round(T / dt))
    for sd in seeds:
        u, v = turbulence(sd, T, dt, U, TI, z)
        if vessel:
            xt, yt, azt, psv, dpsv = vessel_inputs(sd, T, dt, heading, Hs)
        else:
            xt = yt = azt = psv = dpsv = np.zeros(n)
        s0 = np.array([psib if np.isfinite(psib) else 0.0, 0, 0, 0, 0, 0])
        X = integrate(s0, dt, U, u, v, xt, yt, azt, psv, dpsv, P)
        i0 = int(burn / dt)
        psi = X[i0:, 0]
        stats.append(dict(mean=float(np.degrees(psi.mean())), std=float(np.degrees(psi.std())),
                          max_abs=float(np.degrees(np.abs(psi).max())),
                          frac_in_tol=float(np.mean(np.abs(psi) < np.radians(1.0))),
                          frac_in_tol_dyn=float(np.mean(np.abs(psi - psi.mean()) < np.radians(1.0))),
                          sway_std=float(np.hypot(X[i0:, 2].std(), X[i0:, 4].std()))))
    return stats


def summarize(stats, key):
    x = np.array([s[key] for s in stats])
    n = len(x)
    from scipy.stats import t as tdist
    hw = tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(x.mean()), float(hw)


def floquet_max_multiplier(keff, k, I, c, eps, we, nsteps=400):
    """Largest |Floquet multiplier| of I y'' + c y' + (keff + eps k cos(we t)) y = 0."""
    T = 2 * np.pi / we
    dt = T / nsteps
    Phi = np.eye(2)
    for i in range(nsteps):
        t = i * dt
        def A(tt):
            return np.array([[0.0, 1.0], [-(keff + eps * k * np.cos(we * tt)) / I, -c / I]])
        # RK4 on the fundamental matrix
        K1 = A(t) @ Phi
        K2 = A(t + dt / 2) @ (Phi + dt / 2 * K1)
        K3 = A(t + dt / 2) @ (Phi + dt / 2 * K2)
        K4 = A(t + dt) @ (Phi + dt * K3)
        Phi = Phi + dt / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return float(np.max(np.abs(np.linalg.eigvals(Phi))))


def run(quick=False):
    t0 = time.time()
    res = {}
    with open(f"{cm.RES}/s2_aero_bem.json") as fh:
        CKd = json.load(fh)["CK"]
    models = {"munk": Munk(CKd["bem_box"]), "box": BluffBox(0.09, 0.04)}
    seeds = list(range(8 if quick else 20))
    configs = {"mating": cm.L_mating, "long": cm.L_ref}

    # ---------------- (A) verification: time step, spectral vs time domain (linear regime)
    ver = {}
    L = cm.L_ref
    z = z_payload(L)
    m = models["munk"]
    Ud = U_div(cm.suspension(L).k_psi, m.CK)
    U = 0.5 * Ud
    for dt in (0.2, 0.1, 0.05, 0.025):
        st = simulate(L, m, 0.01, U, 0.06, z, seeds[:4], dt=dt, rope=False)
        ver[f"dt={dt}"] = summarize(st, "std")
    sp = spectral(L, m, 0.01, U, 0.06, z, rope=False)
    ver["spectral_sigma_deg"] = float(np.degrees(sp["sigma"]))
    st = simulate(L, m, 0.01, U, 0.06, z, seeds, rope=False)
    ver["td_sigma_deg"] = summarize(st, "std")
    res["verification"] = ver
    print("verification", ver, time.time() - t0)

    # ---------------- (B) response vs U/U_div (spectral curves + time-domain points)
    ugrid = np.linspace(0.05, 0.95, 37)
    upts = [0.3, 0.5, 0.7, 0.85]
    resp = {}
    for cname, L in configs.items():
        z = z_payload(L)
        kpsi = cm.suspension(L).k_psi
        for mname, m in models.items():
            Ud = U_div(kpsi, m.CK)
            for TI in cm.TI_values:
                for zeta in cm.zeta_values:
                    key = f"{cname}|{mname}|TI={TI}|zeta={zeta}"
                    curve = []
                    for uu in ugrid:
                        spx = spectral(L, m, zeta, uu * Ud, TI, z)
                        if not np.isfinite(spx["psib"]):
                            break
                        curve.append((float(uu), float(np.degrees(spx["sigma"])),
                                      float(np.degrees(spx["sigma_qs"])),
                                      float(np.degrees(spx["psib"])), float(spx["zeta_tot"])))
                    pts = []
                    if TI == 0.06 or zeta == 0.01:
                        for uu in upts:
                            stt = simulate(L, m, zeta, uu * Ud, TI, z, seeds[:10])
                            pts.append(dict(u=uu, U=uu * Ud, std=summarize(stt, "std"),
                                            mean=summarize(stt, "mean"),
                                            frac=summarize(stt, "frac_in_tol_dyn")))
                    resp[key] = dict(U_div=Ud, curve=curve, points=pts)
            print(cname, mname, "done", time.time() - t0)
    res["response"] = resp

    # ---------------- (C) damping sensitivity exponent (spectral), physical wind speeds
    sens = {}
    for cname, L in configs.items():
        z = z_payload(L)
        for mname, m in models.items():
            for U in (8.0, 12.0):
                if U >= U_div(cm.suspension(L).k_psi, m.CK):
                    continue
                zs = np.logspace(np.log10(0.002), np.log10(0.05), 9)
                sig = [spectral(L, m, zz, U, 0.06, z)["sigma"] for zz in zs]
                p = np.polyfit(np.log(zs), np.log(sig), 1)[0]
                spx = spectral(L, m, 0.01, U, 0.06, z)
                sens[f"{cname}|{mname}|U={U}"] = dict(
                    exponent=float(p), sigma_deg=[float(np.degrees(s_)) for s_ in sig],
                    zetas=zs.tolist(), qs_fraction=float((spx["sigma_qs"] / spx["sigma"]) ** 2),
                    f_n=spx["f_n"], zeta_aero=float(params_vector(L, m, 0, U)[3] /
                                                    (2 * np.sqrt(cm.suspension(L).k_psi * cm.I_zz))))
    res["damping_sensitivity"] = sens

    # ---------------- (D) vessel coupling at mating: with / without vessel motion
    L = cm.L_mating
    z = z_payload(L)
    vc = {}
    for mname, m in models.items():
        for U in (8.0, 12.0):
            for hd in (120.0, 150.0, 180.0):
                st0 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=False)
                st1 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=True, heading=hd)
                st2 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=True, heading=hd, Hs=3.0)
                vc[f"{mname}|U={U}|hd={hd}"] = dict(
                    no_vessel=summarize(st0, "std"), vessel=summarize(st1, "std"),
                    vessel_Hs3=summarize(st2, "std"),
                    sway_no=summarize(st0, "sway_std"), sway_v=summarize(st1, "sway_std"))
        print("vessel", mname, time.time() - t0)
    res["vessel_coupling"] = vc

    # transfer functions: vessel yaw -> payload yaw, wind direction -> payload yaw
    f = np.logspace(-3, 0, 400)
    tf = {"f": f.tolist()}
    for cname, L in configs.items():
        s = cm.suspension(L)
        for mname, m in models.items():
            for U in (0.0, 8.0, 12.0):
                Ka = 0.5 * cm.rho_air * U ** 2 * cm.A_side * cm.L_n * m.CK
                P = params_vector(L, m, 0.01, U)
                ctot = P[2] + P[3]
                w = 2 * np.pi * f
                den = s.k_psi - Ka - cm.I_zz * w ** 2 + 1j * ctot * w
                tf[f"{cname}|{mname}|U={U}|vessel"] = np.abs((s.k_psi + 1j * P[2] * w) / den).tolist()
                tf[f"{cname}|{mname}|U={U}|wind"] = np.abs(Ka / den).tolist()
    res["transfer"] = tf

    # ---------------- (E) Floquet analysis: heave-induced modulation of k_psi
    with open(f"{cm.RES}/s4_vessel.json") as fh:
        V = json.load(fh)
    az = max(V["rms"][h]["tip_az"] for h in V["rms"])
    eps_real = 3 * az / cm.g            # 3-sigma amplitude of the modulation
    we = 2 * np.pi / cm.Tp
    fl = {"eps_real_3sigma": eps_real}
    rows = []
    for L in (cm.L_mating, cm.L_ref):
        s = cm.suspension(L)
        c = 2 * 0.002 * np.sqrt(s.k_psi * cm.I_zz)
        for u in (0.0, 0.9, 0.99, 0.999):
            keff = s.k_psi * (1 - u ** 2)
            for eps in (eps_real, 0.1, 0.5):
                mu = floquet_max_multiplier(keff, s.k_psi, cm.I_zz, c, eps, we)
                rows.append(dict(L=L, u=u, eps=eps, max_multiplier=mu))
        # Kapitza averaging estimate of the shift of the threshold
        fl[f"dU_over_Udiv_L{int(L)}"] = float((eps_real * s.k_psi) ** 2 /
                                              (4 * cm.I_zz * we ** 2 * s.k_psi))
    fl["rows"] = rows
    res["floquet"] = fl
    res["wall_s"] = time.time() - t0
    with open(f"{cm.RES}/s5_dynamics.json", "w") as fh:
        json.dump(res, fh)
    return res


if __name__ == "__main__":
    import sys
    r = run(quick="quick" in sys.argv)
    print(json.dumps(r["verification"], indent=0))
    print(json.dumps(r["damping_sensitivity"], indent=0)[:3000])
    print(json.dumps(r["vessel_coupling"], indent=0)[:3000])
    print(json.dumps({k: v for k, v in r["floquet"].items() if k != "rows"}, indent=0))
    for row in r["floquet"]["rows"]:
        print(row)
    print("wall", r["wall_s"])
