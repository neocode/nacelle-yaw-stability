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
from aero_models import Munk, MeasuredBox, U_div, ref_slope, zero_near_max_slope
from s3_statics import Z_BLOCK_CG, z_payload

CDA = 1.05 * cm.W_n * cm.H_n        # m^2  end-on drag area of the payload
CYB = 1.35                          # side-force slope (Greenwell 2008), for strip damping


def kaimal(f, sigma, Lk, U):
    """IEC 61400-1 Kaimal one-sided spectrum [ (m/s)^2 / Hz ]."""
    x = f * Lk / U
    return sigma ** 2 * 4 * Lk / U / (1 + 6 * x) ** (5 / 3)


def froya(f, sigma, U, z):
    """Froya wind spectrum (DNV-RP-C205) for the mean wind U at height z, used as a
    spectral shape and scaled to the variance sigma^2 [ (m/s)^2 / Hz ]."""
    U0 = U * (10.0 / z) ** cm.alpha_shear          # mean wind at 10 m (power law)
    n = 0.468

    def raw(ff):
        ft = 172.0 * ff * (z / 10.0) ** (2 / 3) * (U0 / 10.0) ** -0.75
        return 320.0 * (U0 / 10.0) ** 2 * (z / 10.0) ** 0.45 / (1.0 + ft ** n) ** (5.0 / (3.0 * n))
    fi = np.logspace(-7, 2, 20000)
    return sigma ** 2 * raw(f) / np.trapezoid(raw(fi), fi)


def turbulence(seed, T, dt, U, TI, z, T_syn=None):
    """Point time series u(t), v(t) by spectral synthesis (random phases, FFT).
    The series is synthesised over T_syn >= T (frequency spacing 1/T_syn) and its
    first T seconds are returned; a fine spacing is needed when the yaw resonance
    is narrower than 1/T."""
    rng = np.random.default_rng(seed)
    n_out = int(round(T / dt))
    n = max(n_out, int(round((T_syn or T) / dt)))
    n += n % 2
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
        out.append(np.fft.irfft(X, n)[:n_out])
    return out[0], out[1]


_V = {}


_C = {}


def vessel_inputs(seed, T, dt, heading_deg=150.0, Hs=cm.Hs, Tp=cm.Tp, coupled_L=None, zeta_p=None,
                  variant=None):
    """Boom-tip horizontal motion and velocity, vertical acceleration and vessel
    yaw [rad] from the BEM RAOs and a JONSWAP sea (random phases).
    coupled_L: fall length of the two-way coupled vessel-pendulum solution, evaluated
    with the pendulum damping zeta_p directly on the synthesis frequencies
    (None: one-way RAOs of the vessel alone)."""
    if not _V:
        with open(f"{cm.RES}/s4_vessel.json") as fh:
            _V.update(json.load(fh))
    V = _V
    w_r = np.array(V["omega"])
    j = V["headings_deg"].index(min(V["headings_deg"], key=lambda h: abs(h - heading_deg)))
    rng = np.random.default_rng(10_000 + seed)
    n = int(round(T / dt))
    dw = 2 * np.pi / T
    w = np.fft.rfftfreq(n, dt) * 2 * np.pi            # multiples of dw
    act = (w >= w_r[0]) & (w <= w_r[-1])
    wa = np.where(act, w, 1.0)
    if coupled_L is None:
        # BEM RAOs use the time factor exp(-i w t); the synthesis below uses exp(+i w t)
        rao_r = np.conj(np.array(V["rao_re"])[:, j, :] + 1j * np.array(V["rao_im"])[:, j, :])
        tip_r = np.conj(np.array(V["tip_re"])[:, j, :] + 1j * np.array(V["tip_im"])[:, j, :])
        ip = lambda Z: np.interp(wa, w_r, Z.real) + 1j * np.interp(wa, w_r, Z.imag)
        rao = np.stack([ip(rao_r[:, k]) for k in range(6)], axis=1)
        tip = np.stack([ip(tip_r[:, k]) for k in range(3)], axis=1)
    else:
        from s4_vessel import coupled_on_grid
        key = (round(coupled_L, 3), zeta_p, n, dt, j, json.dumps(variant, sort_keys=True))
        if key not in _C:
            R, Tq = coupled_on_grid(w, coupled_L, zeta_p, variant)
            _C.clear()
            _C[key] = (np.conj(R[:, j, :]), np.conj(Tq[:, j, :]))
        rao, tip = _C[key]
    wp = 2 * np.pi / Tp
    sig = np.where(wa <= wp, 0.07, 0.09)
    S = ((1 - 0.287 * np.log(cm.gamma_J)) * (5 / 16) * Hs ** 2 * wp ** 4 / wa ** 5
         * np.exp(-1.25 * (wp / wa) ** 4) * cm.gamma_J ** np.exp(-0.5 * ((wa - wp) / (sig * wp)) ** 2))
    S = np.where(act, S, 0.0)
    a = np.sqrt(2 * S * dw)
    ph = rng.uniform(0, 2 * np.pi, len(w))

    def synth(Z):
        # x(t) = Re sum a_k Z_k exp(i(w_k t + ph_k))
        return np.fft.irfft(a * Z * np.exp(1j * ph) * n / 2, n)

    xt = synth(tip[:, 0])
    yt = synth(tip[:, 1])
    dxt = synth(1j * w * tip[:, 0])
    dyt = synth(1j * w * tip[:, 1])
    azt = synth(-w ** 2 * tip[:, 2])
    psv = synth(rao[:, 5])
    dpsv = synth(1j * w * rao[:, 5])
    return xt, yt, azt, psv, dpsv, dxt, dyt


# --------------------------------------------------------------- integrator
NF = 8    # number of sine coefficients of the moment curve


@njit(cache=True)
def _aero_c(beta, P):
    """c(beta) = sum_k a_k sin(2 k beta) - cyx sin(beta); a_k = P[23:23+NF], cyx = P[13]."""
    c = -P[13] * np.sin(beta)
    for k in range(NF):
        c += P[23 + k] * np.sin(2 * (k + 1) * beta)
    return c


TUG = 23 + NF  # first index of the two-line tugger parameters
NP = TUG + 6   # length of the parameter vector


@njit(cache=True)
def _rhs(s, U, q, P):
    # q (inputs): [u, v, xt, yt, azt, psv, dpsv, dth, dxt, dyt]
    # P: [k, I, c_s, c_a, Mr, m, W, lp, rhoAsL_half, CDA_half_rho, kind, -, -, cyx, K_ref,
    #     kp, kd, Mt_max, psi_ref, ctrl, c_pend, side_half_rho, theta0, a_1..a_NF]
    psi, dpsi, x, dx, y, dy = s[0], s[1], s[2], s[3], s[4], s[5]
    u, v, xt, yt, azt, psv, dpsv, dth, dxt, dyt = q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]
    k = P[0]; I = P[1]
    ux = U + u - dx
    uy = v - dy
    sp2 = ux * ux + uy * uy
    th = np.arctan2(uy, ux)
    beta = psi - th - P[22]
    Ma = P[8] * sp2 * _aero_c(beta, P)
    fac = 1.0 + azt / 9.80665
    Mt = 0.0
    Fx = 0.0
    Fy = 0.0
    if P[19] > 1.5:
        # two tugger lines attached at +-r_t on the payload axis, pulling towards anchors
        # fixed to the boom at (x_t +- r_t, y_t - l_t); the yaw couple is set by the tension
        # difference and a transverse sway damping by the tension sum; one-sided tension
        Mc = -P[15] * (psi - P[18]) - P[16] * dpsi
        T0, rt, lt, Tmin, Tmax, csw = P[TUG], P[TUG + 1], P[TUG + 2], P[TUG + 3], P[TUG + 4], P[TUG + 5]
        Tc = T0 + 0.5 * csw * (dy - dyt)
        cp_, sp_ = np.cos(psi), np.sin(psi)
        for sg in (1.0, -1.0):
            T = Tc + sg * (-Mc) / (2.0 * rt)
            if T < Tmin:
                T = Tmin
            elif T > Tmax:
                T = Tmax
            ax = sg * rt * cp_
            ay = sg * rt * sp_
            lx = xt + sg * rt - (x + ax)
            ly = yt - lt - (y + ay)
            ln = np.sqrt(lx * lx + ly * ly)
            fx = T * lx / ln
            fy = T * ly / ln
            Mt += ax * fy - ay * fx
            Fx += fx
            Fy += fy
    elif P[19] > 0.5:
        Mt = -P[15] * (psi - P[18]) - P[16] * dpsi
        if Mt > P[17]:
            Mt = P[17]
        elif Mt < -P[17]:
            Mt = -P[17]
    # aerodynamic damping acts on the rotation rate relative to the rotating wind
    # direction (strip theory with frozen turbulence: includes the gust-gradient moment)
    ddpsi = (-k * fac * np.sin(psi - psv) - P[2] * (dpsi - dpsv) + P[3] * (dth - dpsi)
             + Ma + P[4] * fac + Mt) / I
    sp = np.sqrt(sp2)
    kx = P[6] * fac / P[7]
    ddx = (-kx * (x - xt) - P[20] * (dx - dxt) + P[9] * sp * ux + Fx) / P[5]
    ddy = (-kx * (y - yt) - P[20] * (dy - dyt) + P[21] * sp * uy + Fy) / P[5]
    out = np.empty(6)
    out[0] = dpsi; out[1] = ddpsi; out[2] = dx; out[3] = ddx; out[4] = dy; out[5] = ddy
    return out


@njit(cache=True)
def _integrate(s0, dt, U, Q, P):
    """RK4 with inputs sampled on the integration grid (midpoints averaged)."""
    n = Q.shape[0]
    out = np.empty((n, 6))
    s = s0.copy()
    out[0] = s
    for i in range(n - 1):
        qm = 0.5 * (Q[i] + Q[i + 1])
        k1 = _rhs(s, U, Q[i], P)
        k2 = _rhs(s + 0.5 * dt * k1, U, qm, P)
        k3 = _rhs(s + 0.5 * dt * k2, U, qm, P)
        k4 = _rhs(s + dt * k3, U, Q[i + 1], P)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


def integrate(s0, dt, U, u, v, xt, yt, azt, psv, dpsv, P, dth=None, dxt=None, dyt=None):
    """Wrapper: missing inputs are zero; short parameter vectors are padded."""
    n = len(u)
    z = np.zeros(n)
    Q = np.column_stack([u, v, xt, yt, azt, psv, dpsv,
                         z if dth is None else dth, z if dxt is None else dxt,
                         z if dyt is None else dyt]).astype(float)
    P = np.asarray(P, float)
    if len(P) < NP:
        P = np.r_[P, np.zeros(NP - len(P))]
    return _integrate(np.asarray(s0, float), dt, U, Q, P)


ZETA_P = 0.02   # damping ratio of the pendulum mode (hoist friction, rigging; ASSUMED)


TUGGER = dict(T0=50e3, r_t=10.0, l_t=30.0, T_min=5e3, T_max=100e3, c_sway=0.0)


def params_vector(L, model, zeta_s, U, control=False, kp=cm.k_p_nom, kd=cm.k_d_nom,
                  rope=True, aero_damping=True, aero_damping_munk=False, zeta_p=ZETA_P,
                  tugger=None):
    """control: False, True (ideal saturated couple) or 'lines' (two tugger lines with
    the geometry and limits of `tugger`, default TUGGER)."""
    s = cm.suspension(L)
    c_s = 2 * zeta_s * np.sqrt(s.k_psi * cm.I_zz)
    kind = 0 if isinstance(model, Munk) else 1
    c_a = 0.0
    if (kind == 1 and aero_damping) or aero_damping_munk:
        # quasi-steady strip estimate of the yaw damping of a box with linear side force
        c_a = cm.rho_air * U * cm.H_n * cm.L_n ** 3 * CYB / 24.0
    a, cyx = model.fourier()
    af = np.zeros(NF)
    af[:len(a)] = a
    m = cm.m_p + cm.m_hb
    lp = L + Z_BLOCK_CG
    c_p = 2 * zeta_p * m * np.sqrt(cm.g / lp)
    tg = dict(TUGGER, **(tugger or {}))
    mode = 2.0 if control == "lines" else (1.0 if control else 0.0)
    return np.r_[s.k_psi, cm.I_zz, c_s, c_a, s.M_r if rope else 0.0, m, s.W_load,
                 lp, 0.5 * cm.rho_air * cm.A_side * cm.L_n,
                 0.5 * cm.rho_air * CDA, kind, 0.0, 0.0, cyx, ref_slope(model),
                 kp, kd, cm.M_t_max, 0.0, mode,
                 c_p, 0.5 * cm.rho_air * cm.A_side * CYB, getattr(model, "theta0", 0.0), af,
                 tg["T0"], tg["r_t"], tg["l_t"], tg["T_min"], tg["T_max"], tg["c_sway"]]


def static_equilibrium(P, U):
    """Mean yaw from the full nonlinear static balance (continuation from 0)."""
    from scipy.optimize import brentq
    k, Mr = P[0], P[4]
    # primary branch: the stable root closest to the still-air twist, found by
    # continuation in U (a fold of the primary branch returns NaN)
    psi = np.arcsin(min(Mr / k, 1.0))
    for Ui in np.linspace(0.0, U, 41)[1:]:
        g = lambda p: (-k * np.sin(p) + P[8] * Ui ** 2 * float(_aero_c(p - P[22], P)) + Mr
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
             vessel=None, rope=True, f=None, aero_damping_munk=False, spectrum="kaimal",
             ca_factor=1.0):
    """Linearised yaw response spectrum and its decomposition.
    spectrum: 'kaimal' (IEC) or 'froya' (DNV-RP-C205 shape, same variances);
    ca_factor scales the aerodynamic yaw damping (sensitivity)."""
    P = params_vector(L, model, zeta_s, U, bool(control), kp, kd, rope, aero_damping_munk=aero_damping_munk)
    k, I, c_s, c_a = P[0], P[1], P[2], P[3] * ca_factor
    psib = static_equilibrium(P, U)
    qasl = P[8] * U ** 2
    th0 = getattr(model, "theta0", 0.0)
    dc = float(model.dc(psib - th0))
    cval = float(model.c(psib - th0))
    Ka = qasl * dc
    keff = k * np.cos(psib) - Ka + (kp if control else 0.0)
    ctot = c_s + c_a + (kd if control else 0.0)
    if f is None:
        f = np.logspace(-4, 0.3, 6000)
    w = 2 * np.pi * f
    H = 1.0 / (keff - I * w ** 2 + 1j * ctot * w)
    lam = cm.Lambda1(z)
    if spectrum == "froya":
        Su = froya(f, TI * U, U, z)
        Sv = froya(f, cm.r_sigma_v * TI * U, U, z)
    else:
        Su = kaimal(f, TI * U, 8.1 * lam, U)
        Sv = kaimal(f, cm.r_sigma_v * TI * U, 2.7 * lam, U)
    # forcing: lateral gust rotates the flow by v/U -> (-Ka + i w c_a) v/U (the second
    # term is the gust-gradient moment of strip theory); along-wind gust changes q
    SF = ((Ka ** 2 + (w * c_a) ** 2) / U ** 2) * Sv + (2 * qasl * cval / U) ** 2 * Su
    out = dict(psib=psib, keff=keff, Ka=Ka, ctot=ctot, f=f)
    Spsi = np.abs(H) ** 2 * SF
    if vessel is not None:
        fv, Spv = vessel   # vessel yaw spectrum (one-sided, rad^2/Hz)
        Spsi = Spsi + np.abs(H) ** 2 * np.interp(f, fv, Spv, left=0, right=0) * (k ** 2 + (c_s * w) ** 2)
    out["Spsi"] = Spsi
    out["sigma"] = float(np.sqrt(np.trapezoid(Spsi, f)))
    # quasi-static part: response to the direction and speed fluctuations with inertia and
    # damping removed (the gust-gradient term, a rate input, is excluded)
    SF_qs = (Ka / U) ** 2 * Sv + (2 * qasl * cval / U) ** 2 * Su
    out["sigma_qs"] = float(np.sqrt(np.trapezoid(SF_qs, f)) / abs(keff))
    out["zeta_tot"] = ctot / (2 * np.sqrt(abs(keff) * I))
    out["f_n"] = np.sqrt(max(keff, 0) / I) / (2 * np.pi)
    # expected largest |psi| in a 20 min window (narrow-band Gaussian, Davenport peak factor)
    nu = np.sqrt(np.trapezoid(f ** 2 * Spsi, f) / max(np.trapezoid(Spsi, f), 1e-30))
    nT = max(nu * 1200.0, 1.0001)
    gpk = np.sqrt(2 * np.log(nT)) + 0.5772 / np.sqrt(2 * np.log(nT))
    out["peak_factor"] = float(gpk)
    out["expected_max"] = float(abs(psib) + gpk * out["sigma"])
    return out


def simulate(L, model, zeta_s, U, TI, z, seeds, record=None, burn=600.0, dt=0.05,
             vessel=False, heading=150.0, control=False, kp=cm.k_p_nom, kd=cm.k_d_nom,
             rope=True, Hs=cm.Hs, Tp=cm.Tp, zeta_p=ZETA_P, only=None, two_way=True,
             windows=None, tugger=None, vessel_variant=None):
    """Time-domain runs (one record of length `record` per seed after the burn-in).
    only: restrict the vessel input to 'yaw', 'horiz' or 'heave'.
    two_way: boom-tip motion from the coupled vessel-pendulum solution.
    The burn-in is extended to 3/(zeta_tot w_0), so that the resonant variance of the
    yaw mode, which builds up as 1 - exp(-2 zeta w_0 t), is within 0.3 % of its
    stationary value.  The record lasts at least one hour and at least ten correlation
    times 1/(zeta_tot w_0) of the response envelope (unless `record` is given).
    windows: length [s] of the windows whose largest |psi| is returned (win_max, and
    win_max_dyn about the record mean).  control='lines': two physical tugger lines."""
    P = params_vector(L, model, zeta_s, U, control, kp, kd, rope, zeta_p=zeta_p, tugger=tugger)
    psib = static_equilibrium(P, U)
    spx = spectral(L, model, zeta_s, U, TI, z, bool(control), kp, kd, rope=rope)
    tau = 1.0 / (spx["zeta_tot"] * 2 * np.pi * max(spx["f_n"], 1e-4))   # envelope correlation time
    burn = max(burn, min(3.0 * tau, 6.0e4))
    if record is None:
        record = float(np.ceil(max(3600.0, min(10.0 * tau, 6.0e4)) / 100.0) * 100.0)
    burn = float(np.ceil(burn / 100.0) * 100.0)
    T = burn + record
    # wind frequency spacing at least ten lines per half-power bandwidth of the yaw mode
    T_syn = max(T, min(10.0 / (2 * spx["zeta_tot"] * max(spx["f_n"], 1e-4)), 4.0e5))
    stats = []
    n = int(round(T / dt))
    Z = np.zeros(n)
    for sd in seeds:
        u, v = turbulence(sd, T, dt, U, TI, z, T_syn)
        dth = np.gradient(np.arctan2(v, U + u), dt)
        if vessel:
            xt, yt, azt, psv, dpsv, dxt, dyt = vessel_inputs(sd, T, dt, heading, Hs, Tp,
                                                             L if two_way else None, zeta_p,
                                                             vessel_variant)
            if only == "yaw":
                xt = yt = azt = dxt = dyt = Z
            elif only == "horiz":
                azt = psv = dpsv = Z
            elif only == "heave":
                xt = yt = psv = dpsv = dxt = dyt = Z
        else:
            xt = yt = azt = psv = dpsv = dxt = dyt = Z
        s0 = np.array([psib if np.isfinite(psib) else 0.0, 0, 0, 0, 0, 0])
        X = integrate(s0, dt, U, u, v, xt, yt, azt, psv, dpsv, P, dth, dxt, dyt)
        i0 = int(burn / dt)
        psi = X[i0:, 0]
        stats.append(dict(mean=float(np.degrees(psi.mean())), std=float(np.degrees(psi.std())),
                          max_abs=float(np.degrees(np.abs(psi).max())),
                          frac_in_tol=float(np.mean(np.abs(psi) < np.radians(1.0))),
                          frac_in_tol_dyn=float(np.mean(np.abs(psi - psi.mean()) < np.radians(1.0))),
                          sway_std=float(np.hypot(X[i0:, 2].std(), X[i0:, 4].std())),
                          sway_rel_std=float(np.hypot((X[i0:, 2] - xt[i0:]).std(),
                                                      (X[i0:, 4] - yt[i0:]).std())),
                          sway_angle_std=float(np.degrees(np.hypot((X[i0:, 2] - xt[i0:]).std(),
                                                                   (X[i0:, 4] - yt[i0:]).std())
                                                          / (L + Z_BLOCK_CG))),
                          burn=burn, record=record))
        if windows:
            nw = int(len(psi) * dt // windows)
            m_w = int(round(windows / dt))
            pa = np.abs(psi[:nw * m_w]).reshape(nw, m_w).max(1)
            pd = np.abs(psi[:nw * m_w] - psi.mean()).reshape(nw, m_w).max(1)
            stats[-1]["win_max"] = np.degrees(pa).tolist()
            stats[-1]["win_max_dyn"] = np.degrees(pd).tolist()
        if P[19] > 1.5:
            stats[-1].update(line_diagnostics(X[i0:], xt[i0:], yt[i0:], dyt[i0:], P, dt))
    return stats


def line_diagnostics(X, xt, yt, dyt, P, dt):
    """Tensions, tension rates and reeling speeds of the two tugger lines (control='lines')."""
    psi, dpsi, x, y, dy = X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5]
    Mc = -P[15] * (psi - P[18]) - P[16] * dpsi
    T0, rt, lt, Tmin, Tmax, csw = P[TUG:TUG + 6]
    Tc = T0 + 0.5 * csw * (dy - dyt)
    out = {}
    Ts, Ls = [], []
    for sg in (1.0, -1.0):
        T = np.clip(Tc - sg * Mc / (2 * rt), Tmin, Tmax)
        lx = xt + sg * rt - (x + sg * rt * np.cos(psi))
        ly = yt - lt - (y + sg * rt * np.sin(psi))
        Ts.append(T); Ls.append(np.hypot(lx, ly))
    Ts, Ls = np.array(Ts), np.array(Ls)
    dT = np.gradient(Ts, dt, axis=1)
    dl = np.gradient(Ls, dt, axis=1)
    out["T_min_kN"] = float(Ts.min() / 1e3)
    out["T_max_kN"] = float(Ts.max() / 1e3)
    out["dT_max_kN_s"] = float(np.abs(dT).max() / 1e3)
    out["reel_std"] = float(dl.std(axis=1).max())
    out["reel_max"] = float(np.abs(dl).max())
    out["frac_saturated"] = float(np.mean((Ts <= Tmin + 1.0) | (Ts >= Tmax - 1.0)))
    out["sway_rel_y_std"] = float((y - yt).std())
    out["sway_rel_y_mean"] = float((y - yt).mean())
    return out


def paired(base, other, key="std"):
    """Mean and 95 % half-width of the paired difference (other - base) over common
    seeds, in per cent of the mean of base."""
    a = np.array([x[key] for x in base]); b = np.array([x[key] for x in other])
    d = 100.0 * (b - a) / a.mean()
    from scipy.stats import t as tdist
    hw = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(hw)


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


def box_model(**kw):
    """Measured sharp-edged box with the mean wind at the zero of c(beta) next to the
    steepest destabilising slope (about 20 deg off the payload axis): the wind then
    exerts no mean moment on the aligned payload.  The side is chosen so that the
    rope-torque twist moves the incidence towards the steepest slope."""
    m = MeasuredBox(**kw)
    m.theta0 = +zero_near_max_slope(m)
    return m


def U_ref(L, model):
    """Critical speed of the model at its mean wind direction (U_div for Munk)."""
    return U_div(cm.suspension(L).k_psi, ref_slope(model))


def run(quick=False):
    t0 = time.time()
    res = {}
    with open(f"{cm.RES}/s2_aero_bem.json") as fh:
        CKd = json.load(fh)["CK"]
    models = {"munk": Munk(CKd["bem_box"]), "box": box_model()}
    res["box_theta0_deg"] = float(np.degrees(models["box"].theta0))
    res["box_ref_slope"] = ref_slope(models["box"])
    seeds = list(range(8 if quick else 20))
    configs = {"mating": cm.L_mating, "long": cm.L_ref, "liftoff": cm.L_liftoff}

    # ---------------- (A) verification: time step, spectral vs time domain (linear regime)
    ver = {}
    L = cm.L_ref
    z = z_payload(L)
    m = models["munk"]
    Ud = U_ref(L, m)
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
    ugrid = np.r_[0.01, 0.02, 0.03, 0.04, np.linspace(0.05, 0.95, 37)]
    upts = [0.3, 0.5, 0.7, 0.85]
    resp = {}
    for cname, L in configs.items():
        z = z_payload(L)
        kpsi = cm.suspension(L).k_psi
        for mname, m in models.items():
            Ud = U_ref(L, m)
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
                    if (cname != "liftoff" and (TI == 0.06 or zeta == 0.01)) or \
                            (cname == "liftoff" and TI == 0.06 and zeta == 0.01):
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
                if U >= U_ref(L, m):
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
    # potential-flow coefficient with the strip-theory aerodynamic damping (sensitivity)
    for U in (8.0, 12.0):
        a0 = spectral(cm.L_mating, models["munk"], 0.01, U, 0.06, z_payload(cm.L_mating))
        a1 = spectral(cm.L_mating, models["munk"], 0.01, U, 0.06, z_payload(cm.L_mating),
                      aero_damping_munk=True)
        sens[f"munk_with_aero_damping|U={U}"] = dict(sigma_without=float(np.degrees(a0["sigma"])),
                                                     sigma_with=float(np.degrees(a1["sigma"])))
    res["damping_sensitivity"] = sens

    # ---------------- (C2) expected maximum yaw in 20 min vs the 1 deg tolerance
    ext = {}
    Ugrid = np.linspace(1.0, 40.0, 157)
    for cname in ("mating", "liftoff"):
        L = configs[cname]; z = z_payload(L)
        for mname, m in models.items():
            for TI in cm.TI_values:
                for ctrl in (False, True):
                    em, sg, dy = [], [], []
                    for U in Ugrid:
                        spx = spectral(L, m, 0.01, U, TI, z, control=ctrl)
                        if not np.isfinite(spx["psib"]) or spx["keff"] <= 0:
                            break
                        em.append(float(np.degrees(spx["expected_max"])))
                        sg.append(float(np.degrees(spx["sigma"])))
                        dy.append(float(np.degrees(spx["peak_factor"] * spx["sigma"])))
                    em = np.array(em); sg = np.array(sg); dy = np.array(dy)

                    def crossing(y):
                        # first up-crossing of 1 deg, linearly interpolated between grid points
                        i = int(np.argmax(y > 1.0))
                        if not np.any(y > 1.0):
                            return None
                        if i == 0:
                            return float(Ugrid[0])
                        return float(np.interp(1.0, [y[i - 1], y[i]], [Ugrid[i - 1], Ugrid[i]]))
                    ext[f"{cname}|{mname}|TI={TI}|ctrl={ctrl}"] = dict(
                        U=Ugrid[:len(em)].tolist(), expected_max=em.tolist(), sigma=sg.tolist(),
                        dynamic_max=dy.tolist(),
                        U_limit_expected_max=crossing(em), U_limit_sigma=crossing(sg),
                        U_limit_dynamic_max=crossing(dy))
    res["extremes"] = ext

    # ---------------- (D) vessel coupling at mating: with / without vessel motion
    L = cm.L_mating
    z = z_payload(L)
    vc = {}
    for mname, m in models.items():
        for U in (8.0, 12.0):
            for hd in (90.0, 120.0, 150.0, 180.0):
                st0 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=False)
                st1 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=True, heading=hd)
                st2 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=True, heading=hd, Hs=3.0)
                row = dict(
                    no_vessel=summarize(st0, "std"), vessel=summarize(st1, "std"),
                    vessel_Hs3=summarize(st2, "std"),
                    sway_no=summarize(st0, "sway_std"), sway_v=summarize(st1, "sway_std"),
                    sway_rel_v=summarize(st1, "sway_rel_std"), sway_rel_Hs3=summarize(st2, "sway_rel_std"),
                    sway_angle_v=summarize(st1, "sway_angle_std"),
                    sway_angle_Hs3=summarize(st2, "sway_angle_std"),
                    diff_vessel_pct=paired(st0, st1), diff_Hs3_pct=paired(st0, st2))
                if hd == 120.0:
                    st4 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=True, heading=hd,
                                   two_way=False)
                    row["vessel_one_way"] = summarize(st4, "std")
                    row["sway_v_one_way"] = summarize(st4, "sway_std")
                    row["sway_rel_v_one_way"] = summarize(st4, "sway_rel_std")
                    row["diff_one_way_vs_two_way_pct"] = paired(st1, st4)
                    for zp in (0.002, 0.05):
                        st3 = simulate(L, m, 0.01, U, 0.06, z, seeds[:10], vessel=True, heading=hd,
                                       zeta_p=zp)
                        row[f"vessel_zeta_p={zp}"] = summarize(st3, "std")
                        row[f"sway_v_zeta_p={zp}"] = summarize(st3, "sway_std")
                        row[f"sway_rel_v_zeta_p={zp}"] = summarize(st3, "sway_rel_std")
                        row[f"diff_zeta_p={zp}_pct"] = paired(st0, st3)
                vc[f"{mname}|U={U}|hd={hd}"] = row
        print("vessel", mname, time.time() - t0)
    res["vessel_coupling"] = vc

    # transfer functions: vessel yaw -> payload yaw, wind direction -> payload yaw
    f = np.logspace(-3, 0, 400)
    tf = {"f": f.tolist()}
    for cname, L in configs.items():
        s = cm.suspension(L)
        for mname, m in models.items():
            for U in (0.0, 8.0, 12.0):
                Ka = 0.5 * cm.rho_air * U ** 2 * cm.A_side * cm.L_n * ref_slope(m)
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
    # principal parametric resonance: excitation at twice the (wind-softened) yaw frequency
    pr = []
    w_r = np.array(V["omega"])
    for L in (cm.L_mating, cm.L_ref):
        s = cm.suspension(L)
        for zeta in (0.002, 0.01):
            c = 2 * zeta * np.sqrt(s.k_psi * cm.I_zz)
            for u in (0.0, 0.5):
                keff = s.k_psi * (1 - u ** 2)
                w0 = np.sqrt(keff / cm.I_zz)
                we2 = 2 * w0
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if floquet_max_multiplier(keff, s.k_psi, cm.I_zz, c, mid, we2, 600) > 1.0:
                        hi = mid
                    else:
                        lo = mid
                pr.append(dict(L=L, zeta=zeta, u=u, T_exc=float(2 * np.pi / we2), eps_crit=hi))
    # boom-tip vertical acceleration in a swell with Tp equal to these excitation periods
    sw = {}
    for Tsw in (13.0, 16.0, 20.0):
        wp = 2 * np.pi / Tsw
        sig = np.where(w_r <= wp, 0.07, 0.09)
        S = ((1 - 0.287 * np.log(cm.gamma_J)) * (5 / 16) * 1.0 ** 2 * wp ** 4 / w_r ** 5
             * np.exp(-1.25 * (wp / w_r) ** 4) * cm.gamma_J ** np.exp(-0.5 * ((w_r - wp) / (sig * wp)) ** 2))
        for j, h in enumerate(V["headings_deg"]):
            tz = np.array(V["tip_abs"])[:, j, 2]
            az = np.sqrt(np.trapezoid((w_r ** 2 * tz) ** 2 * S, w_r))
            sw[f"Tp={Tsw}|hd={round(h)}"] = float(3 * az / cm.g)
    fl["principal_resonance"] = pr
    fl["swell_eps_3sigma_Hs1"] = sw
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
