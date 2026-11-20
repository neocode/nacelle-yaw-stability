"""
Aerodynamic yaw-moment models of the suspended payload (Section 4).

Convention: beta is the incidence of the mean relative wind measured from the
payload's long (x) axis; beta = 0 means the wind arrives along the long axis,
on the end that is 'upwind' at beta = 0.  A POSITIVE slope dM/dbeta at beta = 0
is DESTABILISING (it acts like a negative torsional stiffness).

All models are written as  M_a(beta) = q * A_s * L * c(beta),
q = 0.5 rho U^2, A_s = L*H (side-face area), L = payload length, so that
C_K = c'(0) is the non-dimensional aerodynamic yaw stiffness.
"""
import json
import numpy as np
import common as cm
import wind_tunnel_data as wt

r_WL = cm.W_n / cm.L_n


def _bem_CK():
    try:
        with open(f"{cm.RES}/s2_aero_bem.json") as fh:
            return json.load(fh)["CK"]["bem_box"]
    except FileNotFoundError:
        return 0.496


class Munk:
    """Potential flow: pure couple, M = q A_s L (C_K/2) sin(2 beta)."""
    name = "potential flow (Munk)"

    def __init__(self, CK=None):
        self.CK = _bem_CK() if CK is None else CK

    def c(self, beta):
        return 0.5 * self.CK * np.sin(2 * beta)

    def dc(self, beta):
        return self.CK * np.cos(2 * beta)

    def K3_over_K(self):
        return 2.0 / 3.0

    def fourier(self):
        return np.array([0.5 * self.CK]), 0.0


class BluffBox:
    """Sharp-edged rectangular box, Greenwell (ICAS 2008; Aeronaut J 2011) fit to
    NASA TM-X 62169 and Technion TAE-844 data):
        c(beta) = c_s (1-(W/L)^2) sin 2beta - c_a (W/L) sin 4beta
    plus the moment of the body-axis side force C_Y = C_Yb sin(beta) (Greenwell
    Eq. 6) acting at the side-face centroid, when the suspension axis is
    offset x_s (positive towards the upwind end) from that centroid:
        - C_Yb (x_s / L) sin(beta)
    """
    name = "sharp-edged box (wind-tunnel fit)"

    def __init__(self, cs=0.09, ca=0.04, CYb=1.35, xs=0.0):
        self.cs, self.ca, self.CYb, self.xs = cs, ca, CYb, xs
        self.CK = 2 * cs * (1 - r_WL ** 2) - 4 * ca * r_WL - CYb * xs / cm.L_n

    def c(self, beta):
        return (self.cs * (1 - r_WL ** 2) * np.sin(2 * beta)
                - self.ca * r_WL * np.sin(4 * beta)
                - self.CYb * self.xs / cm.L_n * np.sin(beta))

    def dc(self, beta):
        return (2 * self.cs * (1 - r_WL ** 2) * np.cos(2 * beta)
                - 4 * self.ca * r_WL * np.cos(4 * beta)
                - self.CYb * self.xs / cm.L_n * np.cos(beta))

    def K3_over_K(self):
        # c(beta) = CK beta - K3 beta^3 + ...
        K3 = (self.cs * (1 - r_WL ** 2) * 8 / 6 - self.ca * r_WL * 64 / 6
              - self.CYb * self.xs / cm.L_n / 6)
        return K3 / self.CK

    def fourier(self):
        return (np.array([self.cs * (1 - r_WL ** 2), -self.ca * r_WL]),
                self.CYb * self.xs / cm.L_n)


class MeasuredBox:
    """Sharp-edged box, sine-series fit to the measured moments of the 2.5:1 box
    (container and smooth-sided box, NASA TM-X-62169; wind_tunnel_data.py):
        c(beta) = sum_k a_k sin(2 k beta) - C_Yb (x_s / L) sin(beta)
    The last term is the moment of the body-axis side force when the suspension
    axis lies x_s upwind of the geometric centre (as in BluffBox)."""
    name = "sharp-edged box (measured)"

    def __init__(self, K=wt.K_NOMINAL, xs=0.0, CYb=1.35):
        self.a = wt.fit(K)[0]
        self.k = np.arange(1, len(self.a) + 1)
        self.K, self.xs, self.CYb = K, xs, CYb
        self.CK = float(self.dc(0.0))

    def c(self, beta):
        b = np.asarray(beta, float)
        out = np.zeros_like(b)
        for k, a in zip(self.k, self.a):
            out = out + a * np.sin(2 * k * b)
        return out - self.CYb * self.xs / cm.L_n * np.sin(b)

    def dc(self, beta):
        b = np.asarray(beta, float)
        out = np.zeros_like(b)
        for k, a in zip(self.k, self.a):
            out = out + 2 * k * a * np.cos(2 * k * b)
        return out - self.CYb * self.xs / cm.L_n * np.cos(b)

    def fourier(self):
        return self.a.copy(), self.CYb * self.xs / cm.L_n


def ref_slope(model):
    """Aerodynamic stiffness coefficient c'(-theta0) at psi = 0 for the mean wind
    direction theta0 assigned to the model (attribute theta0, default 0)."""
    return float(model.dc(-getattr(model, "theta0", 0.0)))


def zero_near_max_slope(model):
    """Incidence of the zero of c(beta) nearest to the steepest destabilising slope:
    a wind from this direction exerts no moment on the aligned payload."""
    from scipy.optimize import brentq
    bm = max_slope(model)[1]
    if bm == 0.0:
        return 0.0
    return float(brentq(lambda b: float(model.c(b)), bm - 0.2, bm + 0.2))


def max_slope(model, beta_max=np.pi / 2):
    """Largest destabilising slope c'(beta) over wind directions and the
    incidence at which it occurs [rad]."""
    b = np.linspace(0.0, beta_max, 9001)
    d = model.dc(b)
    i = int(np.argmax(d))
    return float(d[i]), float(b[i])


def U_div(k_psi, CK):
    """Static divergence speed [m/s]; inf if CK <= 0 (no divergence)."""
    if CK <= 0:
        return np.inf
    return float(np.sqrt(2 * k_psi / (cm.rho_air * cm.A_side * cm.L_n * CK)))


def qASL(U):
    return 0.5 * cm.rho_air * U ** 2 * cm.A_side * cm.L_n
