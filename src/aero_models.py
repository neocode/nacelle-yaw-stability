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


class BluffBox:
    """Sharp-edged rectangular box, Greenwell (ICAS 2008; Aeronaut J 2011) attached-flow fit (Eq. (12) of the paper) to
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


def U_div(k_psi, CK):
    """Static divergence speed [m/s]; inf if CK <= 0 (no divergence)."""
    if CK <= 0:
        return np.inf
    return float(np.sqrt(2 * k_psi / (cm.rho_air * cm.A_side * cm.L_n * CK)))


def qASL(U):
    return 0.5 * cm.rho_air * U ** 2 * cm.A_side * cm.L_n
