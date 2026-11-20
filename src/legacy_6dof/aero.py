"""
Payload aerodynamics.

YAW MOMENT.  For an elongated body in a uniform stream the leading-order yaw moment
is the MUNK MOMENT -- a potential-flow result, no empirical coefficient:

    M(beta) = 0.5 * rho * U^2 * V * (k2 - k1) * sin(2 beta)

(k2 - k1) is the difference of Lamb's apparent-mass factors for a spheroid of the
same fineness ratio (Lamb 1932 §114; Newman, Marine Hydrodynamics).  It is
DESTABILISING: dM/dbeta|_0 = +rho U^2 V (k2-k1) > 0, i.e. a NEGATIVE torsional
stiffness.  Viscous crossflow-drag terms are O(beta^2) and do not affect the slope
at beta = 0, so the divergence criterion is exact at leading order.

STATIC TORSIONAL DIVERGENCE:

    k_psi - rho U^2 V (k2 - k1) = 0   =>   U_div = sqrt( k_psi / (rho V (k2-k1)) )

Divergence is a STATIC instability: damping does not help, and no observer or filter
can suppress it.

Bracketing V:  the inscribed ellipsoid underestimates the added mass of a sharp-edged
box; the bounding box overestimates it.  We report both -> a factor ~1.4 bracket on
U_div.  Refinement via DNV-RP-C205 shape coefficients / RANS is Phase 3.

DRAG.  Orientation-dependent projected area and drag coefficient for the box.
"""
import numpy as np
from params import ENV, PAYLOAD, derive


def munk_moment(beta, U, V=None):
    """Yaw moment on the payload [N m].  beta = angle between body axis and flow."""
    dv = derive()
    V = dv.V_box if V is None else V
    rho = float(ENV["rho_air"])
    return 0.5 * rho * U ** 2 * V * dv.k2mk1 * np.sin(2.0 * beta)


def aero_yaw_stiffness(U, V=None):
    """dM/dbeta at beta=0 [N m/rad].  POSITIVE = destabilising."""
    dv = derive()
    V = dv.V_box if V is None else V
    return float(ENV["rho_air"]) * U ** 2 * V * dv.k2mk1


def U_divergence(k_psi, V=None):
    """Critical wind speed for static torsional divergence [m/s]."""
    dv = derive()
    V = dv.V_box if V is None else V
    return np.sqrt(k_psi / (float(ENV["rho_air"]) * V * dv.k2mk1))


def U_divergence_bracket(k_psi):
    """(U_div_upper, U_div_lower) using the ellipsoid / box volume bracket."""
    dv = derive()
    return U_divergence(k_psi, dv.V_ell), U_divergence(k_psi, dv.V_box)


def projected_area(beta):
    """Projected area of the box normal to the flow [m^2].  beta from the long axis."""
    Ln, Wn, Hn = (float(PAYLOAD[k]) for k in ("L_n", "W_n", "H_n"))
    return Ln * Hn * abs(np.sin(beta)) + Wn * Hn * abs(np.cos(beta))


def drag_coefficient(beta):
    """Cd of a rectangular box, interpolated between end-on and broadside."""
    Cd_end, Cd_side = 1.05, 1.55        # DNV-RP-C205 order of magnitude; refine in Phase 3
    return Cd_end * abs(np.cos(beta)) + Cd_side * abs(np.sin(beta))


def drag_force(beta, U_rel):
    """Magnitude of the drag force [N].  U_rel = |relative wind| [m/s]."""
    return (0.5 * float(ENV["rho_air"]) * drag_coefficient(beta)
            * projected_area(beta) * U_rel ** 2)


# --------------------------------------------------------------------------
#  CROSSFLOW (STRIP) YAW DAMPING  --  derived from the SAME C_D, no new parameter
# --------------------------------------------------------------------------
#
# A body yawing at rate psi_dot in a flow of speed U at incidence beta sees, on a
# strip at distance s from the yaw axis, a local lateral (cross-body) velocity
#
#       v(s) = U sin(beta) + psi_dot * s .
#
# Crossflow (Morison) drag on that strip resists it: dF = 1/2 rho C_Dc H |v| v ds,
# giving a yaw moment  dM = -s dF .  Integrating over the body half-length a = L_n/2:
#
#       M_damp(psi_dot) = -1/2 rho C_Dc H_n INT_{-a}^{a} s |U sin b + psi_dot s|
#                                                         (U sin b + psi_dot s) ds .
#
# Two things fall straight out and both are physics, not choices:
#
#   * SELF-DAMPING (beta = 0):   M = -(1/4) rho C_Dc H a^4 psi_dot|psi_dot|
#     -- quadratic, and TINY (zeta ~ 2e-5 at the operating amplitude).  So at the
#     operating point the payload has essentially NO aerodynamic yaw damping.
#
#   * FLOW-COUPLED damping (beta != 0):  linearising at psi_dot = 0,
#         c_psi = rho C_Dc H_n |U sin beta| * (2 a^3 / 3)
#     -- proportional to |sin beta|, i.e. it VANISHES exactly where the divergence
#     lives (beta = 0) and only appears once the payload is already yawed.
#
# Consequence (this is a RESULT, not an omission): the mode that can diverge is the
# one with no aerodynamic damping.  Whatever damps it must be STRUCTURAL -- rope
# hysteresis, sheave friction -- which is why zeta_struct (params.DAMPING) is the
# governing uncertainty and is carried as an explicit, swept parameter.
#
# C_Dc is the CROSSFLOW drag coefficient of the section; we reuse the broadside box
# value drag_coefficient(pi/2) rather than invent a new number.

def crossflow_damping_moment(psi_dot, beta, U, n_gl=8):
    """
    Yaw moment [N m] from crossflow drag on a body yawing at psi_dot [rad/s] in a
    flow U [m/s] at incidence beta [rad].  Full nonlinear strip integral (Gauss-
    Legendre, exact for this smooth-except-at-one-point integrand at n_gl>=6);
    negative = opposing psi_dot.  No parameter that is not already in C_D.
    """
    a = 0.5 * float(PAYLOAD["L_n"])
    H = float(PAYLOAD["H_n"])
    C_Dc = drag_coefficient(np.pi / 2)                 # broadside section value
    xi, wi = np.polynomial.legendre.leggauss(n_gl)
    s = a * xi                                         # nodes on [-a, a]
    v = U * np.sin(beta) + psi_dot * s
    integ = s * np.abs(v) * v
    return -0.5 * float(ENV["rho_air"]) * C_Dc * H * a * float(np.dot(wi, integ))


def crossflow_damping_coeff(beta, U):
    """
    Linearised crossflow yaw-damping coefficient c_psi = -dM/dpsi_dot|_0 [N m s/rad].

        c_psi = rho C_Dc H_n |U sin beta| (2 a^3 / 3)

    Zero at beta = 0 -- the divergence point has no aerodynamic damping.
    """
    a = 0.5 * float(PAYLOAD["L_n"])
    H = float(PAYLOAD["H_n"])
    C_Dc = drag_coefficient(np.pi / 2)
    return float(ENV["rho_air"]) * C_Dc * H * abs(U * np.sin(beta)) * (2.0 * a ** 3 / 3.0)
