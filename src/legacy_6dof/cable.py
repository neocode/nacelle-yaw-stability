"""
Hoisting-cable model.

Two independent parts:

  (A) TORSION.  The suspension's torsional stiffness is GRAVITATIONAL, not elastic:
      the falls, spaced at radius S/2 on the hook block, form a multifilar pendulum.
      Rotating the block by psi raises it by delta = (S/2)^2 psi^2 / (2L), so

          U(psi) = W (S/2)^2 psi^2 / (2L)   =>   k_psi = W (S/2)^2 / L

      Neither E nor A appears.  Excitation is the rope's torque-tension coupling,
      M = C * T * d, summed over n falls carrying T = W/n each  =>  M = C * W * d,
      INDEPENDENT OF n.  Static twist:

          sin(psi) = 4 C d L / S^2        (weight cancels)

      This is identical to Bridon's industrial formula sin(theta)=4000*L*Tv/S^2
      (4000 = 4 x 1000, the m->mm conversion).  See tests/test_cable.py.

  (B) TRANSVERSE + AXIAL.  Distributed lumped-mass model with N elements, used
      OFFLINE to justify the N=1 reduction by modal separation.  Not used in the
      time-domain plant.
"""
import numpy as np
from scipy.linalg import eigh

from params import ENV, PAYLOAD, HOOK, ROPE, derive


# --------------------------------------------------------------------------
# (A)  TORSION  --  the industrial model
# --------------------------------------------------------------------------
def k_psi(L, n=None, DoD=None):
    """Torsional stiffness of the suspension [N m/rad]."""
    return derive(L=L, n=n, DoD=DoD).k_psi


def rope_torque(L=None, n=None, DoD=None):
    """
    Constant yaw moment delivered to the hook block [N m].

        M = n * C * T_fall * d ,  T_fall = W_load / n   =>   M = C * W_load * d

    n cancels.  Uses W_load (payload + block), NOT W_eff: the rope self-weight
    tensions the falls ABOVE the block, not at it.
    """
    dv = derive(L=L, n=n, DoD=DoD)
    return float(ROPE["C_torque"]) * dv.W_load * float(ROPE["d"])


def static_twist(L, n=None, DoD=None):
    """Hook-block angular displacement under rope torque [rad].  Bridon formula."""
    dv = derive(L=L, n=n, DoD=DoD)
    # sin(psi) = M / (k_psi)  ~  4 C d L / S^2 * (W_load / W_eff)
    # The ratio W_load/W_eff = 1/(1 + eps_rope/2) is the rope-self-weight correction;
    # it is exactly 1 in Bridon's massless-rope formula.
    s = (4.0 * float(ROPE["C_torque"]) * float(ROPE["d"]) * L / dv.S ** 2
         * dv.W_load / dv.W_eff)
    return np.arcsin(np.clip(s, -1.0, 1.0))


def cabling_length(n=None, DoD=None):
    """Suspension length at which sin(psi)=1 -> 'cabling' (Bridon) [m]."""
    dv = derive(n=n, DoD=DoD)
    return dv.S ** 2 / (4.0 * float(ROPE["C_torque"]) * float(ROPE["d"]))


def f_yaw(L, I_zz=None, n=None, DoD=None):
    """Undamped torsional natural frequency of the suspended payload [Hz]."""
    I = float(PAYLOAD["I_zz"]) if I_zz is None else I_zz
    return np.sqrt(k_psi(L, n, DoD) / I) / (2 * np.pi)


# --------------------------------------------------------------------------
# (B)  DISTRIBUTED MODEL  --  for the N_c reduction study only
# --------------------------------------------------------------------------
def modes_transverse(L, N, n=None, DoD=None):
    """
    Transverse (pendulum + string) modes of the hanging cable with tip payload.

    FE string model, N elements.  Node 0 fixed at the boom tip; node N carries the
    payload + hook block.  Element tension is the weight hanging below it, so the
    fundamental mode is the pendulum (omega^2 -> g/L as mu -> 0) and higher modes
    are the string modes.  Returns frequencies [Hz], ascending.
    """
    dv = derive(L=L, n=n, DoD=DoD)
    g = float(ENV["g"])
    m_tip = float(PAYLOAD["m_p"]) + float(HOOK["m_hb"])
    le = L / N
    m_e = dv.mu_tot * le                       # mass of one element (all falls)

    # s measured DOWN from the boom tip; node i at s = i*le
    # tension at node i = weight of everything below it
    s = np.arange(N + 1) * le
    T_node = (m_tip + dv.mu_tot * (L - s)) * g
    T_el = 0.5 * (T_node[:-1] + T_node[1:])    # element tension

    K = np.zeros((N + 1, N + 1))
    M = np.zeros((N + 1, N + 1))
    for e in range(N):
        ke = T_el[e] / le
        K[e:e+2, e:e+2] += ke * np.array([[1.0, -1.0], [-1.0, 1.0]])
        M[e, e] += m_e / 2.0
        M[e+1, e+1] += m_e / 2.0
    M[N, N] += m_tip

    # node 0 fixed
    Kf, Mf = K[1:, 1:], M[1:, 1:]
    w2 = eigh(Kf, Mf, eigvals_only=True)
    w2 = np.clip(w2, 0.0, None)
    return np.sqrt(w2) / (2 * np.pi)


def modes_axial(L, N, n=None, DoD=None):
    """Axial (bouncing) modes: fixed-free spring-mass chain with tip payload [Hz]."""
    dv = derive(L=L, n=n, DoD=DoD)
    m_tip = float(PAYLOAD["m_p"]) + float(HOOK["m_hb"])
    le = L / N
    k_e = dv.EA_tot / le                       # all falls in parallel
    m_e = dv.mu_tot * le

    K = np.zeros((N + 1, N + 1))
    M = np.zeros((N + 1, N + 1))
    for e in range(N):
        K[e:e+2, e:e+2] += k_e * np.array([[1.0, -1.0], [-1.0, 1.0]])
        M[e, e] += m_e / 2.0
        M[e+1, e+1] += m_e / 2.0
    M[N, N] += m_tip

    Kf, Mf = K[1:, 1:], M[1:, 1:]
    w2 = eigh(Kf, Mf, eigvals_only=True)
    return np.sqrt(np.clip(w2, 0.0, None)) / (2 * np.pi)


def mass_ratio(L, n=None, DoD=None):
    """Total rope mass / payload mass  --  the small parameter of the reduction."""
    dv = derive(L=L, n=n, DoD=DoD)
    return dv.mu_tot * L / float(PAYLOAD["m_p"])
