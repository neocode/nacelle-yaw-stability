"""
The suspended payload: 6-DOF rigid body, 13-state, NED throughout (frames.py).

STATE     x = [ p_I(3) | q(4) | v_B(3) | omega_B(3) ]        13 states
          p_I    CG position, NED  (height = -p_I[2])
          q      attitude, BODY -> INERTIAL, scalar first
          v_B    CG velocity in BODY
          omega_B  angular rate in BODY

DYNAMICS  (Newton-Euler about the CG, body frame -- Fossen)
          p_dot_I  = R(q) v_B
          q_dot    = 0.5 q (x) [0, omega_B]
          v_dot_B  = F_B/m - omega x v_B
          w_dot_B  = J^-1 ( M_B - omega x (J omega) )

-------------------------------------------------------------------------------
WHAT THE RIGID BODY IS.  Nacelle (820 888 kg) + hook block (60 000 kg), rigidly
slung.  m = m_p + m_hb = W_load/g -- the SAME W_load that sets the rope torque in
cable.py, so the static cable tension comes out equal to derive().W_load by
construction (there is a test).  The block's own rotational inertia is neglected:
60 t at ~2 m gives ~2.4e5 kg m^2, i.e. 1.3 % of I_zz.  Flagged, sensitivity-testable.

-------------------------------------------------------------------------------
SUSPENSION.  Reduced model, N_c = 1 (justified in Phase 2.3: the first discarded
mode is at 0.543 Hz, 4.2x above the wave peak).

  AXIAL: a spring between the boom tip and the lift point,
      T = (EA_tot / L0) * (l - L0)     for l > L0,     0 otherwise  (a rope pulls)

  TORSION: the multifilar pendulum of R1.  The EXACT restoring moment is

      M_restore = -k_psi * sin(psi_rel),        k_psi = W_eff (S/2)^2 / L

  NOT -k_psi * psi_rel.  Derivation: rotating the block by psi shortens the
  vertical drop to h = sqrt(L^2 - 2R^2(1-cos psi)), R = S/2, so it RISES by
  delta = R^2 (1 - cos psi)/L and U(psi) = W R^2 (1 - cos psi)/L, whence
  M = -dU/dpsi = -W R^2 sin(psi) / L = -k_psi sin(psi).
  Small angles give back k_psi psi, and setting k_psi sin(psi) = M_rope gives back
  Bridon's sin(psi) = 4 C d L / S^2 EXACTLY -- which is why the equilibrium of this
  ODE reproduces cable.static_twist() to machine precision (there is a test).
  At the 3.6 deg baseline sin(psi) vs psi differs by 0.07 %; at Bridon's 30 deg
  cabling limit, by 4.7 %.  The linear form is therefore NOT admissible over the
  operational range and is not used.

  EXCITATION: the constant rope torque M_rope = C * W_load * d (independent of the
  number of falls -- cable.py).

-------------------------------------------------------------------------------
AERODYNAMICS.  Munk moment (aero.py) about the vertical, plus orientation-dependent
drag at the CG.  Two deliberate omissions, both stated in the paper:

  * NO aerodynamic yaw DAMPING.  Rotation-induced crossflow would add damping.
    It cannot move the STATIC divergence threshold -- that is the entire content of
    R4 -- so omitting it is conservative for stability and honest for the claim.
    It DOES matter after divergence: without damping the payload oscillates about
    the post-divergence equilibrium instead of settling on it.  See
    psi_post_divergence() -- the equilibrium ANGLE is damping-independent, only the
    approach to it is not.
  * NO added-mass INERTIA of the air.  rho_air * V * k ~ 1.7e3 kg against 8.8e5 kg
    of payload: 0.2 %.  The Munk MOMENT is kept; the added INERTIA is dropped.
    (There is a test on that ratio.)
"""
import numpy as np
from scipy.integrate import solve_ivp

import frames as fr
import cable
import aero
from params import ENV, PAYLOAD, HOOK, ROPE, CRANE, DAMPING, derive


# ==========================================================================
#  STATE PACKING
# ==========================================================================
IP, IQ, IV, IW = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13)
N_STATE = 13


def pack(p_I, q, v_B, omega_B):
    return np.concatenate([np.asarray(p_I, float), fr.quat_normalize(q),
                           np.asarray(v_B, float), np.asarray(omega_B, float)])


def unpack(x):
    x = np.asarray(x, float)
    return x[IP], x[IQ], x[IV], x[IW]


# ==========================================================================
#  THE RIGID BODY  --  pure function, no physics of the crane in it
# ==========================================================================
def rigid_body_derivative(x, F_B, M_B, m, J, J_inv=None):
    """
    Newton-Euler about the CG.  F_B, M_B are the TOTAL body-frame wrench.
    Kept free of any application logic so it can be tested against conservation
    laws in isolation (torque-free tumbling, free fall).
    """
    p_I, q, v_B, w_B = unpack(x)
    J_inv = np.linalg.inv(J) if J_inv is None else J_inv
    return np.concatenate([
        fr.rotate(q, v_B),                                   # p_dot_I
        fr.quat_dot(q, w_B),                                 # q_dot
        F_B / m - np.cross(w_B, v_B),                        # v_dot_B
        J_inv @ (M_B - np.cross(w_B, J @ w_B)),              # w_dot_B
    ])


# ==========================================================================
#  THE PAYLOAD
# ==========================================================================
class Payload:
    """
    Nacelle + hook block on the hoist, in a wind field.

    wind : callable (t, p_I) -> inertial wind velocity [3] in NED, or None (still air).
    slew : callable t -> boom slew angle alpha [rad], or a constant.  The suspension
           twists the payload toward alpha; this is the ONLY yaw authority the crane
           has, and R5 says it is not enough.  Tugger lines are Phase 6.
    """

    def __init__(self, L=None, h_att=None, boom_tip=None, wind=None, slew=0.0,
                 use_suspension=True, use_aero=True, use_gravity=True,
                 rope_torque=True, use_damping=True, zeta_struct=None,
                 use_crossflow_damping=True):
        self.L = float(ROPE["L0"]) if L is None else float(L)
        self.dv = derive(L=self.L)

        # mass: nacelle + hook block  ->  W_load, the same one cable.py uses
        self.m = float(PAYLOAD["m_p"]) + float(HOOK["m_hb"])
        self.J = np.diag([float(PAYLOAD["I_xx"]), float(PAYLOAD["I_yy"]),
                          float(PAYLOAD["I_zz"])])
        self.J_inv = np.linalg.inv(self.J)

        # lift point: on the yaw axis, above the CG.  BODY z is DOWN => above is -z.
        self.h_att = 0.5 * float(PAYLOAD["H_n"]) if h_att is None else float(h_att)
        self.r_a_B = np.array([0.0, 0.0, -self.h_att])

        # boom tip, NED.  May be a FIXED vector or a CALLABLE t -> [3] (the plant
        # feeds a moving tip: vessel heave/pitch + crane kinematics).  Default: a fixed
        # point directly above the payload's hanging position.
        if boom_tip is None:
            h_tip = float(ENV["z_hub"]) + self.h_att + self.L
            boom_tip = fr.from_height(float(CRANE["L_boom"]), 0.0, h_tip)
        if callable(boom_tip):
            self._boom_tip_fn = boom_tip
            fr.assert_above_water(np.asarray(boom_tip(0.0), float), "boom tip")
        else:
            tip = np.asarray(boom_tip, float)
            fr.assert_above_water(tip, "boom tip")
            self._boom_tip_fn = lambda t, _tip=tip: _tip
        # keep the t=0 tip as the reference used by equilibrium()
        self.boom_tip = np.asarray(self._boom_tip_fn(0.0), float)

        self.k_ax = self.dv.EA_tot / self.L
        self.k_psi = self.dv.k_psi
        self.M_rope = cable.rope_torque(L=self.L) if rope_torque else 0.0

        self.wind = wind
        self.slew = slew if callable(slew) else (lambda t, a=float(slew): a)
        self.use_suspension = use_suspension
        self.use_aero = use_aero
        self.use_gravity = use_gravity
        self.use_damping = use_damping
        self.use_crossflow_damping = use_crossflow_damping
        self.zeta_struct = (float(DAMPING["zeta_struct"]) if zeta_struct is None
                            else float(zeta_struct))

    # ------------------------------------------------------------------
    def boom_tip_at(self, t):
        """Boom-tip position [3] NED at time t (fixed or vessel/crane-driven)."""
        return np.asarray(self._boom_tip_fn(t), float)

    # ------------------------------------------------------------------
    #  EQUILIBRIUM
    # ------------------------------------------------------------------
    def equilibrium(self, psi=None):
        """
        Hanging at rest: the cable stretched so that T = m g, yawed to the angle at
        which the rope torque is balanced by the multifilar restoring moment.
        psi=None -> use the analytic Bridon twist (cable.static_twist).
        """
        stretch = self.m * float(ENV["g"]) / self.k_ax
        p_att = self.boom_tip + np.array([0.0, 0.0, self.L + stretch])   # NED: +z = down
        if psi is None:
            psi = float(cable.static_twist(self.L)) if self.M_rope else 0.0
        q = fr.euler_to_quat(0.0, 0.0, psi)
        p_cg = p_att - fr.rotate(q, self.r_a_B)
        return pack(p_cg, q, np.zeros(3), np.zeros(3))

    # ------------------------------------------------------------------
    #  WRENCHES  (all returned in the BODY frame, about the CG)
    # ------------------------------------------------------------------
    def wrench_gravity(self, x):
        _, q, _, _ = unpack(x)
        return fr.rotate_inv(q, self.m * fr.g_I()), np.zeros(3)

    def wrench_suspension(self, t, x):
        p_I, q, _, _ = unpack(x)
        R = fr.quat_to_dcm(q)
        r_a_I = R @ self.r_a_B
        d = (p_I + r_a_I) - self.boom_tip_at(t)     # boom tip -> lift point, NED
        ell = np.linalg.norm(d)
        e = d / ell                                # points DOWN the cable

        # axial: a rope pulls, never pushes
        T = self.k_ax * max(ell - self.L, 0.0)
        F_I = -T * e
        M_I = np.cross(r_a_I, F_I)                 # moment of the cable force about CG

        # torsion: multifilar pendulum, EXACT (sin, not the linearisation)
        psi_rel = fr.yaw(q) - self.slew(t)
        psi_rel = np.arctan2(np.sin(psi_rel), np.cos(psi_rel))          # wrap
        M_tw = -self.k_psi * np.sin(psi_rel) + self.M_rope
        M_I = M_I + M_tw * e                       # applied about the cable axis

        return R.T @ F_I, R.T @ M_I

    def wrench_aero(self, t, x):
        p_I, q, v_B, w_B = unpack(x)
        U_w = np.zeros(3) if self.wind is None else np.asarray(self.wind(t, p_I), float)
        U_rel = U_w - fr.rotate(q, v_B)            # inertial relative wind

        Uh = U_rel[:2]
        speed = float(np.linalg.norm(Uh))
        if speed < 1e-9:
            return np.zeros(3), np.zeros(3)
        chi = np.arctan2(Uh[1], Uh[0])             # flow heading, N -> E about +z(down)

        ex_I = fr.rotate(q, [1.0, 0.0, 0.0])       # nacelle long axis in NED
        psi_b = np.arctan2(ex_I[1], ex_I[0])
        beta = np.arctan2(np.sin(psi_b - chi), np.cos(psi_b - chi))     # signed, wrapped

        # Munk: destabilising, about the VERTICAL (NED +z = down)
        M_z = aero.munk_moment(beta, speed)
        # crossflow (strip) yaw damping -- from the SAME C_D, ~|sin beta|, ~0 at beta=0
        if self.use_crossflow_damping:
            M_z += aero.crossflow_damping_moment(float(w_B[2]), beta, speed)
        M_I = np.array([0.0, 0.0, M_z])
        # drag at the CG, along the flow
        F_I = np.array([*(aero.drag_force(beta, speed) * Uh / speed), 0.0])

        R = fr.quat_to_dcm(q)
        return R.T @ F_I, R.T @ M_I

    def wrench_damping(self, x):
        """
        STRUCTURAL yaw damping of the suspension (rope hysteresis, sheave friction).
        Linear, referenced to the STILL-AIR stiffness because it is a property of the
        hardware, not of the wind:

            M = -c_s omega_z ,   c_s = 2 zeta_s sqrt(k_psi I_zz) = 2 zeta_s omega_n I_zz

        This is the paper's governing uncertainty; zeta_s is an explicit swept param.
        Applied about the body z (the yaw axis), so it enters M_B directly.
        """
        _, _, _, w_B = unpack(x)
        c_s = 2.0 * self.zeta_struct * np.sqrt(self.k_psi * float(PAYLOAD["I_zz"]))
        return np.zeros(3), np.array([0.0, 0.0, -c_s * float(w_B[2])])

    # ------------------------------------------------------------------
    def derivative(self, t, x):
        F = np.zeros(3)
        M = np.zeros(3)
        if self.use_gravity:
            f, m_ = self.wrench_gravity(x);        F, M = F + f, M + m_
        if self.use_suspension:
            f, m_ = self.wrench_suspension(t, x);  F, M = F + f, M + m_
        if self.use_aero:
            f, m_ = self.wrench_aero(t, x);        F, M = F + f, M + m_
        if self.use_damping:
            f, m_ = self.wrench_damping(x);        F, M = F + f, M + m_
        return rigid_body_derivative(x, F, M, self.m, self.J, self.J_inv)

    # ------------------------------------------------------------------
    def simulate(self, T, dt=0.5, x0=None, rtol=1e-10, atol=1e-12, method="DOP853"):
        """Integrate.  Returns (t, X) with X.shape = (13, N).  q is NOT renormalised:
        the norm drift is left visible so the tests can measure it."""
        x0 = self.equilibrium() if x0 is None else np.asarray(x0, float)
        t_eval = np.arange(0.0, T + 0.5 * dt, dt)
        sol = solve_ivp(self.derivative, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                        method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t, sol.y

    # ------------------------------------------------------------------
    #  DIAGNOSTICS
    # ------------------------------------------------------------------
    @staticmethod
    def yaw_of(X):
        """psi(t) [rad] from a state history."""
        return np.array([fr.yaw(X[IQ, k]) for k in range(X.shape[1])])

    def cable_tension(self, x):
        p_I, q, _, _ = unpack(x)
        d = (p_I + fr.quat_to_dcm(q) @ self.r_a_B) - self.boom_tip
        return self.k_ax * max(float(np.linalg.norm(d)) - self.L, 0.0)

    def f_yaw_linear(self, U=0.0):
        """
        Linearised yaw frequency in wind [Hz], and the growth rate above divergence.
        k_eff = k_psi - rho U^2 V (k2-k1).  k_eff < 0  =>  STATIC DIVERGENCE.
        Returns (f_Hz, lambda_growth_1/s); exactly one of them is non-zero.
        """
        k_eff = self.k_psi - aero.aero_yaw_stiffness(U)
        I = float(PAYLOAD["I_zz"])
        if k_eff > 0.0:
            return np.sqrt(k_eff / I) / (2 * np.pi), 0.0
        return 0.0, np.sqrt(-k_eff / I)

    def psi_post_divergence(self, U):
        """
        POST-DIVERGENCE EQUILIBRIUM [rad].  Divergence is a SUPERCRITICAL PITCHFORK,
        not a runaway.  The full nonlinear yaw moment is

            M(psi) = -k_psi sin(psi) + 0.5 rho U^2 V (k2-k1) sin(2 psi)

        and M(psi) = 0 has, besides psi = 0, a second root whenever U > U_div:

            k_psi sin psi = rho U^2 V (k2-k1) sin psi cos psi
            =>  cos(psi*) = (U_div / U)^2                                      (!)

        So above the critical speed the payload does NOT spin freely: it snaps to a
        LARGE FINITE yaw offset and sits there.  U = 1.2 U_div  ->  psi* = 46 deg.
        U = 1.5 U_div  ->  64 deg.

        This is the operational meaning of R4 and it is sharper than "loss of
        stability": against a mating tolerance of 1 deg (PAYLOAD['psi_tol']), crossing
        U_div does not degrade the lift, it ENDS it.  It also explains why the
        industry wind limit is a hard cut-off rather than a soft derating.

        Returns NaN below U_div, where psi = 0 is the only equilibrium.
        """
        U_div = aero.U_divergence(self.k_psi)
        if U <= U_div:
            return np.nan
        return float(np.arccos((U_div / U) ** 2))

    def added_mass_ratio(self):
        """rho_air V (k2-k1) / m  --  why the added INERTIA of the air is dropped."""
        return float(ENV["rho_air"]) * self.dv.V_box * self.dv.k2mk1 / self.m
