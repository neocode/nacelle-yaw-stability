"""
Frames, attitude, and the ONE place where the coordinate convention lives.

WHY THIS FILE EXISTS.  Mixing NED and z-up conventions inside one model is a
class of bug that is invisible to inspection and fatal to results, so it is fixed
STRUCTURALLY: nothing else in the repository is allowed to take a sign or a
rotation into its own hands.  If you need a height, you call height().  If you need
gravity, you call g_I().  If you rotate, you use rotate().

-------------------------------------------------------------------------------
THE CONVENTION  (and there is exactly one)

Inertial frame {I}:  NED.
    x_I -> North          y_I -> East          z_I -> DOWN
    origin at the mean water level.
    gravity  g_I = [0, 0, +g]        (POSITIVE, because z points down)
    height above water   h = -z      (a point 150 m up has z = -150)

Body frame {B}:  fixed in the payload, origin at the payload CG.
    x_B -> along the nacelle long axis (the 20 m dimension, drivetrain axis)
    y_B -> starboard        z_B -> down
    the inertia tensor of params.PAYLOAD is diagonal in {B} and about the CG.

Attitude:  unit quaternion q = [w, x, y, z], SCALAR FIRST, Hamilton product,
    mapping BODY -> INERTIAL:        v_I = R(q) v_B
Euler angles:  ZYX (3-2-1), the marine standard (Fossen):
    R(q) = Rz(psi) Ry(theta) Rx(phi)
    phi = roll,  theta = pitch,  psi = YAW  <- psi is THE state of this paper.

Kinematics:  q_dot = 0.5 * q (x) [0, omega_B]

Euler angles are used for REPORTING ONLY.  They are singular at theta = +-90 deg;
the state vector carries the quaternion.  test_frames.py demonstrates the failure
of the Euler representation and the survival of the quaternion at the same attitude.
"""
import numpy as np

from params import ENV


# ==========================================================================
#  THE CONVENTION -- call these, never hard-code a sign
# ==========================================================================
def g_I():
    """Gravity in the inertial frame [m/s^2].  NED => POSITIVE z."""
    return np.array([0.0, 0.0, float(ENV["g"])])


def height(p_I):
    """Height above mean water level [m] of a point given in NED.  h = -z."""
    p = np.asarray(p_I, float)
    return -p[..., 2]


def from_height(north, east, h):
    """Build an NED position from a height above water.  Inverse of height()."""
    return np.array([north, east, -h], float)


def assert_above_water(p_I, what="point"):
    if np.any(height(p_I) <= 0.0):
        raise ValueError(
            f"{what} is at or below the water line (z >= 0 in NED). "
            "Heights must be entered as z = -h.  See frames.py header.")


# ==========================================================================
#  ALGEBRA
# ==========================================================================
def skew(v):
    """[v]x such that [v]x a = v cross a."""
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_normalize(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero quaternion")
    q = q / n
    return -q if q[0] < 0.0 else q          # canonical: w >= 0, kills the double cover


def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_conj(q):
    w, x, y, z = np.asarray(q, float)
    return np.array([w, -x, -y, -z])


def quat_mult(a, b):
    """Hamilton product a (x) b.  Composition: R(a (x) b) = R(a) R(b)."""
    aw, ax, ay, az = np.asarray(a, float)
    bw, bx, by, bz = np.asarray(b, float)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])


def quat_to_dcm(q):
    """R(q): BODY -> INERTIAL.  v_I = R v_B."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def dcm_to_quat(R):
    """Shepperd's method: pick the largest denominator, never divide by ~0."""
    R = np.asarray(R, float)
    t = np.trace(R)
    cands = [t, R[0, 0], R[1, 1], R[2, 2]]
    i = int(np.argmax(cands))
    if i == 0:
        s = np.sqrt(1.0 + t) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif i == 1:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif i == 2:
        s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def rotate(q, v):
    """v_I = R(q) v_B."""
    return quat_to_dcm(q) @ np.asarray(v, float)


def rotate_inv(q, v):
    """v_B = R(q)^T v_I."""
    return quat_to_dcm(q).T @ np.asarray(v, float)


# ==========================================================================
#  EULER (REPORTING ONLY -- the state carries the quaternion)
# ==========================================================================
def euler_to_quat(phi, theta, psi):
    """ZYX (3-2-1): roll phi, pitch theta, yaw psi [rad]."""
    cp, sp = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cs, ss = np.cos(psi / 2), np.sin(psi / 2)
    return quat_normalize(np.array([
        cs * ct * cp + ss * st * sp,
        cs * ct * sp - ss * st * cp,
        cs * st * cp + ss * ct * sp,
        ss * ct * cp - cs * st * sp]))


def quat_to_euler(q):
    """(phi, theta, psi) [rad].  SINGULAR at theta = +-90 deg -- reporting only."""
    w, x, y, z = quat_normalize(q)
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    theta = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return phi, theta, psi


def yaw(q):
    """psi [rad] -- the state this whole paper is about."""
    return quat_to_euler(q)[2]


# ==========================================================================
#  KINEMATICS
# ==========================================================================
def quat_dot(q, omega_B):
    """q_dot = 0.5 q (x) [0, omega_B]   for q: BODY -> INERTIAL."""
    return 0.5 * quat_mult(q, np.concatenate(([0.0], np.asarray(omega_B, float))))


def quat_integrate(q, omega_B, dt):
    """
    Exact exponential map over dt for constant omega_B: no norm drift, no
    small-angle error.  Used by the tests as the reference against which the
    ODE integrator's quaternion is checked.
    """
    w = np.asarray(omega_B, float)
    th = np.linalg.norm(w) * dt
    if th < 1e-14:
        return quat_normalize(q)
    ax = w / np.linalg.norm(w)
    dq = np.concatenate(([np.cos(th / 2)], np.sin(th / 2) * ax))
    return quat_normalize(quat_mult(q, dq))
