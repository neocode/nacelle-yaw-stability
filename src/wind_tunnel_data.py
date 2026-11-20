"""
Measured static yawing moments of rectangular boxes (Section 4.1).

Source: Laub GH, Kodani HM, NASA TM-X-62169 (1972), pitch angle 0:
  Tables VIII-1..3  cargo containers No. 1-3 (data corrected for strut interference),
  Table II-1        cargo container No. 1, uncorrected, yaw angles -50..90 deg,
  Tables AI-1, AII-1, AIII-1  smooth-sided boxes No. 1-3 of the same dimensions
                    (corrected for the strut loads only, not for interference).
Full-scale coefficients based on unit dimensions: yawing moment / q in ft^3.
Sign convention of the report: yaw angle and yawing moment both positive nose
right, so a negative slope is restoring.  Moments are about the geometric centre.

Each data set is normalised by A_s L (side-face area x length) and a constant
balance offset is removed (the moments at 0 and 90 deg vanish by symmetry).
The box closest to the payload envelope (L/W = 2.5, H/W = 1) is fitted with
    c(beta) = sum_k a_k sin(2 k beta),
which is odd and antisymmetric about 90 deg (fore-aft symmetric box).
"""
import numpy as np

FT = 0.3048
# dimensions W x H x L in ft
DIMS = {1: (8.0, 8.0, 20.0), 2: (8.0, 8.0, 40.0), 3: (8.0, 8.0, 8.0)}

YM = {
    "container 1": (1, [0, 4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90],
                    [30.4, -71.6, -131.5, -118.5, -57.2, -4.2, 72.5, 114.6, 158.6, 198.3, 246.1,
                     273.8, 285.8, 235.2, 48.2]),
    "container 2": (2, [0, 4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 60, 75],
                    [37.2, 16.1, 26.9, 107.7, 240.9, 389.7, 532.1, 642.8, 740.1, 876.5, 1048.9,
                     988.0, 1308.6, 846.6]),
    "container 3": (3, [0, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 60, 75],
                    [13.7, -6.6, -30.0, -44.2, -45.6, -36.2, -22.1, -7.1, 2.9, 14.2, 27.6, 48.0,
                     72.0]),
    "container 1 uncorrected": (1, [-50, -45, -40, -35, -30, -25, -20, -16, -8, 0, 4, 8, 12, 16, 20,
                                     25, 30, 35, 40, 45, 50, 60, 75, 90],
                                [-251.0, -244.6, -213.9, -181.0, -141.8, -95.0, -34.3, 21.8, 119.3, 30.7,
                                 -60.2, -97.6, -68.1, -13.4, 40.9, 111.6, 165.8, 206.0, 242.7, 273.5,
                                 296.0, 291.7, 237.0, 38.5]),
    "smooth box 1": (1, [0, 8, 16, 30, 45, 60, 75, 90],
                     [29.9, -94.2, 3.0, 201.6, 245.4, 270.2, 224.6, 31.9]),
    "smooth box 2": (2, [0, 8, 16, 30, 45, 60, 75, 90],
                     [9.2, 54.9, 352.0, 934.9, 1109.9, 1262.4, 869.1, 58.1]),
    "smooth box 3": (3, [0, 8, 16, 30, 45, 90],
                     [9.3, -18.8, -49.2, -19.1, 10.9, 4.2]),
}
FIT_SETS = ("container 1", "smooth box 1")      # L/W = 2.5, nominal data choice
K_NOMINAL = 7
K_RANGE = (4, 5, 6, 7)
# alternative, equally defensible choices of the 2.5:1 data (uncertainty envelope)
DATA_CHOICES = {
    "corrected container + smooth box": FIT_SETS,
    "corrected container": ("container 1",),
    "uncorrected container": ("container 1 uncorrected",),
    "uncorrected container + smooth box": ("container 1 uncorrected", "smooth box 1"),
}


def ASL(box):
    W, H, L = DIMS[box]
    return L * H * L


def normalised(name):
    """Yaw angle [deg] and C_YM = YM / (A_s L), raw (offset not removed)."""
    box, b, m = YM[name]
    return np.array(b, float), np.array(m, float) / ASL(box)


def fit(K=K_NOMINAL, sets=FIT_SETS):
    """Least-squares sine series with one constant offset per data set.
    Returns (a_1..a_K, offsets, rms residual)."""
    X, y = [], []
    for j, nm in enumerate(sets):
        b, c = normalised(nm)
        br = np.radians(b)
        off = np.zeros((len(b), len(sets)))
        off[:, j] = 1.0
        X.append(np.column_stack([off] + [np.sin(2 * k * br) for k in range(1, K + 1)]))
        y.append(c)
    X, y = np.vstack(X), np.concatenate(y)
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - X @ p) ** 2)))
    return p[len(sets):], dict(zip(sets, p[:len(sets)])), rms


def corrected(name, K=K_NOMINAL, sets=None):
    """Data with the balance offset removed: offset from the fit for the fitted
    sets, otherwise the mean of the values at 0 deg and at the largest angle
    when it is 90 deg (else the value at 0 deg)."""
    b, c = normalised(name)
    sets = FIT_SETS if sets is None else sets
    if name in sets:
        off = fit(K, sets)[1][name]
    elif b[-1] == 90.0:
        off = 0.5 * (c[0] + c[-1])
    else:
        off = c[0]
    return b, c - off


if __name__ == "__main__":
    for K in (4, 5, 6, 7):
        a, off, rms = fit(K)
        bb = np.radians(np.linspace(0, 90, 901))
        k = np.arange(1, K + 1)
        c = (a * np.sin(2 * k * bb[:, None])).sum(1)
        dc = (2 * k * a * np.cos(2 * k * bb[:, None])).sum(1)
        print(K, np.round(a, 4), {n: round(v, 4) for n, v in off.items()}, f"rms={rms:.4f}",
              f"CK={dc[0]:.3f} max c'={dc.max():.3f} at {np.degrees(bb[dc.argmax()]):.1f}")
