"""
Shared parameters, derived quantities and figure style for the revised study.

Every baseline value is listed in Table 1 of the paper together
with its source.  Values marked ASSUMED are engineering assumptions and are
varied in the parameter studies.
"""
from dataclasses import dataclass
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG_PNG = os.path.join(ROOT, "figures", "png")
FIG_TIF = os.path.join(ROOT, "figures", "tiff")
for _d in (RES, FIG_PNG, FIG_TIF):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- environment
g = 9.80665            # m/s^2
rho_air = 1.225        # kg/m^3   (ISO standard atmosphere, IEC 61400-1)
rho_w = 1025.0         # kg/m^3

# -------------------------------------------------- payload: IEA 15 MW nacelle
m_p = 820_888.0        # kg       Gaertner et al. (2020); v1.0 nacelle mass properties, nacelle total incl. hub (190 t) and yaw system (100 t)
I_zz = 1.8682468e7     # kg m^2   same table, about the vertical axis through the CG of the total
L_n, W_n, H_n = 20.0, 10.0, 10.0   # m  rectangular envelope (ASSUMED from component geometry)
V_box = L_n * W_n * H_n            # m^3
A_side = L_n * H_n                 # m^2  side-face area (reference area)

# ----------------------------------------------- hook block, rigging and hoist rope
m_hb = 60_000.0        # kg   hook block + lifting yoke (ASSUMED)
d_rope = 0.076         # m    rope diameter (ASSUMED, sized below)
mu_rope = 26.6         # kg/m mass per metre (assumed; catalogue 76 mm rotation-resistant offshore crane rope: 28.9 kg/m)
MBL = 5.3e6            # N    minimum breaking force, 76 mm rotation-resistant offshore crane rope (ArcelorMittal Notor HP/35(W)xK31WS)
C_torque = 0.0076      # -    torque factor, approximate maximum for a rotation-resistant rope at 20 % MBF (Bridon Dyform 34LR)
n_falls = 12           # -    falls (rope parts) between upper and lower blocks
n_sheaves = 6          # -    sheaves per block (two falls per sheave)
D_sheave = 2.4         # m    sheave pitch diameter, upper and lower blocks (ASSUMED)
p_sheave = 0.25        # m    axial pitch of the sheaves on the block axle (ASSUMED)

# ------------------------------------------------------------ crane vessel
# Main particulars and crane data of the floating heavy-lift crane vessel
# "Les Alizes" (Jan De Nul datasheet): LOA 236.8 m, B 52 m, moulded depth 16 m,
# max draught 10.5 m; Huisman crane, auxiliary block 1500 t at 46 m,
# lift above deck of the auxiliary block 167 m at 29.5 m radius.
V_LOA, V_B, V_DEPTH = 236.8, 52.0, 16.0      # m   datasheet
V_T = 9.0              # m   operating draught (ASSUMED; max. 10.5 m)
V_KG = 18.0            # m   CG above keel incl. crane and suspended load (ASSUMED)
AUX_LIFT_ABOVE_DECK = 167.0   # m  datasheet, auxiliary block at 29.5 m radius
AUX_RADIUS = 29.5             # m  datasheet
L_MIN_BLOCKS = 6.0     # m   minimum distance between upper and lower block (ASSUMED)
BOOM_TIP_HEIGHT = AUX_LIFT_ABOVE_DECK + (V_DEPTH - V_T) + L_MIN_BLOCKS   # m above MSL = 180

# -------------------------------------------------------- lift sequence
z_hub = 150.0          # m   hub height of the IEA 15 MW turbine above MSL
Z_CG_BELOW_BLOCK = 17.0  # m  payload CG below the lower block (yoke, slings, half nacelle; ASSUMED)
L_mating = BOOM_TIP_HEIGHT - Z_CG_BELOW_BLOCK - z_hub      # m  = 13 m (payload CG at hub height)
L_liftoff = 150.0      # m   fall length when the nacelle leaves the deck (CG ~13 m above MSL)
L_ref = 100.0          # m   intermediate reference length (original manuscript baseline)
alpha_shear = 0.14     # -   power-law shear exponent, IEC 61400-3-1 (offshore)

# ------------------------------------------------------------------- turbulence
TI_values = (0.06, 0.12)  # -  typical offshore and conservative turbulence intensity
r_sigma_v = 0.8           # -  sigma_v / sigma_u, IEC 61400-1 Kaimal model


def Lambda1(z):
    """IEC 61400-1 turbulence scale parameter [m]."""
    return 0.7 * z if z < 60.0 else 42.0


# -------------------------------------------------------------------- damping
zeta_values = (0.002, 0.01, 0.05)   # structural yaw damping ratio (not measured; swept)

# ----------------------------------------------------------------------- waves
Hs, Tp, gamma_J = 1.5, 7.65, 3.3   # expected sea state at 10 m/s, IEA 15 MW reference site (Stewart et al. 2016)

# ---------------------------------------------------------------------- tuggers
M_t_max = 4.0e5        # N m   available tugger yaw couple (four 100 kN lines, ASSUMED)
k_p_nom = 2.0e6        # N m/rad  proportional (stiffness) gain (ASSUMED)
k_d_nom = 6.0e6        # N m s/rad derivative (damping) gain (ASSUMED)


# ======================================================================
#  Reeving geometry: falls on two rows of a common sheave axle
# ======================================================================
def fall_positions(n_sh=n_sheaves, D=D_sheave, p=p_sheave):
    """Horizontal positions (x, y) of the 2*n_sh falls on a block [m].

    Falls leave each sheave on both sides of its axle, i.e. at x = +-D/2,
    and the sheaves are stacked along the axle (y) with pitch p.
    The ordering follows the rope path of a single continuous rope
    (used only for the sheave-friction study)."""
    y = (np.arange(n_sh) - 0.5 * (n_sh - 1)) * p
    pts = []
    for j in range(n_sh):
        pts.append((+0.5 * D, y[j]))
        pts.append((-0.5 * D, y[j]))
    return np.array(pts)


def r_eq2(a=None, b=None, w=None):
    """Tension-weighted mean of a_i . b_i [m^2] (equivalent lever arm squared)."""
    if a is None:
        a = fall_positions()
    if b is None:
        b = a
    n = len(a)
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w) / np.sum(w)
    return float(np.sum(w * np.einsum("ij,ij->i", a, b)))


@dataclass
class Susp:
    L: float          # fall length [m]
    W_load: float     # payload + block weight [N]
    W_rope: float     # weight of all falls [N]
    W_eff: float      # weight governing k_psi [N]
    r2: float         # equivalent lever arm squared [m^2]
    k_psi: float      # gravitational torsional stiffness [N m/rad]
    M_r: float        # rope torque [N m]
    psi0: float       # static twist in still air [rad]
    f_psi: float      # natural frequency [Hz]
    T_psi: float      # natural period [s]
    f_pend: float     # pendulum frequency [Hz]


def suspension(L, r2=None):
    """Derived suspension quantities for fall length L."""
    r2 = r_eq2() if r2 is None else r2
    W_load = (m_p + m_hb) * g
    W_rope = n_falls * mu_rope * L * g
    W_eff = W_load + 0.5 * W_rope        # raising the block also raises the rope CG by half
    k = W_eff * r2 / L
    M_r = C_torque * W_load * d_rope
    psi0 = float(np.arcsin(min(M_r / k, 1.0)))
    f = np.sqrt(k / I_zz) / (2 * np.pi)
    return Susp(L, W_load, W_rope, W_eff, r2, k, M_r, psi0, f, 1.0 / f,
                np.sqrt(g / (L + Z_CG_BELOW_BLOCK)) / (2 * np.pi))


def z_payload(L):
    """Height of the payload CG above MSL for fall length L."""
    return BOOM_TIP_HEIGHT - L - Z_CG_BELOW_BLOCK


def wind_at(z, U_hub):
    """Mean wind speed at height z for a power-law profile."""
    return U_hub * (z / z_hub) ** alpha_shear


# ======================================================================
#  Figure style (single style for every figure of the paper)
# ======================================================================
# Categorical slots in fixed order (validated default palette) + neutral ink.
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [C1, C2, C3, C4, C5]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v"]

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM      # single and double column widths (Elsevier)


def setup_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "none",
        "legend.facecolor": "white",
        "legend.borderpad": 0.3,
        "legend.handlelength": 2.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name):
    """Save PNG (600 dpi) and LZW-compressed TIFF (600 dpi)."""
    from PIL import Image
    png = os.path.join(FIG_PNG, name + ".png")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    im = Image.open(png).convert("RGB")
    im.save(os.path.join(FIG_TIF, name + ".tif"), compression="tiff_lzw", dpi=(600, 600))
    return png


if __name__ == "__main__":
    for L in (L_liftoff, L_ref, L_mating):
        s = suspension(L)
        print(f"L={L:6.1f} m  r_eq={np.sqrt(s.r2):.3f} m  k={s.k_psi:.4g} N m/rad  "
              f"f={s.f_psi*1e3:.2f} mHz T={s.T_psi:.1f} s  psi0={np.degrees(s.psi0):.2f} deg  "
              f"M_r={s.M_r:.0f} N m  f_pend={s.f_pend:.3f} Hz")
    print("safety factor", n_falls * MBL / ((m_p + m_hb) * g))
