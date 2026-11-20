"""
Parameter set for the offshore crane / nacelle-lift model.

RULE: every parameter carries a `source` string. A parameter with source=None
raises on validate(), so that no number enters the model without provenance.

Sign convention: NED  (x North, y East, z DOWN).  g_I = [0, 0, +g].
"""
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

# --------------------------------------------------------------------------
SOURCES = {
    "NREL": "Gaertner et al. (2020), NREL/TP-5000-75698, IEA Wind 15 MW RWT",
    "BRIDON": "Bridon American, Wire Rope Technical Information (via Mazzella, "
              "published with permission)",
    "LAMB": "Lamb (1932) Hydrodynamics §114; Newman, Marine Hydrodynamics",
    "IEC": "IEC 61400-1 Ed.4 (Kaimal spectrum, exponential coherence)",
    "IEC3": "IEC 61400-3-1:2019 (offshore wind profile exponent)",
    "DAVENPORT": "Davenport (1961), QJRMS 87:194 (exponential coherence)",
    "SHINOZUKA": "Shinozuka & Deodatis (1991), Appl. Mech. Rev. 44(4):191 "
                 "(spectral representation with Cholesky decomposition)",
    "TURBSIM": "Jonkman (2009), NREL/TP-500-46198, TurbSim User's Guide",
    "DNVC205": "DNV-RP-C205 (2010), Environmental Conditions and Environmental Loads",
    "CAPYTAINE": "Ancellin & Dias (2019), JOSS 4(36):1341; open-source BEM",
    "DERIVED": "derived in this work (see Appendix A)",
    "ASSUMED": "engineering assumption -- flagged, sensitivity-tested",
}


class P:
    """A parameter with units and provenance."""
    __slots__ = ("value", "unit", "source", "note")

    def __init__(self, value, unit, source, note=""):
        if source not in SOURCES:
            raise ValueError(f"unknown source tag {source!r}")
        self.value, self.unit, self.source, self.note = value, unit, source, note

    def __repr__(self):
        return f"{self.value!r} [{self.unit}] <{self.source}>"

    def __float__(self):
        return float(self.value)


# ==========================================================================
#  ENVIRONMENT
# ==========================================================================
ENV = dict(
    g       = P(9.80665,  "m/s^2",  "NREL"),
    rho_air = P(1.225,    "kg/m^3", "IEC"),
    rho_w   = P(1025.0,   "kg/m^3", "ASSUMED", "standard seawater"),
    # installation weather window for a nacelle lift (NOT a storm)
    U10     = P(10.0,     "m/s",    "ASSUMED", "nominal nacelle-lift wind limit"),
    Hs      = P(1.5,      "m",      "NREL",    "metocean Tab.6-2, U=10 m/s -> Hs=1.54"),
    Tp      = P(7.65,     "s",      "NREL",    "metocean Tab.6-2, U=10 m/s"),
    L_kaimal= P(340.2,    "m",      "IEC",     "integral length scale, z>60 m"),
    TI      = P(0.14,     "-",      "IEC",     "I_ref, class B (Tab.1). NOT the actual TI: "
                                                "NTM gives sigma_1 = I_ref(0.75 V + 5.6)"),
    z_hub   = P(150.0,    "m",      "NREL",    "hub height"),
)

# ==========================================================================
#  PAYLOAD -- IEA 15 MW nacelle assembly (incl. hub)
#  NREL/TP-5000-75698 Table 5-1, row "Nacelle total".
#  I_zz is about the VERTICAL AXIS THROUGH THE NACELLE CG: verified by checking
#  sum(component I_zz) + sum(m_i * dx_i^2 about x_CG=5.486) == 1.868e7.  <- see tests
# ==========================================================================
PAYLOAD = dict(
    m_p   = P(820_888.0,   "kg",     "NREL", "Tab.5-1 'Nacelle total'"),
    I_xx  = P(1.2607277e7, "kg m^2", "NREL", "Tab.5-1"),
    I_yy  = P(2.1433958e7, "kg m^2", "NREL", "Tab.5-1"),
    I_zz  = P(1.8682468e7, "kg m^2", "NREL", "Tab.5-1"),
    # bounding box, reconstructed from Tab.5-1/5-2/5-3 component geometry
    L_n   = P(20.0, "m", "NREL",    "hub overhang 11.35 + bedplate ~5 + hub radius ~4"),
    W_n   = P(10.0, "m", "NREL",    "generator air-gap radius 5.08 -> dia 10.2"),
    H_n   = P(10.0, "m", "NREL",    "as above"),
    D_yaw = P(6.5,  "m", "NREL",    "Tab.5-3 tower-top / yaw bearing diameter"),
    psi_tol = P(1.0, "deg", "ASSUMED", "yaw-bearing bolt-circle mating tolerance"),
)

# ==========================================================================
#  HOISTING ROPE + REEVING
# ==========================================================================
ROPE = dict(
    d        = P(0.120,   "m",   "ASSUMED", "sized: 720 kN/fall @ n=12, SF~5 -> MBL 3.6 MN"),
    C_torque = P(0.0076,  "-",   "BRIDON",  "Dyform 34LR torque factor 0.76% of rope dia"),
    fill     = P(0.62,    "-",   "BRIDON",  "worked example: 0.62 in^2 metallic / 0.994 in^2 gross"),
    E_eff    = P(93.1e9,  "Pa",  "BRIDON",  "worked example: 13.5e6 psi"),
    rho_s    = P(7850.0,  "kg/m^3", "NREL", "Tab.4-1 steel density"),
    G_s      = P(79.3e9,  "Pa",  "NREL",    "Tab.4-1 steel shear modulus"),
    n_falls  = P(12,      "-",   "ASSUMED", "typical heavy-lift reeving"),
    D_over_d = P(20.0,    "-",   "ASSUMED", "sheave/rope dia ratio, offshore practice 18-25"),
    L0       = P(100.0,   "m",   "ASSUMED", "nominal deployed suspension length"),
    psi_cabling_limit = P(30.0, "deg", "BRIDON",
                          "crane design limit on hook-block angular displacement"),
)

HOOK = dict(
    m_hb = P(60_000.0, "kg", "ASSUMED", "hook block for a >800 t lift; sensitivity-tested"),
)

CRANE = dict(
    L_boom  = P(100.0, "m", "ASSUMED", "boom radius at the lift point"),
)

# ==========================================================================
#  CRANE ACTUATORS  --  every actuator is rate- AND magnitude-limited
#  Heavy-lift offshore cranes move SLOWLY under an 800+ t load.  The saturation is
#  not a detail: R5 shows the yaw authority is already marginal, and a rate-limited
#  slew makes it more so.  All limits ASSUMED from offshore heavy-lift practice and
#  sensitivity-tested; none is a number chosen to make a result come out.
# ==========================================================================
CRANE_ACT = dict(
    slew_rate_max  = P(0.30,  "deg/s", "ASSUMED", "max slew rate under heavy load"),
    luff_rate_max  = P(0.20,  "deg/s", "ASSUMED", "max luff (boom-elevation) rate"),
    hoist_rate_max = P(0.25,  "m/s",   "ASSUMED", "max hoist line speed under load"),
    slew_range     = P((-180.0, 180.0), "deg", "ASSUMED", "slew travel"),
    luff_range     = P((30.0, 85.0),    "deg", "ASSUMED", "boom-elevation travel"),
    hoist_range    = P((5.0, 200.0),    "m",   "ASSUMED", "deployable suspension length"),
    tau_act        = P(1.0, "s", "ASSUMED", "actuator first-order lag (servo bandwidth)"),
)

# ==========================================================================
#  TUGGER WINCHES  --  the yaw authority R5 proves is a NECESSITY, not an option
#  A tugger line runs from a deck anchor to a point r_t off the payload yaw axis.
#  It delivers a yaw moment reacting against the DECK, so -- unlike the crane slew --
#  its moment is independent of the payload's horizontal position (R5).  A line PULLS
#  only (T >= 0), is tension- and rate-limited, and 2-4 lines form a controllable couple.
# ==========================================================================
TUGGER = dict(
    n_lines   = P(4,     "-",  "ASSUMED", "number of tugger lines (2-4 typical)"),
    r_att     = P(5.0,   "m",  "ASSUMED", "attachment radius on the payload from yaw axis"),
    T_max     = P(100e3, "N",  "ASSUMED", "max line tension per winch"),
    rate_max  = P(0.50,  "m/s","ASSUMED", "max winch payout/haul-in rate"),
    anchor_R  = P(25.0,  "m",  "ASSUMED", "deck anchor radius from below the lift point"),
    anchor_drop = P(30.0,"m",  "ASSUMED", "vertical drop from payload to deck anchors"),
)

# ==========================================================================
#  DAMPING
#  The yaw mode sits at f_yaw ~ 0.013 Hz, inside the Kaimal band, and has NO
#  aerodynamic damping at its operating point (crossflow damping ~ |sin beta| -> 0
#  at beta = 0; see aero.crossflow_damping_*).  Its resonant RMS response therefore
#  scales as 1/sqrt(zeta), so whatever damps it sets EVERY RMS number in the paper.
#  That damping is STRUCTURAL (rope hysteresis, sheave friction, block bearings) and
#  is NOT a measured quantity in the literature.  It is carried here as an EXPLICIT,
#  FLAGGED, SWEPT parameter -- never a silently chosen value; the damping is
#  reported as a sensitivity axis.
# ==========================================================================
DAMPING = dict(
    zeta_struct = P(0.01, "-", "ASSUMED",
                    "structural yaw damping ratio of the suspension. NOT MEASURED. "
                    "Plausible range 0.002-0.05 (steel-rope hysteresis + sheave "
                    "friction); RMS yaw ~ 1/sqrt(zeta) => a 4.5x spread. MUST be "
                    "reported as a sweep, never as this single nominal value."),
    zeta_range  = P((0.002, 0.05), "-", "ASSUMED",
                    "sweep bracket for zeta_struct; the operability map is a function "
                    "of this, not a point"),
)

# ==========================================================================
#  CONTROL  --  tugger yaw loop gains (Phase 6.1 / Phase 7)
#  STATED design gains, not tuned to a target: the sub-1 deg hold and the raised
#  divergence speed are OUTPUTS of these fixed numbers (CTL-TUG-04/06). Centralised
#  here so the campaign and the controller read one source of truth. k_p is a
#  STIFFNESS gain -- the only kind that moves the static divergence threshold (R4).
# ==========================================================================
CONTROL = dict(
    k_p     = P(2.0e6, "N*m/rad",   "ASSUMED",
                "proportional (stiffness) yaw gain; stated, not fitted. Lifts U_div "
                "10.4 -> ~42 m/s while unsaturated (Phase 6.1)."),
    k_d     = P(6.0e6, "N*m*s/rad", "ASSUMED",
                "derivative (damping) yaw gain; shapes the transient only."),
    psi_ref = P(0.0,   "rad",       "ASSUMED",
                "commanded yaw: align the nacelle with the deck reference."),
)

# ==========================================================================
#  SA_RANGES  --  input brackets for the Phase 8 Sobol sensitivity study
#  Uniform ranges over ONLY the uncertain / ASSUMED inputs.  Fixed sourced
#  quantities (payload mass, g, ...) are NOT sampled -- they are not uncertain.
#  Each bracket is (lo, hi) with a source string, like every other parameter.
# ==========================================================================
SA_RANGES = dict(
    C_torque    = P((0.0065, 0.0090), "-",  "BRIDON",
                    "Dyform 34LR torque factor 0.76% +/- ~15% (rope-to-rope / wear spread "
                    "around the Bridon nominal 0.0076)."),
    D_over_d    = P((18.0, 24.0),     "-",  "ASSUMED",
                    "bottom-block sheave-to-rope diameter ratio (sets spacing S=D/d*d); "
                    "practical reeving range around the nominal 20."),
    L           = P((60.0, 140.0),    "m",  "ASSUMED",
                    "suspension length over a nacelle-lift operating envelope."),
    L_n         = P((16.0, 26.0),     "m",  "ASSUMED",
                    "nacelle bounding length (sets aspect a/b and box volume); envelope "
                    "around the assumed 20 m gabarit, NREL Tab.5-1 gives no crate size."),
    rho_air     = P((1.20, 1.29),     "kg/m^3", "ASSUMED",
                    "air density over the metocean temperature/pressure window."),
    zeta_struct = P((0.002, 0.05),    "-",  "ASSUMED",
                    "structural yaw damping bracket (== DAMPING['zeta_range']); a NEGATIVE "
                    "control here -- it enters no STATIC output (R4/R6)."),
    k_p         = P((1.0e6, 4.0e6),   "N*m/rad", "ASSUMED",
                    "tugger proportional gain design range around params.CONTROL['k_p']."),
)

# ==========================================================================
#  VESSEL  --  installation barge (own published hull; BEM via Capytaine)
# ==========================================================================
VESSEL = dict(
    L      = P(80.0,  "m",  "ASSUMED", "barge length; typical heavy-lift installation vessel"),
    B      = P(40.0,  "m",  "ASSUMED", "barge beam"),
    T_draft= P(4.0,   "m",  "ASSUMED", "draft; displacement ~ 12800 t at rho_w"),
    D_hull = P(8.0,   "m",  "ASSUMED", "hull depth"),
    KG     = P(6.0,   "m",  "ASSUMED", "vertical CG above keel; sensitivity-tested"),
    boom_reach  = P(40.0, "m", "ASSUMED", "horizontal reach of the lift point from vessel CG"),
    boom_height = P(90.0, "m", "ASSUMED", "boom-tip height above the vessel CG"),
)

# ==========================================================================
#  WAVES  --  JONSWAP sea state (DNV-RP-C205)
# ==========================================================================
WAVES = dict(
    gamma_j = P(3.3,  "-", "DNVC205", "JONSWAP peak-enhancement, mean value (sec.3.5.5)"),
    grav    = P(9.80665, "m/s^2", "NREL", "same g as ENV, named for the dispersion reln"),
    depth   = P(200.0, "m", "ASSUMED", "water depth; deep-water dispersion for T_p<12 s"),
    beta_w  = P(0.0,  "rad", "ASSUMED", "wave heading rel. to vessel x (head seas)"),
    # sigma_a/sigma_b are fixed constants of the JONSWAP shape, not free params
    sigma_a = P(0.07, "-", "DNVC205", "JONSWAP spectral width, omega<=omega_p"),
    sigma_b = P(0.09, "-", "DNVC205", "JONSWAP spectral width, omega>omega_p"),
)


# ==========================================================================
#  WIND / TURBULENCE  --  IEC 61400-1 Ed.4 Kaimal model
# ==========================================================================
WIND = dict(
    Lambda_1 = P(42.0,  "m", "IEC", "sec.6.3.1.2: Lambda_1 = 0.7 z (z<60), 42 m (z>=60)"),
    cL1      = P(8.1,   "-", "IEC", "Tab.B.1: L_1 = 8.1 Lambda_1 = 340.2 m"),
    cL2      = P(2.7,   "-", "IEC", "Tab.B.1: L_2 = 2.7 Lambda_1"),
    cL3      = P(0.66,  "-", "IEC", "Tab.B.1: L_3 = 0.66 Lambda_1"),
    r_sig2   = P(0.8,   "-", "IEC", "sec.6.3: sigma_2 = 0.8 sigma_1"),
    r_sig3   = P(0.5,   "-", "IEC", "sec.6.3: sigma_3 = 0.5 sigma_1"),
    b_NTM    = P(5.6,   "m/s", "IEC", "Eq.(11): sigma_1 = I_ref (0.75 V_hub + b)"),
    a_coh    = P(12.0,  "-", "IEC", "Eq.(D.1) coherence decay; == Davenport C"),
    b_coh    = P(0.12,  "-", "IEC", "Eq.(D.1) second (length-scale) coherence term"),
    alpha_shear = P(0.14, "-", "IEC3", "sec.6.4.2: offshore power-law exponent"),
    z_ref    = P(10.0,  "m", "ASSUMED", "reference height of U10"),
    coh_vw   = P(True,  "-", "ASSUMED",
                 "apply the u-coherence model to v,w as well.  IEC specifies coherence "
                 "for u only (TurbSim IECKAI leaves v,w spatially incoherent), which is "
                 "unphysical for a 20 m body: two points 20 m apart cannot have "
                 "independent lateral gust. Sensitivity-tested; set False to recover IEC."),
)

# ==========================================================================
#  DERIVED QUANTITIES
# ==========================================================================
@dataclass
class Derived:
    A_met:   float   # metallic area of ONE fall            [m^2]
    mu_1:    float   # mass/length of ONE fall              [kg/m]
    mu_tot:  float   # mass/length of ALL falls in parallel [kg/m]
    S:       float   # rope spacing = bottom-block sheave dia [m]
    W_load:  float   # payload + hook block                 [N]   <- drives rope torque
    W_rope:  float   # self-weight of all falls             [N]
    W_eff:   float   # W_load + W_rope/2                    [N]   <- drives k_psi
    eps:     float   # rope mass / payload mass  (small parameter)
    EA_tot:  float   # sum of E*A over all falls            [N]
    k_psi:   float   # torsional stiffness of suspension    [N m/rad]
    Tv:      float   # Bridon torque value = C*d            [m]
    k2mk1:   float   # Lamb (k2 - k1)                       [-]
    V_box:   float   # payload bounding volume              [m^3]
    V_ell:   float   # inscribed ellipsoid volume           [m^3]

    # NOTE on W_load vs W_eff.  Rotating the hook block by psi raises the block by
    # delta = (S/2)^2 psi^2 / (2L) and the rope CG by ~delta/2, hence
    #     k_psi = (W_load + W_rope/2) (S/2)^2 / L .
    # The rope torque delivered to the block, however, is set by the tension AT the
    # block, T_i = W_load / n, so  M = n C d (W_load/n) = C d W_load  -- n cancels
    # and the rope self-weight does NOT enter.  Unit tests enforce both.


# Lamb apparent-mass factors, prolate spheroid (Lamb 1932 §114 / Newman)
_LAMB_AR = np.array([1.0, 1.50, 2.00, 2.51, 2.99, 4.00, 5.00, 6.00, 7.00, 8.00, 10.0])
_LAMB_K1 = np.array([.500, .305, .209, .156, .122, .082, .059, .045, .036, .029, .021])
_LAMB_K2 = np.array([.500, .621, .702, .763, .803, .860, .895, .918, .933, .945, .960])


def lamb_k2_minus_k1(fineness: float) -> float:
    """(k2 - k1) for a prolate spheroid of given a/b.  Lamb (1932) Table."""
    if fineness < 1.0:
        raise ValueError("fineness a/b must be >= 1")
    return float(np.interp(fineness, _LAMB_AR, _LAMB_K2)
                 - np.interp(fineness, _LAMB_AR, _LAMB_K1))


def derive(L: float = None, n: float = None, DoD: float = None) -> Derived:
    """Compute derived quantities.  L, n, D/d may be overridden for sweeps."""
    d   = float(ROPE["d"]);      fill = float(ROPE["fill"])
    E   = float(ROPE["E_eff"]);  rho_s = float(ROPE["rho_s"])
    C   = float(ROPE["C_torque"])
    n   = float(ROPE["n_falls"])   if n   is None else float(n)
    DoD = float(ROPE["D_over_d"])  if DoD is None else float(DoD)
    L   = float(ROPE["L0"])        if L   is None else float(L)

    g       = float(ENV["g"])
    A_gross = np.pi * d**2 / 4.0
    A_met   = fill * A_gross
    mu_1    = rho_s * A_met / 0.85          # +core/lube: metallic mass / 0.85
    mu_tot  = n * mu_1
    S       = DoD * d                        # Bridon: spacing = bottom-block sheave dia

    m_load  = float(PAYLOAD["m_p"]) + float(HOOK["m_hb"])
    W_load  = m_load * g                     # tension at the block  -> rope torque
    W_rope  = mu_tot * L * g                 # self-weight of all falls
    W_eff   = W_load + 0.5 * W_rope          # raising the block also raises the rope CG

    Ln, Wn, Hn = (float(PAYLOAD[k]) for k in ("L_n", "W_n", "H_n"))

    return Derived(
        A_met  = A_met,
        mu_1   = mu_1,
        mu_tot = mu_tot,
        S      = S,
        W_load = W_load,
        W_rope = W_rope,
        W_eff  = W_eff,
        eps    = mu_tot * L / float(PAYLOAD["m_p"]),
        EA_tot = n * E * A_met,
        k_psi  = W_eff * (S / 2.0) ** 2 / L,   # multifilar pendulum  (== Bridon)
        Tv     = C * d,
        k2mk1  = lamb_k2_minus_k1(Ln / Wn),
        V_box  = Ln * Wn * Hn,
        V_ell  = 4.0 / 3.0 * np.pi * (Ln / 2) * (Wn / 2) * (Hn / 2),
    )


def validate() -> None:
    """Every parameter must have a recognised source."""
    for grp in (ENV, PAYLOAD, ROPE, HOOK, CRANE, WIND, DAMPING, CONTROL, SA_RANGES,
                WAVES, VESSEL, CRANE_ACT, TUGGER):
        for k, p in grp.items():
            assert isinstance(p, P), f"{k} is not a sourced parameter"
            assert p.source in SOURCES, f"{k}: bad source"
    assert float(ENV["g"]) > 0


def report() -> str:
    lines = ["PARAMETER SET (every value sourced)", "=" * 78]
    for name, grp in (("ENVIRONMENT", ENV), ("PAYLOAD", PAYLOAD),
                      ("ROPE/REEVING", ROPE), ("HOOK", HOOK), ("CRANE", CRANE),
                      ("WIND", WIND), ("DAMPING", DAMPING), ("CONTROL", CONTROL),
                      ("SA_RANGES", SA_RANGES), ("WAVES", WAVES),
                      ("VESSEL", VESSEL), ("CRANE_ACT", CRANE_ACT), ("TUGGER", TUGGER)):
        lines.append(f"\n[{name}]")
        for k, p in grp.items():
            lines.append(f"  {k:<10} = {str(p.value):<14} {p.unit:<8} <{p.source}> {p.note}")
    return "\n".join(lines)
