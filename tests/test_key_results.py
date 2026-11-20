"""Verification tests of the key results of the paper (run with: pytest)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import common as cm                                   # noqa: E402
from aero_models import Munk, BluffBox, MeasuredBox, U_div, max_slope   # noqa: E402
from s1_suspension import BlockEquilibrium, moment_closed_form   # noqa: E402
from s3_statics import equilibria                     # noqa: E402
from s6_control import eig_threshold                  # noqa: E402

RES = os.path.join(ROOT, "results")


# ---------------------------------------------------------------- Section 3
@pytest.mark.parametrize("eta, tol", [(1.0, 1e-3), (0.96, 5e-3)])
def test_restoring_moment_closed_form_vs_3d(eta, tol):
    a = cm.fall_positions()
    W = (cm.m_p + cm.m_hb) * cm.g
    eq = BlockEquilibrium(a, a, W, cm.L_mating, cm.Z_CG_BELOW_BLOCK, eta)
    q = None
    for psi in np.radians([1.0, 10.0, 45.0, 90.0, 135.0]):
        q, Mz, res = eq.solve(psi, q)
        M_cf, _ = moment_closed_form(psi, a, a, eq.ratio, W, eq.ell_tot)
        assert res < 1e-9
        assert Mz == pytest.approx(M_cf, rel=tol)


def test_cabling_benchmark():
    C, d, S, L = 0.0076, 0.020, 0.360, 30.0
    assert 4 * C * d * L / S ** 2 == pytest.approx(0.1407, abs=1e-4)   # published: 0.141
    assert S ** 2 / (4 * C * d) == pytest.approx(213.2, abs=0.1)        # published: 213 m


def test_suspension_properties_at_mating():
    s = cm.suspension(cm.L_mating)
    assert cm.L_mating == pytest.approx(13.0)
    assert s.k_psi == pytest.approx(1.081e6, rel=2e-3)
    assert s.f_psi == pytest.approx(0.0383, abs=2e-4)
    assert np.degrees(s.psi0) == pytest.approx(0.26, abs=0.01)


# ---------------------------------------------------------------- Section 4
def test_lamb_coefficients_and_bem():
    from s2_aero_bem import lamb_prolate, mesh_superellipsoid, added_masses
    k1, k2 = lamb_prolate(2.0)
    assert (k1, k2) == pytest.approx((0.2100, 0.7042), abs=2e-4)
    m = mesh_superellipsoid(10.0, 5.0, 5.0, 2.0, (36, 54))
    m11, m22, _, _ = added_masses(m)
    V = np.pi / 6 * 20 * 10 * 10
    assert m11 / V == pytest.approx(k1, rel=0.08)
    assert m22 / V == pytest.approx(k2, rel=0.05)


def test_divergence_speeds_table3():
    k13 = cm.suspension(13.0).k_psi
    k150 = cm.suspension(150.0).k_psi
    assert U_div(k13, 0.496) == pytest.approx(29.8, abs=0.1)
    assert U_div(k150, 0.496) == pytest.approx(8.9, abs=0.1)
    assert U_div(k13, BluffBox(0.09, 0.04).CK) == pytest.approx(89.5, abs=0.2)
    assert BluffBox(0.09, 0.04, xs=1.0).CK < 0          # weathervane: no divergence


def test_measured_box_fit():
    import wind_tunnel_data as wt
    m = MeasuredBox()
    assert wt.fit()[2] < 0.007                          # rms residual of the fit
    assert m.CK == pytest.approx(-0.436, abs=0.005)     # wind along the axis: stabilising
    cmax, bmax = max_slope(m)
    assert cmax == pytest.approx(0.335, abs=0.003)
    assert np.degrees(bmax) == pytest.approx(18.4, abs=0.3)
    for K in (4, 5, 6, 7):                              # sign of C_K independent of the fit order
        assert MeasuredBox(K).CK < 0


def test_critical_speeds_sharp_edged_box():
    cmax = max_slope(MeasuredBox())[0]
    assert U_div(cm.suspension(13.0).k_psi, cmax) == pytest.approx(36.3, abs=0.1)
    assert U_div(cm.suspension(150.0).k_psi, cmax) == pytest.approx(10.8, abs=0.1)


def test_munk_post_critical_branch():
    s = cm.suspension(100.0)
    m = Munk(0.496)
    for u, psi_star in ((1.05, 24.90), (1.2, 46.02), (1.5, 63.61)):
        eq = [np.degrees(r) for r, st in equilibria(s.k_psi, 0.0, u * U_div(s.k_psi, m.CK), m)
              if st and r > 0]
        assert eq[0] == pytest.approx(psi_star, abs=0.01)
        assert eq[0] == pytest.approx(np.degrees(np.arccos(u ** -2)), abs=1e-6)


def test_supercriticality_criterion():
    assert Munk().K3_over_K() == pytest.approx(2 / 3)
    assert BluffBox(0.09, 0.04).K3_over_K() < 1 / 6


# ---------------------------------------------------------------- Section 5.3
def test_coupled_vessel_pendulum_limits():
    """Two-way coupling: at low frequency the load follows the boom tip (rigid limit);
    the horizontal load motion obeys the pendulum transfer function."""
    from s4_vessel import coupled_rao, pendulum, J_TIP
    rng = np.random.default_rng(1)
    om = np.array([1e-3, 0.3, 0.8])
    M = np.diag([1e8, 1e8, 1e8, 4e10, 5e11, 5e11])
    C = np.diag([0.0, 0.0, 3e7, 1.3e10, 6e11, 0.0])
    Kh = np.diag([1.8e5, 1.8e5, 0.0, 0.0, 0.0, 9e8])
    bem = dict(omega=om, M=M, A=np.zeros((3, 6, 6)), B=np.zeros((3, 6, 6)), Bext=np.diag([0, 0, 0, 1e9, 0, 0]),
               C=C, Kh=Kh, Fex=rng.normal(size=(3, 1, 6)) * 1e6 + 0j)
    rao2, tip2, xL2 = coupled_rao(bem, cm.L_mating)
    rao3, tip3, _ = coupled_rao(bem, cm.L_mating, rigid=True)
    assert np.allclose(rao2[0], rao3[0], rtol=1e-4, atol=1e-12)
    mL, Kp, cp = pendulum(cm.L_mating)
    for i, w in enumerate(om):
        G = (Kp - 1j * w * cp) / (Kp - 1j * w * cp - w ** 2 * mL)
        assert np.allclose(xL2[i, 0], G * tip2[i, 0, :2])
    assert np.allclose(tip2, np.einsum("ij,fhj->fhi", J_TIP, rao2))


# ---------------------------------------------------------------- Section 6
@pytest.mark.parametrize("kp", [0.0, 2e5, 2e6])
def test_closed_loop_threshold_independent_of_damping(kp):
    s = cm.suspension(cm.L_mating)
    closed_form = U_div(s.k_psi, 0.496) * np.sqrt(1 + kp / s.k_psi)
    for kd in (0.0, 6e6, 6e7):
        u_eig = eig_threshold(s.k_psi, cm.I_zz, 1e4, kp, kd, 0.496)
        assert u_eig == pytest.approx(closed_form, abs=0.1)


# ------------------------------------------- stored results vs the paper
def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not computed yet")
    with open(path) as fh:
        return json.load(fh)


def test_stored_results_match_paper():
    s2 = _load("s2_aero_bem.json")
    assert s2["CK"]["bem_box"] == pytest.approx(0.496, abs=0.002)
    s7 = _load("s7_verify_6dof.json")["turbulent"]
    assert s7["std_red"] == pytest.approx(s7["std_6dof"], rel=0.005)
    assert s7["corr"] > 0.9999
    s5 = _load("s5_dynamics.json")
    v = s5["verification"]
    assert v["td_sigma_deg"][0] == pytest.approx(v["spectral_sigma_deg"], abs=2 * v["td_sigma_deg"][1])
