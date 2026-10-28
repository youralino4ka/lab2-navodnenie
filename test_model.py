"""Regression tests for the transcribed source equations."""
import numpy as np
from model import f, rhs, scaled_rhs


def test_all_66_auxiliary_functions_are_defined_and_finite():
    z = np.ones(14)
    values = [f(i, z) for i in range(1, 67)]
    assert len(values) == 66
    assert np.isfinite(values).all()


def test_rhs_shape_and_finite_for_positive_state():
    z = np.linspace(0.8, 1.4, 14)
    p = {k: v for k, v in zip(
        ('Rw', 'Hw', 'Sem', 'Vhr', 'Chr', 'Sev', 'Ch', 'Hr', 'Vh', 'Lr'),
        (1.1, 0.9, 1.2, 1.3, 0.8, 0.7, 1.4, 1.1, 0.6, 0.95))}
    d = rhs(0.0, z, p)
    assert d.shape == (14,)
    assert np.isfinite(d).all()


def test_source_external_symbols_in_z1_z6_z7_z10_equations():
    # Distinct driver values catch visual confusions between Sem/Sev, Chr/Ch, and Vhr/Vh.
    z = np.linspace(0.8, 1.4, 14)
    p = {'Rw': 1.1, 'Hw': 0.9, 'Sem': 1.2, 'Vhr': 1.3, 'Chr': 0.8,
         'Sev': 0.7, 'Ch': 1.4, 'Hr': 1.1, 'Vh': 0.6, 'Lr': 0.95}
    d = rhs(0.0, z, p)
    F = lambda i: f(i, z)
    expected_z1 = (F(1)*F(2)*p['Rw'] - F(3)*p['Vhr']*p['Chr']*p['Sev']) / z[0]
    expected_z6 = (F(22)*F(23)*F(24)*F(25)*p['Sem'] - F(26)*F(27)*F(28)*p['Chr']*p['Hr']*p['Vhr']) / z[5]
    expected_z7 = (F(29)*p['Chr']*p['Hr']*p['Vhr'] - p['Rw']*p['Sev']*p['Lr']) / z[6]
    expected_z10 = (F(39)*F(40)*F(41)*F(42)*F(43)*p['Rw'] - p['Chr']*p['Hr']*p['Vhr']) / z[9]
    assert np.isclose(d[0], expected_z1)
    assert np.isclose(d[5], expected_z6)
    assert np.isclose(d[6], expected_z7)
    assert np.isclose(d[9], expected_z10)


def test_scenario_scaling_is_separate_from_source_equations():
    z = np.ones(14)
    p = {k: 1.0 for k in ('Rw', 'Hw', 'Sem', 'Vhr', 'Chr', 'Sev', 'Ch', 'Hr', 'Vh', 'Lr')}
    assert np.allclose(scaled_rhs(0.0, z, p, 1e-12), 1e-12 * rhs(0.0, z, p))
