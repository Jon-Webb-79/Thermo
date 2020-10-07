# Import modules here
from thermo.thermo import Helmholtz, ThermoProps

from math import isclose
import numpy as np
# ==============================================================================
# ==============================================================================
# Date:    September 17, 2020
# Purpose: This code contains functions that test the functions and classes
#          in the isentropic_models.py file

# Source Code Metadata
__author__ = "Jonathan A. Webb"
__copyright__ = "Copyright 2020, Jon Webb Inc."
__version__ = "1.0"
# ==============================================================================
# ==============================================================================


critical_temp = 126.192
critical_dens = 11.1839
nk = np.array([0.924803575, -0.492448489, 0.661883337, -1.929026492,
               -0.062246931, 0.349943958, 0.564857472, -1.61720006,
               -0.481395032, 0.421150636, -0.016196223, 0.172100994,
               0.007354489, 0.016807731, -0.001076267, -0.013731809,
               0.000635467, 0.003044323, -0.043576234, -0.072317489,
               0.038964432, -0.021220136, 0.00408823, -5.5199E-05,
               -0.046201672, -0.003003117, 0.036882589, -0.002558568,
               0.008969153, -0.004415134, 0.001337229, 0.000264832,
               19.6688194, -20.91156007, 0.016778831, 2627.675663])
tk = np.array([0.25, 0.875, 0.5, 0.875, 0.375, 0.75, 0.5, 0.75, 2.0,
               1.25, 3.5, 1.0, 0.5, 3.0, 0.0, 2.75, 0.75, 2.5, 4.0,
               6.0, 6.0, 3.0, 3.0, 6.0, 16.0, 11.0, 15.0, 12.0, 12.0,
               7.0, 4.0, 16.0, 0.0, 1.0, 2.0, 3.0])
dk = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 3.0, 3.0,
               4.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 1.0, 2.0, 3.0, 4.0,
               5.0, 8.0, 4.0, 5.0, 5.0, 8.0, 3.0, 5.0, 6.0, 9.0, 1.0,
               1.0, 3.0, 2.0])
lk = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
               2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0,
               2.0, 2.0, 2.0])
etak = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 20.0, 20.0, 15.0, 25.0])
betak = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 325.0, 325.0, 300.0, 275.0])
gammak = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.16, 1.16, 1.13, 1.25])
epsk = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

ideal_coefs = {'a1': 2.5, 'a2': -12.76952708, 'a3': -0.00784163,
               'a4': -0.0001934819, 'a5': -0.00001247742,
               'a6': 0.00000006678326, 'a7': 1.012941,
               'a8': 26.65788}
temperature = 340.8
density = 2.7843
helm = Helmholtz(critical_dens, critical_temp, tk, nk, dk, lk,
                 etak, epsk, betak, gammak, 6, 32, 36, ideal_coefs)


def test_helmholtz_energy():
    """

    This function tests the helmholtz energy function
    """
    energy = helm.helmholtz_energy(density, temperature)
    assert isclose(energy, -15.634, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_first_alpha_delta_partial():
    """

    Test the first_alpha_delta_partial() function to the criteria
    on page 1423
    """
    alpha_delta = helm.first_alpha_delta_partial(density, temperature)
    assert isclose(alpha_delta, 0.01669, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_first_alpha_tau_one_partial():
    """

    This function tests the first_alpha_zero_tau_partial() function
    """
    alpha_tau = helm.first_alpha_tau_zero_partial(temperature)
    assert isclose(alpha_tau, -7.50035, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_first_alpha_tau_two_partial():
    """

    This function tests the first_alpha_one_tau_partial() function
    """
    alpha_tau = helm.first_alpha_tau_one_partial(density, temperature)
    assert isclose(alpha_tau, -0.13262, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_second_alpha_tau_zero_partial():
    """

    This function tests the second_alpha_tau_zero_partial() function
    """
    alpha_tau = helm.second_alpha_tau_zero_partial(temperature)
    assert isclose(alpha_tau, -96.197, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_second_alpha_tau_one_partial():
    """

    This function tests the second_alpha_tau_one_partial() function
    """
    alpha_tau = helm.second_alpha_tau_one_partial(density, temperature)
    assert isclose(-26.147, alpha_tau, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_second_alpha_delta_partial():
    """

    This function tests the second_alpha_delta_partial() function
    """
    alpha_delta = helm.second_alpha_delta_partial(density, temperature)
    assert isclose(alpha_delta, 0.01116, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_second_alpha_delta_tau_partial():
    """

    This function tests the second_alpha_tau_delta_partial() function
    """
    alpha_delta_tau = helm.second_alpha_tau_delta_partial(density, temperature)
    assert isclose(alpha_delta_tau, -0.13103, rel_tol=1.0e-3)
# ==============================================================================
# ==============================================================================
# Test ThermoProps class


therm = ThermoProps(critical_dens, critical_temp, tk, nk,
                    dk, lk, etak, epsk, betak, gammak, 6,
                    32, 36, ideal_coefs, 28.01348)


def test_pressure():
    """

    This function tests the pressure() function against diatomic
    nitrogen inputs
    """
    dens = 0.07796151  # corresponds to 2.783 moles/dm3 as used in Ref. 2.
    pressure = therm.pressure(dens, temperature)
    assert isclose(pressure, 8017385.0, rel_tol=1.0e-3)
# ------------------------------------------------------------------------------


def test_temperature():
    """

    This function tests the temperature() function
    """
    dens = 0.07796151  # corresponds to 2.783 moles/dm3 as used in Ref. 2.
    press = therm.temperature(dens, 8017385.0)
    assert isclose(press, 340.7, rel_tol=1.0e-3)
# ==============================================================================
# ==============================================================================
# eof
