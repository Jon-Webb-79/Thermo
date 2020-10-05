# Import necessary packages here
import numpy as np
# ============================================================================
# ============================================================================
# Date:    October 4, 2020
# Purpose: This file contains classes describing the thermodynamic
#          properties of gases

# Source Code Metadata
__author__ = "Jonathan A. Webb"
__copyright__ = "Copyright 2020, Jon Webb Inc."
__version__ = "1.0"

# ============================================================================
# ============================================================================


class Helmholtz:
    """
    This class determines the residuals Helmoltz energies that are
    necessary to calculate thermodynamic properties.  All equations
    are derived from the following references

    1. R. Span, E. W. Lemmon, R. T. Jacobson, W. Wagner, and A. Yokozeki,
    "A Reference Equation of State for the Thermodynamic Properties
    of Nitrogen for Temperatures from 63.151 to 1000 K and Pressures
    to 2200 MPa", J. Phy. Chem. Ref. Data 29, 1361 (2000)

    2. D. O. Vega, "A New Wide Range Equation of State for Helium-4",
    Ph.D. Dissertation, Texas A&M University, August 2013
    """
    def __init__(self, critical_density: float, critical_temperature: float,
                 tk: np.array, nk: np.array, dk: np.array, lk: np.array,
                 etak: np.array, epsk: np.array, betak: np.array,
                 gammak: np.array, upper1: int, upper2: int, upper3: int):
        """

        :param critical_density: The density associated with the critical point
                                 in units of moles per cubic decimeter
        :param critical_temperature: The temperature associated with the critical
                                     point in units of Kelvins
        :param tk: fitting coefficients
        :param nk: fitting coefficeints
        :param dk: fitting coefficeints
        :param lk: fitting coefficients
        :param etak: fitting coefficients
        :param epsk: fitting coefficients
        :param betak: fitting coefficients
        :param gammak: fitting coefficients
        :param upper1: The limits of the first summation
        :param upper2: The limits of the second summation
        :param upper3: The limits of the third summation
        """
        self.critical_temperature = critical_temperature
        self.critical_density = critical_density
        self.tk = tk
        self.nk = nk
        self.dk = dk
        self.lk = lk
        self.etak = etak
        self.epsk = epsk
        self.betak = betak
        self.gammak = gammak
        self.upper1 = upper1
        self.upper2 = upper2
        self.upper3 = upper3
# ----------------------------------------------------------------------------

    def first_alpha_delta_partial(self, density: float, temperature: float):
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The static density in units of Kelvins
        :return partial: The partial derivative of alpha with
                         respect to delta.

        This function solves for the residual Helmholtz energy resulting
        from the first partial of alpha with respect to delta as shown on
        page 1386 or Ref. 1
        """
        delta = density / self.critical_density
        tau = self.critical_temperature / temperature

        sum1 = (self.nk[:self.upper1] * delta ** self.dk[:self.upper1] *
                tau ** self.tk[:self.upper1] * self.dk[:self.upper1]).sum()

        sum2 = (self.nk[self.upper1:self.upper2] *
                delta ** self.dk[self.upper1:self.upper2] *
                tau ** self.tk[self.upper1:self.upper2] *
                np.exp(-delta ** self.lk[self.upper1:self.upper2]) *
                (self.dk[self.upper1:self.upper2] -
                self.lk[self.upper1:self.upper2] *
                delta ** self.lk[self.upper1:self.upper2])).sum()

        sum3 = (self.nk[self.upper2:self.upper3] * delta ** self.dk[self.upper2:self.upper3] *
                tau ** self.tk[self.upper2:self.upper3] *
                np.exp(-self.etak[self.upper2:self.upper3] *
                (delta - self.epsk[self.upper2:self.upper3]) ** 2.0 -
                self.betak[self.upper2:self.upper3] *
                (tau - self.gammak[self.upper2:self.upper3]) ** 2.0) *
                (self.dk[self.upper2:self.upper3] - 2.0 *
                self.etak[self.upper2:self.upper3] * delta *
                (delta - self.epsk[self.upper2:self.upper3]))).sum()
        return sum1 + sum2 + sum3
# ============================================================================
# ============================================================================
# eof
