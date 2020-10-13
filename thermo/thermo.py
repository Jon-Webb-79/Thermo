# Import necessary packages here
import numpy as np
from typing import Dict
import sys
from scipy.optimize import bisect
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
                 gammak: np.array, upper1: int, upper2: int, upper3: int,
                 helm_coefs: Dict[str, float]):
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
        :param helm_coefs: The Helmholtz coefficients as a
                           dictionary with the following keys,
                           `a1`, `a1`, `a3`, `a4`, `a5`, `a6`,
                           `a7`, and `a8`
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
        self.helm_coefs = helm_coefs
# ----------------------------------------------------------------------------

    def helmholtz_energy(self, density: float, temperature: float) -> float:
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The static temperature in units of Kelvins
        :return energy: The ideal Helmholtz energy

        This function determines the ideal Helmholtz energy in accordance
        with Eq. 53 from Ref. 1. shown below

        .. math::
           \\alpha^o = \ln{\delta} + a_1 + \ln{\\tau} + a_2 + a_3\\tau +
           a_4 \\tau^{-1} + a_5\\tau^{-2} + a_6\\tau^{-3} +
           a_7 \\ln{\\left[1 - exp\\left(-a_8\\tau \\right) \\right]}
        """
        delta = density / self.critical_density
        tau = self.critical_temperature / temperature
        energy = np.log(delta) + self.helm_coefs['a1'] * np.log(tau) + \
        self.helm_coefs['a2'] + self.helm_coefs['a3'] * tau + \
        (self.helm_coefs['a4'] / tau) + (self.helm_coefs['a5'] / tau ** 2.0) + \
        (self.helm_coefs['a6'] / tau ** 3.0) + self.helm_coefs['a7'] * \
        (1.0 - np.exp(-self.helm_coefs['a8'] * tau))
        return energy
# ----------------------------------------------------------------------------

    def first_alpha_delta_partial(self, density: float, temperature: float) -> float:
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The static temperature in units of Kelvins
        :return partial: The partial derivative of alpha with
                         respect to delta.

        This function solves for the residual Helmholtz energy resulting
        from the first partial of alpha with respect to delta as shown on
        below and page 1386 or Ref. 1

        .. math::
           \\delta\\left(\\frac{\partial\\alpha^r}{\partial\\delta} \\right)_{\\tau}
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
# ----------------------------------------------------------------------------

    def first_alpha_tau_zero_partial(self, temperature: float) -> float:
        """

        :param temperature: The critical temperature in units of Kelvins
        :return derivative: The first partial derivative of alpha zero
                            with respect to tau

        This function determines the first partial derivative of
        alpha zero with respect to tau in accordance with Eq. 79
        of Ref. 1. as shown below

        .. math::
           \\tau\\left(\\frac{\partial\\alpha^o}{\partial\\tau} \\right)_{\\delta}
        """
        tau = self.critical_temperature / temperature

        der = self.helm_coefs['a1'] + self.helm_coefs['a3'] * tau - \
            self.helm_coefs['a4'] * tau ** -1.0 - \
            2.0 * self.helm_coefs['a5'] * tau ** -2.0 - \
            3.0 * self.helm_coefs['a6'] * tau ** -3.0 + \
            (self.helm_coefs['a7'] * self.helm_coefs['a8'] *
             tau * (1/ (np.exp(self.helm_coefs['a8'] * tau) - 1.0)))
        return der
# ----------------------------------------------------------------------------

    def first_alpha_tau_one_partial(self, density: float, temperature: float) -> float:
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The temperature in units of Kelvins
        :return derivative: The first derivative of alpha one with respect
                            to tau

        This function determines the first derivative of alpha one with
        respect to tau as outlined in Eq. 84 or Ref. 1. as shown below


        .. math::
           \\tau\\left(\\frac{\partial\\alpha^r}{\partial\\tau} \\right)_{\\delta}
        """
        delta = density / self.critical_density
        tau = self.critical_temperature / temperature

        sum1 = (self.nk[:self.upper1] * delta ** self.dk[:self.upper1] *
                tau ** self.tk[:self.upper1] * self.tk[:self.upper1]).sum()

        sum2 = (self.nk[self.upper1:self.upper2] *
                delta ** self.dk[self.upper1:self.upper2] *
                tau ** self.tk[self.upper1:self.upper2] *
                np.exp(-delta ** self.lk[self.upper1:self.upper2]) *
                self.tk[self.upper1:self.upper2]).sum()

        sum3 = (self.nk[self.upper2:self.upper3] * delta ** self.dk[self.upper2:self.upper3] *
                tau ** self.tk[self.upper2:self.upper3] * self.tk[self.upper2:self.upper3] *
                np.exp(-self.etak[self.upper2:self.upper3] * (delta - self.epsk[self.upper2:self.upper3]) ** 2.0 -
                       self.betak[self.upper2:self.upper3] * (tau - self.gammak[self.upper2:self.upper3]) ** 2.0)
                *(self.tk[self.upper2:self.upper3] - 2.0 * self.betak[self.upper2:self.upper3] * tau *
                  (tau - self.gammak[self.upper2:self.upper3]))).sum()
        return sum1 + sum2 + sum3
# ----------------------------------------------------------------------------

    def second_alpha_tau_zero_partial(self, temperature: float) -> float:
        """

        :param temperature: The static temperature in units of Kelvins
        :return derivative: The second partial derivative of alpha zero
                            with respect to tau

        This function determines the seconc partial derivative of alpha
        zero with respect to tau in accordance with Eq. 80 of Ref. 1.
        as shown below

        .. math::
           \\tau^2\\left(\\frac{\partial^2\\alpha^o}{\partial\\tau^2} \\right)_{\\delta}
        """
        tau = self.critical_temperature / temperature
        derivative = self.helm_coefs['a1'] + 2.0 * (self.helm_coefs['a4'] / tau) + \
                     6.0 * (self.helm_coefs['a5'] / tau ** 2.0) + \
                     12.0 * (self.helm_coefs['a6'] / tau ** 3.0) - \
                     self.helm_coefs['a7'] * self.helm_coefs['a8'] ** 2.0 * \
                     tau ** 2.0 * (np.exp(self.helm_coefs['a8'] *
                     tau / (np.exp(self.helm_coefs['a8'] * tau) - 1.0) ** 2.0))
        return derivative
# ----------------------------------------------------------------------------

    def second_alpha_tau_one_partial(self, density: float, temperature: float) -> float:
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The temperature in units of Kelvins
        :return derivative: The second partial of alpha one with respect
                            to tau

        This function determines the value of the second derivative of alpha
        one with respect to tau according to Eq. 85 of Ref. 1. as shown
        below

        .. math::
           \\tau^2\\left(\\frac{\partial^2\\alpha^r}{\partial\\tau^2} \\right)_{\\delta}
        """
        delta = density / self.critical_density
        tau = self.critical_temperature / temperature
        sum1 = (self.nk[:self.upper1] * delta ** self.dk[:self.upper1] *
               tau ** self.tk[:self.upper1] * (self.tk[:self.upper1] *
               (self.tk[:self.upper1] - 1.0))).sum()

        sum2 = (self.nk[self.upper1:self.upper2] *
                delta ** self.dk[self.upper1:self.upper2] *
                tau ** self.tk[self.upper1:self.upper2] *
                np.exp(-delta * self.lk[self.upper1:self.upper2]) *
                (self.tk[self.upper1:self.upper2]) *
                (self.tk[self.upper1:self.upper2]) - 1.0).sum()

        sum3 = (self.nk[self.upper2:self.upper3] * delta ** self.dk[self.upper2:self.upper3] * \
               tau ** self.tk[self.upper2:self.upper3] * \
               np.exp(-self.etak[self.upper2:self.upper3] *
               (delta - self.epsk[self.upper2:self.upper3]) ** 2.0 -
               self.betak[self.upper2:self.upper3] *
               (tau - self.gammak[self.upper2:self.upper3]) ** 2.0) * \
               ((self.tk[self.upper2:self.upper3] - 2.0 * self.betak[self.upper2:self.upper3] *
               tau * (tau - self.gammak[self.upper2:self.upper3]) ** 2.0) -
                self.tk[self.upper2:self.upper3] - 2.0 * self.betak[self.upper2:self.upper3] *
                tau ** 2.0)).sum()
        return sum1 + sum2 + sum3
# ----------------------------------------------------------------------------

    def second_alpha_delta_partial(self, density: float, temperature: float) -> float:
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The temperature in units of Kelvins
        :return derivative: The second partial of alpha one with respect
                            to tau

        This function determines the value of the second derivative of alpha
        one with respect to tau according to Eq. 82 of Ref. 1. as shown below

        .. math::
           \\delta^2\\left(\\frac{\partial^2\\alpha^r}{\partial\\delta^2} \\right)_{\\tau}
        """
        delta = density / self.critical_density
        tau = self.critical_temperature / temperature
        sum1 = (self.nk[:self.upper1] * delta ** self.dk[:self.upper1] *
                tau ** self.tk[:self.upper1] *
                (self.dk[:self.upper1] * (self.dk[:self.upper1] - 1.0))).sum()

        sum2 = (self.nk[self.upper1:self.upper2] * delta ** self.dk[self.upper1:self.upper2] *
                tau ** self.tk[self.upper1:self.upper2] *
                np.exp(-delta ** self.lk[self.upper1:self.upper2]) *
                ((self.dk[self.upper1:self.upper2] -
                  self.lk[self.upper1:self.upper2] *
                  delta ** self.lk[self.upper1:self.upper2]) *
                 (self.dk[self.upper1:self.upper2] - 1.0 -
                  self.lk[self.upper1:self.upper2] *
                  delta ** self.lk[self.upper1:self.upper2]) -
                 self.lk[self.upper1:self.upper2] ** 2.0 *
                 delta ** self.lk[self.upper1:self.upper2])).sum()

        sum3 = (self.nk[self.upper2:] * delta ** self.dk[self.upper2:] *
                tau ** self.tk[self.upper2:] *
                np.exp(-self.etak[self.upper2:] *
                (delta - self.epsk[self.upper2:]) ** 2.0 -
                self.betak[self.upper2:] *
                (tau - self.gammak[self.upper2:]) ** 2.0) *
                ((self.dk[self.upper2:] - 2.0 * self.etak[self.upper2:] *
                  delta * (delta - self.epsk[self.upper2:])) ** 2.0 -
                 self.dk[self.upper2:] - 2.0 * self.etak[self.upper2:] *
                 delta ** 2.0)).sum()
        return sum1 + sum2 + sum3
# ----------------------------------------------------------------------------

    def second_alpha_tau_delta_partial(self, density: float, temperature: float) -> float:
        """

        :param density: The density in units of moles per cubic decimeter
        :param temperature: The temperature in units of Kelvins
        :return derivative: The second partial of alpha one with respect
                            to tau

        This function determines the value of the second derivative of alpha
        one with respect to tau according to Eq. 86 of Ref. 1. as shown below

        .. math::
           \\delta^2\\left(\\frac{\partial^2\\alpha^r}{\partial\\delta\partial\\tau} \\right)
        """
        delta = density / self.critical_density
        tau = self.critical_temperature / temperature

        sum1 = (self.nk[:self.upper1] * delta ** self.dk[:self.upper1] *
                tau ** self.tk[:self.upper1] *
                (self.dk[:self.upper1] * self.tk[:self.upper1])).sum()

        sum2 = (self.nk[self.upper1:self.upper2] *
                delta ** self.dk[self.upper1:self.upper2] *
                tau ** self.tk[self.upper1:self.upper2] *
                np.exp(-delta ** self.lk[self.upper1:self.upper2]) *
                (self.tk[self.upper1:self.upper2] *
                 (self.dk[self.upper1:self.upper2] -
                  self.lk[self.upper1:self.upper2] *
                  delta ** self.lk[self.upper1:self.upper2]))).sum()

        sum3 = (self.nk[self.upper2:] *
                delta ** self.dk[self.upper2:] *
                tau ** self.tk[self.upper2:] *
                np.exp(-self.etak[self.upper2:] * (delta - self.epsk[self.upper2:]) ** 2.0 -
                       self.betak[self.upper2:] * (tau - self.gammak[self.upper2:]) ** 2.0) *
                ((self.dk[self.upper2:] - 2.0 * self.etak[self.upper2:] *
                  delta * (delta - self.epsk[self.upper2:])) * (self.tk[self.upper2:] -
                  2.0 * self.betak[self.upper2:] * tau * (tau - self.gammak[self.upper2:])))).sum()
        return sum1 + sum2 + sum3
# ============================================================================
# ============================================================================


class ThermoProps(Helmholtz):
    """
    This class determines thermodynamic properties of gasses using
    the Helmholtz energy formulations.  All equations
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
                 gammak: np.array, upper1: int, upper2: int, upper3: int,
                 helm_coefs: Dict[str, float], molar_mass: float):
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
        :param helm_coefs: The Helmholtz coefficients as a
                           dictionary with the following keys,
                           `a1`, `a1`, `a3`, `a4`, `a5`, `a6`,
                           `a7`, and `a8`
        :param molar_mass: The molar mass of the fluid in units of
                           grams per mole
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
        self.helm_coefs = helm_coefs
        self.molar_mass = molar_mass
        self.gas_constant = 8.314462618  # J/K-mol
        Helmholtz.__init__(self, critical_density, critical_temperature,
                           tk, nk, dk, lk, etak, epsk, betak, gammak, upper1,
                           upper2, upper3, helm_coefs)
# ----------------------------------------------------------------------------

    def pressure(self, density: float, temperature: float) -> float:
        """

        :param density: Density in units of grams per cubic centimeter
        :param temperature: The static density in units of Kelvins
        :return pressure: The static pressure in units of Pascals

        This function solves for the pressure with knowledge of
        density and temperature using hte relationship below,
        which was derived from Eq. 76 on page 65 of Ref. 2.

        .. math::
           P = \\rho R T\\left[1 + \delta \\left(\\frac{\partial
           \\alpha^r}{\partial \delta} \\right)_{\\tau} \\right]

        where;

        .. math::
           Z = 1 + \delta \\left(\\frac{\partial \\alpha^r}{\partial \delta} \\right)_{\\tau}
        """
        # Convert density from g/cc to moles/dm^3
        molar_density = density * 1000.0 / self.molar_mass
        z = 1.0 + self.first_alpha_delta_partial(molar_density, temperature)
        # multiply by 1000 to convert density from dm^-3 to m^-3
        pressure = 1000.0 * z * self.gas_constant * temperature * molar_density
        return pressure
# ----------------------------------------------------------------------------

    def temperature(self, density: float, pressure: float, max_num: int = 400,
                    tol: float = 1.0e-4):
        """

        :param density: Density in units of grams per cubic centimeter
        :param pressure: The static pressure in units of Pascals
        :param max_num: The maximum number of iterations in the solver
                        before failure is declared.  Defaulted to 400
        :param tol: The tolerance in the acceptable solution.  Defaulted
                    to 1.0e-4
        :return temperature: The static temperature in units of Kelvins

        This function solves for the temperature with knowledge of
        density and pressure using hte relationship below,
        which was derived from Eq. 76 on page 65 of Ref. 2.

        .. math::
           P = \\rho R T\\left[1 + \delta \\left(\\frac{\partial
           \\alpha^r}{\partial \delta} \\right)_{\\tau} \\right]

        where;

        .. math::
           Z = 1 + \delta \\left(\\frac{\partial \\alpha^r}{\partial \delta} \\right)_{\\tau}
        """
        # Convert density from g/cc to moles/dm^3
        temperature = self._bisect_temp(0.0, 1e5, density, pressure, tol,
                                        max_num)
        return temperature
# ----------------------------------------------------------------------------

    def density(self, pressure: float, temperature: float, max_num: int = 400,
                    tol: float = 1.0e-4):
        """

        :param pressure: The static pressure in units of Pascals
        :param temperature: The static Temperature in units of Kelvins
        :param max_num: The maximum number of iterations in the solver
                        before failure is declared.  Defaulted to 400
        :param tol: The tolerance in the acceptable solution.  Defaulted
                    to 1.0e-4
        :return density: The density in units of grams per cubic centimeter

        This function solves for the density with knowledge of
        temperature and pressure using hte relationship below,
        which was derived from Eq. 76 on page 65 of Ref. 2.

        .. math::
           P = \\rho R T\\left[1 + \delta \\left(\\frac{\partial
           \\alpha^r}{\partial \delta} \\right)_{\\tau} \\right]

        where;

        .. math::
           Z = 1 + \delta \\left(\\frac{\partial \\alpha^r}{\partial \delta} \\right)_{\\tau}
        """
        density = self._bisect_dens(1e-6, 10000.0, temperature, pressure, tol,
                                    max_num)
        return density
# ----------------------------------------------------------------------------

    def compressibility_factor(self, pressure: float, temperature: float) -> float:
        """

        :param pressure: The static pressure in units of Pascals
        :param temperature: The static temperature in units of Kelvins
        :return z: The compressibility factor
        """
        density = self.density(pressure, temperature)
        molar_density = density * 1000.0 / self.molar_mass
        z = 1.0 + self.first_alpha_delta_partial(molar_density, temperature)
        return z
# ----------------------------------------------------------------------------

    def internal_energy(self, pressure: float, temperature: float) -> float:
        """

        :param pressure: The static pressure in units of Pascals
        :param temperature: The static temperature in units of Kelvins
        :return u: The internal energy in units of J/kg
        """
        z = self.compressibility_factor(pressure, temperature)
        density = self.density(pressure, temperature)
        molar_density = density * 1000.0 / self.molar_mass
        part1 = self.first_alpha_tau_zero_partial(temperature)
        part2 = self.first_alpha_tau_one_partial(molar_density, temperature)
        u = (part1 + part2) * self.gas_constant * temperature
        return u / (1000.0 * self.molar_mass)
# ============================================================================
# Private functions

    def _bisect_temp(self, low: float, high: float, dens: float, pres: float,
                     tol: float = 1.0e-3, max_num: int = 400):
        """

        :param low: The low value for the solution
        :param high: The upper value for the solution
        :param dens: The density in g/cc
        :param pres: The pressure in units of Pascals
        :param tol: The tolerance for the solution
        :param max_num: The maximum number of iterations
        :return:
        """
        upper = pres + pres * tol
        lower = pres - pres * tol
        for i in range(max_num):
            # define window
            start = low + (high-low) / 2.0
            res = self.pressure(dens, start)
            if res > upper:
                high = start
            elif res < lower:
                low = start
            else:
                return start
        message = 'FATAL ERROR: _bisect_temp function reached maximum iterations of '
        message += str(max_num)
        sys.exit(message)
# ----------------------------------------------------------------------------

    def _bisect_dens(self, low: float, high: float, temp: float, pres: float,
                     tol: float = 1.0e-3, max_num: int = 400):
        """

        :param low: The low value for the solution
        :param high: The upper value for the solution
        :param temp: The temperature in units of Kelvins
        :param pres: The pressure in units of Pascals
        :param tol: The tolerance for the solution
        :param max_num: The maximum number of iterations
        :return:
        """
        upper = pres + pres * tol
        lower = pres - pres * tol
        for i in range(max_num):
            # define window
            start = low + (high-low) / 2.0
            res = self.pressure(start, temp)
            if res > upper:
                high = start
            elif res < lower:
                low = start
            else:
                return start
        message = 'FATAL ERROR: _bisect_dens function reached maximum iterations of '
        message += str(max_num)
        sys.exit(message)
# ============================================================================
# ============================================================================
# eof
