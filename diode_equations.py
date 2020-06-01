""" Physics equations describing diode behavior
"""
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.special import lambertw

from diode_utils import *


def ideal_diode_eq(v_ca, i_s, m, T):
    """Shockley Diode equation

    Args:
        v_ca (float): Cathode-Anode voltage [V].
        i_s_log (float): Saturation current [A].
        m (float): Ideality factor.
        T (float): Temperature [T].

    Returns:
        (float): Diode current I_C [A]
    """
    i_c = i_s * (np.exp(v_ca * const.e)/(m * const.k * T) - 1)
    return i_c

def get_ideal_diode_eq_log(T):
    """ Wrapper function for ideal_diode_eq_log.

    Needed for scipy.optimize.curve_fit()
        TODO: Clarify description! more elegant solution?
    """
    def ideal_diode_eq_log_internal(v_ca, i_s_log, m):
        """Log of Shockley Diode equation

        Log of exponential equation facilates curve fitting
        Shockley Diode equation:
        i_c = i_s * (exp(v_ca * const.e)/(m * const.k * T) - 1)
        For v_ca >> v_t = .(const.k * T)/const.e = 0.026V * T/300K:
        ic approxeq i_s * exp(v_ca / (v_t * m))
        log(ic) approxeq log(i_s) + v_ca / (v_t * m)

        TODO: handle Temperature! (automatically ?)
        Args:
            v_ca (float): Cathode-Anode voltage.
            i_s_log (float): Log of saturation current.
            m (float): Ideality factor.
            T (float): Temperature in Kelvin, defaults to 298.0

        Returns:
            i_c_log (float): logarithm of diode current
        """
        i_c_log = i_s_log + v_ca * (const.e/(m * const.k * T))
        return i_c_log
    return ideal_diode_eq_log_internal


def ohmic_resistance_diode(v_ca, i_c, i_s, m, T=298.0):
    """TODO: Does not work, delete??
    Args:
        v_ca (float): Cathode-Anode voltage.
        i_c (float): Diode current.
        i_s (float): Saturation current.
        m (float): Ideality factor (model parameter)
        T (float): Temperature in Kelvin, defaults to 298.0

    Returns:
        [type]: [description]
    """
    r_D = (v_ca - np.log((i_c + i_s) / i_s) * const.k * T / const.e) / i_c
    return r_D


def i_c_eq_d_r(v_ca, i_s, m, T, r_S):
    """Calculate current of an ideal diode in series with a resistor

    Args:
        v_ca (float): Cathode-Anode voltage [V].
        i_s (float): Saturation current [A].
        m (float): Ideality factor (model parameter)
        T (float): Temperature [K], defaults to 298.0
        r_S (float): Ohmic diode resistance

    Returns:
        i_c_a  (float array): Diode current
    """
    # Ideal diode and ohmic resistance in series:
    # ln((i_c+i_s)/i_s) * v_t + i_c*r_S -v_ca = 0
    # sympy.solve(log((x+a)/a)*b+c*x-d, x) = [(-a*c + 
    #       b*LambertW(a*c*exp((a*c + d)/b)/b))/c]
    v_t = (const.k * T *m) / const.e
    i_c = ((-i_s * r_S + v_t * lambertw(i_s*r_S * np.exp((i_s*r_S + v_ca)/v_t)/v_t))/r_S)
    i_c = np.real(i_c)      # Avoid warning; imaginary part is already zero
    return i_c


def ideal_diode_model(v_ca_a , i_c_a , vca_lim_lower=0.65,
                          vca_lim_upper=0.8, T=298.0):
    """Calculate a best fit model for the Shockley Diode equation

    Args:
        v_ca_a  (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        vca_lim_lower (float, optional):
            Lower limit in Volt of range the model is based on.
            Defaults to 0.65.
        vca_lim_upper (float, optional): [description].
            Upper limit in Volt of range the model is based on.
            Defaults to 0.8.

    Returns:
        i_s (float): Saturation current (model parameter).
        m (float): Ideality factor (model parameter)
    """
    v_ca_cropped, i_c_cropped = crop_data_range_to_x(v_ca_a , i_c_a ,
                                    vca_lim_lower, vca_lim_upper)

    log_vector = np.vectorize(np.log)
    diode_eq_T = get_ideal_diode_eq_log(T)
    p_opt, pcov = curve_fit(diode_eq_T, v_ca_cropped, log_vector(i_c_cropped))
    i_s = np.exp(p_opt[0])      # Result of ideal_diode_eq_log is log(i_s)
    m = p_opt[1]

    return (i_s, m)


def diode_capacitance_TT_eq(i_c, tt):
    """Diode capacitance as function of diode current and transit time

    Diffusion capacitance is linearly dependent on diode current.
    Args:
        i_c (float: Diode current.
        tt (float): Transit time.

    Returns:
        (float): Diode capacitance.
    """
    c_ca = tt * i_c
    return c_ca


def diode_capacitance_model(v_ca_a , i_c_a , c_ca_a , vca_lim_lower=0.65,
                            vca_lim_upper=0.8):
    """Calculate a best fit model for the diffusion capacitance

    Args:
        v_ca_a  (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        c_ca_a  (float array): Diode capacitance.
        vca_lim_lower (float, optional):
            Lower limit in Volt of range the model is based on.
            Defaults to 0.65.
        vca_lim_upper (float, optional): [description].
            Upper limit in Volt of range the model is based on.
            Defaults to 0.8.

    Returns:
        tt (float): Transit time.
    """
    v_ca_cropped, c_ca_cropped = crop_data_range_to_x(v_ca_a , c_ca_a ,
                                            vca_lim_lower, vca_lim_upper)
    v_ca_cropped, i_c_cropped = crop_data_range_to_x(v_ca_a , i_c_a ,
                                            vca_lim_lower, vca_lim_upper)
    # Map to log to reduce numerical error
    # log_vector = np.vectorize(np.log)
    # p_opt, pcov = curve_fit(diode_capacitance_TT_eq, log_vector(i_c_cropped),
    #                         log_vector(c_ca_cropped))
    p_opt, pcov = curve_fit(diode_capacitance_TT_eq, i_c_cropped,
                            c_ca_cropped)
    tt = p_opt[0]   # Transit time
    return tt


def diode_saturation_current(i_s_0, T):
    """Temperature dependent saturation current.

    As described by Michael Schroter in hicumL2V2p4p0_internal.va  
    (line 1150) for an internal Base collector diode saturation current:
    ibcis_t = ibcis*exp(zetabci*ln_qtt0+vgc/VT*(qtt0-1)) 
    Args:
        i_s_0 (float): I_S [A] at T=300.15K.
        T ([type]): Temperature [K].

    Returns:
        float: I_S(T) [A]
    """
    zetaci = 0.0                # Line 813
    # Line 824 "Coefficient K1 in T-dependent band-gap equation":
    f1vg = -1.02377e-4       
    mg = 3 - const.e * f1vg / const.k
    zetabci = mg + 1 - zetaci   # Line 1025
    vgc = 1.17  # Collector bandgap voltage silicon
    VT = (const.k * T) / const.e
    T_nom = 300.15  # Nominal temperature [K]
    qtt0 = T / T_nom
    ln_qtt0 = np.log(qtt0)
    # i_s = i_s_0 * np.exp(zetabci * ln_qtt0 + vgc/VT * (qtt0-1))
    i_s = i_s_0 * np.exp(zetabci * ln_qtt0) * np.exp(vgc/VT * (qtt0-1))
    i_s = i_s_0 * np.exp(vgc/VT * (qtt0-1))     # Better results
    return i_s


def diode_saturation_current_log(i_s_0_log, T, zeta):
    """Log of temperature dependent saturation current.

    Linear equation to be used for curve fitting.
    As described by Michael Schroter in hicumL2V2p4p0_internal.va  
    (line 1150) for an internal Base collector diode saturation current:
    ibcis_t = ibcis*exp(zetabci*ln_qtt0+vgc/VT*(qtt0-1)) 
    Args:
        i_s_0 (float): I_S [A] at T=300.15K.
        T ([type]): Temperature [K].

    Returns:
        float: I_S(T) [A]
    """
    # zetaci = 0.0                # Line 813
    # # Line 824 "Coefficient K1 in T-dependent band-gap equation":
    # f1vg = -1.02377e-4       
    # mg = 3 - const.e * f1vg / const.k
    # zetabci = mg + 1 - zetaci   # Line 1025
    # vgc = 1.17  # Bandgap voltage silicon
    # VT = (const.k * T) / const.e
    # T_nom = 300.15  # Nominal temperature [K]
    # qtt0 = T / T_nom
    # ln_qtt0 = np.log(qtt0)
    # i_s = i_s_0 * np.exp(zetabci * ln_qtt0 + vgc/VT * (qtt0-1))
    i_s_log = i_s_0_log + zeta * np.log(T/300.15) + ((1.17*const.e) / (const.k*T)) * (T/300.15-1)
    return i_s_log


def diode_saturation_current_0(i_s, T):
    """Saturation current at nominal temperature.

    As described by Michael Schroter in hicumL2V2p4p0_internal.va  
    (line 1150) for an internal Base collector diode saturation current:
    ibcis_t = ibcis*exp(zetabci*ln_qtt0+vgc/VT*(qtt0-1)) 
    TODO Reacts too strongly to slight deviations in temperature 
        (298.0K vs 300.15K)
    Args:
        i_s (float): I_S [A] at T.
        T ([type]): Temperature [K].

    Returns:
        float: I_S(T=300.15K) [A]
    """
    zetaci = 0.0                # Line 813
    # Line 824 "Coefficient K1 in T-dependent band-gap equation":
    f1vg = -1.02377e-4       
    mg = 3 - const.e * f1vg / const.k
    zetabci = mg + 1 - zetaci   # Line 1025
    vgc = 1.17  # Bandgap voltage silicon
    VT = (const.k * 300) / const.e
    T_nom = 300.15  # Nominal temperature [K]
    qtt0 = T / T_nom
    ln_qtt0 = np.log(qtt0)
    # i_s = i_s_0 * np.exp(zetabci * ln_qtt0 + vgc/VT * (qtt0-1))
    exponent = vgc/VT * (qtt0-1)
    i_s_0 = i_s / np.exp(exponent)     # Better results
    return i_s_0


def calc_rd_deriv_a(v_ca_a , i_c_a ):
    """Resistance array as a derivative of voltage over current.

    TODO Check input
    Args:
        v_ca_a  (float array): Cathode-Anode voltage [V].
        i_c_a  (float array): Diode current.

    Returns:
        (float array): Differential resistance r_D [Ohm]
    """
    r_D = np.gradient(v_ca_a , i_c_a )
    return r_D
