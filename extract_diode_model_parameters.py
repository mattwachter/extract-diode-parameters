#! /usr/bin/python3

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.special import lambertw


def extract_diode_model_parameters(measurements_fname='data.json'):
    """Extract diode model parameters from JSON file with measurements.
    

    Args:
        measurements_fname (string): File name of a JSON file 
        
    Returns:
        model_parameters_fname (string): File name of a new JSON file
    """
    with open(measurements_fname, 'r') as myfile:
        measurements = json.load(myfile)
    
    #TODO Guarantee correct order of dict entries for Python < 3.7      
    for temperature in measurements.items():
        plot_voltage_sweep(temperature)
      
      
def crop_data_range_to_x(xdata, ydata, lower, upper):
    """Crop two data data vectors so that the second corresponds to the first

    Args:
        xdata (1D array/list): 
        ydata (1D array/list): 
        lower (float): desired lower bound of xdata
        upper (float): desired upper bound of xdata

    Raises:
        ValueError: Length of xdata and ydata must be equal
        ValueError: Lower bound needs to be >= xdata[0]
        ValueError: Upper bound needs to <= xdata[-1]
        ValueError: xdata needs to be a monotonously rising sequence

    Returns:
        tuple of 1D arrays: cropped xdata and ydata
    """
    if (len(xdata) != len(ydata)):
        raise ValueError('Length of xdata and ydata must be equal!')
    
    if (lower < xdata[0]):
        raise ValueError('Lower bound needs to be equal or larger than first value of xdata !')
    
    if (upper > xdata[-1]):
        raise ValueError('Upper bound needs to equal or smaller than last value of xdata!')
    
    x_previous = xdata[0]
    for i in range(1, len(xdata)):
        if (xdata[i] <= x_previous):
            raise ValueError('xdata', xdata, 'needs to be a monotonously rising sequence!')
        else:
            x_previous = xdata[i]
    
    for i in range(len(xdata)):
        if (xdata[i] >= lower):
            index_lower = i
            break
        
    for i in (range(len(xdata) -1, -1, -1)):
        if (xdata[i] <= upper):
            index_upper = i
            break
        
    xdata_cropped = xdata[index_lower:index_upper]
    ydata_cropped = ydata[index_lower:index_upper]
    
    return (xdata_cropped, ydata_cropped)


def ideal_diode_eq_log(v_ca, i_s_log, m):
    """Log of Shockley Diode equation
    
    Shockley Diode equation:
    i_c = i_s * (exp(v_ca * const.e)/(m * const.Boltzmann * T) - 1)
    For v_ca >> v_t = .(const.Boltzmann * T)/const.e = 0.026V * T/300K:
    ic approxeq i_s * exp(v_ca / (v_t * m))
    log(ic) approxeq log(i_s) + v_ca / (v_t * m)

    TODO: handle Temperature! (automatically ?)
    Args:
        v_ca (float): Cathode-Anode voltage.
        i_s_log (float): Log of saturation current.
        m (float): Ideality factor.
        T (float): Temperature in Kelvin, defaults to 298.0

    Returns:
        [type]: [description]
    """
    i_c_log = i_s_log + v_ca * (const.e/(m * const.Boltzmann * 298.0))
    return i_c_log

def get_ideal_diode_eq_log(T):
    """ Wrapper function fort ideal_diode_eq_log.
    
    Needed for scipy.optimize.curve_fit()
        TODO: Clarify description! more elegant solution?
    """
    def ideal_diode_eq_log_internal(v_ca, i_s_log, m):
        """Log of Shockley Diode equation
        
        Shockley Diode equation:
        i_c = i_s * (exp(v_ca * const.e)/(m * const.Boltzmann * T) - 1)
        For v_ca >> v_t = .(const.Boltzmann * T)/const.e = 0.026V * T/300K:
        ic approxeq i_s * exp(v_ca / (v_t * m))
        log(ic) approxeq log(i_s) + v_ca / (v_t * m)

        TODO: handle Temperature! (automatically ?)
        Args:
            v_ca (float): Cathode-Anode voltage.
            i_s_log (float): Log of saturation current.
            m (float): Ideality factor.
            T (float): Temperature in Kelvin, defaults to 298.0

        Returns:
            [type]: [description]
        """
        i_c_log = i_s_log + v_ca * (const.e/(m * const.Boltzmann * T))
        return i_c_log
    return ideal_diode_eq_log_internal
    
def ohmic_resistance_diode(v_ca, i_c, i_s, m, T=298.0):
    """[summary]
    Args:
        v_ca (float): Cathode-Anode voltage.
        i_c (float): Diode current.
        i_s (float): Saturation current.
        m (float): Ideality factor (model parameter)
        T (float): Temperature in Kelvin, defaults to 298.0

    Returns:
        [type]: [description]
    """
    r = (v_ca - np.log((i_c + i_s) / i_s) * const.Boltzmann * T / const.e) / i_c
    return r 


def i_c_eq_d_r(v_ca, i_s, m, T, r):
    """Calculate current of an ideal diode in series with a resistor
    
    Args:
        v_ca (float): Cathode-Anode voltage.
        i_s (float): Saturation current.
        m (float): Ideality factor (model parameter)
        T (float): Temperature in Kelvin, defaults to 298.0

    Returns:
        i_c (float array): Diode current
    """
    # Ideal diode and ohmic resistance in series:
    # ln((i_c+i_s)/i_s) * v_t + i_c*r -v_ca = 0
    # In [12]: solve(log((x+a)/a)*b+c*x-d, x)                                                         
    # Out[12]: [(-a*c + b*LambertW(a*c*exp((a*c + d)/b)/b))/c]
    v_t = (const.Boltzmann * T *m) / const.e 
    i_c = ((-i_s * r + v_t * lambertw(i_s*r * np.exp((i_s*r + v_ca)/v_t)/v_t))/r)
    i_c = np.real(i_c)      # Avoid warning; imaginary part is already zero
    return i_c


def ideal_diode_model(name, v_ca, i_c, vca_lim_lower=0.65,     
                          vca_lim_upper=0.8, T=298.0):
    """Calculate a best fit model for the Shockley Diode equation

    Args:
        name (string): Data set name.
        v_ca (float array): Cathode-Anode voltage.
        i_c (float array): Diode current.
        vca_lim_lower (float, optional): 
            Lower limit in Volt of range the model is based on. 
            Defaults to 0.65.
        vca_lim_upper (float, optional): [description]. 
            Upper limit in Volt of range the model is based on. 
            Defaults to 0.8.

    Returns:
        i_s (float): Saturation current (model parameter).
        m (float): Ideality factor (model parameter)
        ic_model (float array): 
            Values of IC as calculated with the model.
    """
    v_ca_cropped, i_c_cropped = crop_data_range_to_x(v_ca, i_c, 
                                    vca_lim_lower, vca_lim_upper)
    
    log_vector = np.vectorize(np.log)
    diode_eq_T = get_ideal_diode_eq_log(T)
    p_opt, pcov = curve_fit(diode_eq_T, v_ca_cropped, log_vector(i_c_cropped))
    i_s = np.exp(p_opt[0])      # Result of ideal_diode_eq_log is log(i_s)
    m = p_opt[1]
    
    i_c_model = np.zeros(len(v_ca))
    for i in range(len(i_c_model)):
        i_c_model[i] = np.exp(diode_eq_T(v_ca[i], np.log(i_s), m)) 
        
    print('Model parameters for ', name, ': I_S =', i_s, 'm =', m)
    return (i_s, m, i_c_model)


def plot_voltage_sweep(meas_run, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures'):
    """Plot diode current and capacitance over diode voltage
    
    Args:
        meas_run(run_name, data) (tuple(string, dict)): Measurement run
        run_name (string): Name of test run, e.g. "T298.0K"
        data[phys_quantity: values] (dict[string: list]): 
            Dictionary of diode capacitance ('C_CA'),
            current ('I_C) and voltage ('V_CA') values
    """
    v_ca = meas_run[1]['V_CA'][:]
    i_c = meas_run[1]['I_C'][:] 
    c_ca =  meas_run[1]['C_CA'][:]
    T = float(meas_run[0][1:5])     # e.g. meas_run[0] = "T298.0K"

    vca_lim_lower = 0.65
    vca_lim_upper = 0.75
    i_s, m, i_c_ideal_diode_model = ideal_diode_model(
        meas_run[0], v_ca, i_c, vca_lim_lower, vca_lim_upper, T)
    
    label_ideal_diode_model=(
        'I_C_model = I_S * (exp (V_CA/(V_T*m)) -1):\n I_S = ' +
        "{:.4g}".format(i_s) + ' A, m = ' + "{:.4g}".format(m) +
        '\n based on ' + str(vca_lim_lower) + 'V <= V_CA <= ' + 
        str(vca_lim_upper) + 'V')
    
    r = np.zeros(len(v_ca))
    for i in range(1, len(r)):
        # r[i] = ohmic_resistance_diode(v_ca[50], i_c[50], m, T)
        r[i] = (v_ca[i] - v_ca[i-1])/(i_c[i] - i_c[i-1])
    r[0] = r[1]     # No backward derivative possible for first value

    # v_ca_cropped_r, r_cropped = crop_data_range_to_x(v_ca, r, 0.9, 1.0)
    # r_ohm = np.mean(r_cropped)
    # print('R_D =', str(r_ohm))
    # Simple 'derivative' between V_CA=0.9V, V_C=1.0V
    r_ohm_simple = (v_ca[70] - v_ca[50])/(i_c[70] - i_c[50])
    print('R_D_simple =', str(r_ohm_simple))

    # Calculate I_C based on an ideal diode and an ohmic resistance
    ic_r = np.zeros(len(v_ca))
    for i in range(len(ic_r)):
        ic_r[i] = i_c_eq_d_r(v_ca[i], i_s, m, T, r_ohm_simple)
    
    label_diode_ohmic = 'I_C(R_S = ' + "{:.4g}".format(r_ohm_simple) + '\Ohm)'
    label_r = 'r_D = d(I_C)/d(V_CA)'
    
    # Plot I_C over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.semilogy(v_ca, i_c, '.', label='I_C')
    label_ic_model = ax1.semilogy(v_ca, i_c_ideal_diode_model, '-b', 
                                  label=label_ideal_diode_model)
    label_ic_r_model = ax1.semilogy(v_ca, ic_r, '--r', label=label_diode_ohmic)
    # ax2 = ax1.twinx()
    # label_cca = ax2.plot(v_ca, c_ca, 'rx', label='C_CA')
    # ax2.set_ylabel('C_CA [F]')
    
    ax1.set_xlabel('V_CA [V]')
    ax1.set_ylabel('I_C [A]')
    
    # Plot r over V_CA
    axr = ax1.twinx()
    axr.set_ylabel('Resistance [\Ohm]')
    axr.set_ylim([0, 10])
    label_r = axr.plot(v_ca, r, 'g-.', label=label_r)
    axr.legend(label_r, loc='lower right')  # TODO: avoid 'Line2D()
    
    # # Plot C_CA over V_CA
    # plt.plot(v_ca, v_ca, '.')
    # plt.xlabel('V_CA [V]')
    # plt.ylabel('C_CA [F]')
    # fname_pdf = plot_dir + '/' + meas_run[0] + '_V_CA.png'
    # plt.savefig(fname_pdf)
    # plt.clf()
    
    # Prepare Labels of ax1
    # labelnames =  label_ic + label_cca + label_ic_model
    labelnames =  label_ic + label_ic_model + label_ic_r_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    fname_pdf = plot_dir + '/VCA_IC_' + meas_run[0] + '.png'
    plt.savefig(fname_pdf)
    plt.clf()


def main():
  extract_diode_model_parameters('/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/PythonExtraction/data.json')


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
