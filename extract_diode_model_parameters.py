#! /usr/bin/python3

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.special import lambertw


def process_diode_measurement(measurements_fname='data.json', results_fname='model.json', plots_dir='figures' ):
    """Extract diode model parameters from JSON file with measurements.
    
    Structure of imported JSON File:
    Dict:
        meas_run(run_name, data) (tuple(string, dict)): Measurement run
        run_name (string): Name of test run, e.g. "T298.0K"
        data[phys_quantity: values] (dict[string: list]): 
        Dictionary of diode capacitance ('C_CA'),
        current ('I_C) and voltage ('V_CA') values
    Args:
        measurements_fname (string): File name of a JSON file 
        
    Returns:
        model_parameters_fname (string): File name of a new JSON file
    """
    with open(measurements_fname, 'r') as myfile:
        measurements = json.load(myfile)
    
    #TODO Guarantee correct order of dict entries for Python < 3.7      
    for measurement in measurements.items():
        v_ca = measurement[1]['V_CA'][:]
        i_c = measurement[1]['I_C'][:] 
        c_ca =  measurement[1]['C_CA'][:]
        T = float(measurement[0][1:5])     # e.g. measurement[0] = "T298.0K"
        model_params = diode_model_params_isotherm(v_ca, c_ca, i_c, T)
        plot_vca_ic(v_ca, i_c, model_params)
        plot_vca_cca(v_ca, c_ca, i_c, model_params)
   

def diode_model_params_isotherm(v_ca, c_ca, i_c, T):
    """Extract diode model parameters at a fixed temperature

    Args:
        v_ca (float array): Cathode-Anode voltage [V.]
        i_c (float array): Diode current [A].
        v_ca (float array): Diode capacitance [F].
        T (float): Temperature [K].

    Returns:
        (dict): Dictionary of model parameters T, I_S, m, R_S, TT
    """
    # Curve fit where I_c curve is purely exponential
    vca_lim_lower_ic = 0.65
    vca_lim_upper_ic = 0.75
    i_s, m= ideal_diode_model(v_ca, i_c, vca_lim_lower_ic, vca_lim_upper_ic, T)
        
    print('Model parameters for T = ' + str("{:.1f}".format(T)), 'K: I_S = ' + 
          str(i_s) + ', m = ' + str(m))
    
    # TODO: Is  mean of differential resistance better than simple 
    # quotient of differences?
    # # Calculate differential resistance r_D
    # r_D = np.zeros(len(v_ca))
    # # Backward derivative d(v_ca)/d(i_c)
    # for i in range(1, len(r_D)):
    #     r_D[i] = (v_ca[i] - v_ca[i-1])/(i_c[i] - i_c[i-1])
    # r_D[0] = r_D[1]     # No backward derivative possible for first value

    # Mean of differential resistance between V_CA=0.9V and 1.0V
    # v_ca_cropped_r, r_cropped = crop_data_range_to_x(v_ca, r_D, 0.9, 1.0)
    # r_ohm = np.mean(r_cropped)
    # print('R_D =', str(r_ohm))
    
    # Simple difference between V_CA=0.9V, V_C=1.0V
    r_ohm_simple = (v_ca[70] - v_ca[50])/(i_c[70] - i_c[50])
    print('R_D_simple =', str(r_ohm_simple))
    
    # Calculate C_CA model
    vca_lim_lower_cca = 0.65
    vca_lim_upper_cca = 1.0
    tt = diode_capacitance_model(v_ca, i_c, c_ca, vca_lim_lower_cca, vca_lim_upper_cca)
    
    print('Transit time for T = ' + str("{:.1f}".format(T)), 'K: TT =', 
          "{:.4g}".format(tt), 's.')
    
    # TODO: Explanations of parameters?
    model_params = {'T': T, 'I_S': i_s, 'm': m, 'R_S': r_ohm_simple, 
                    'TT': tt, 'vca_lim_lower_ic': vca_lim_lower_ic, 
                    'vca_lim_upper_ic': vca_lim_upper_ic,
                    'vca_lim_lower_cca': vca_lim_lower_cca, 
                    'vca_lim_upper_cca': vca_lim_upper_cca,
                    }
    return model_params
   
      
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
    """ Wrapper function for ideal_diode_eq_log.
    
    Needed for scipy.optimize.curve_fit()
        TODO: Clarify description! more elegant solution?
    """
    def ideal_diode_eq_log_internal(v_ca, i_s_log, m):
        """Log of Shockley Diode equation
        
        Log of exponential equation facilates curve fitting
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
            i_c_log (float): logarithm of diode current
        """
        i_c_log = i_s_log + v_ca * (const.e/(m * const.Boltzmann * T))
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
    r_D = (v_ca - np.log((i_c + i_s) / i_s) * const.Boltzmann * T / const.e) / i_c
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
        i_c (float array): Diode current
    """
    # Ideal diode and ohmic resistance in series:
    # ln((i_c+i_s)/i_s) * v_t + i_c*r_S -v_ca = 0
    # In [12]: solve(log((x+a)/a)*b+c*x-d, x)                                                         
    # Out[12]: [(-a*c + b*LambertW(a*c*exp((a*c + d)/b)/b))/c]
    v_t = (const.Boltzmann * T *m) / const.e 
    i_c = ((-i_s * r_S + v_t * lambertw(i_s*r_S * np.exp((i_s*r_S + v_ca)/v_t)/v_t))/r_S)
    i_c = np.real(i_c)      # Avoid warning; imaginary part is already zero
    return i_c


def ideal_diode_model(v_ca, i_c, vca_lim_lower=0.65,     
                          vca_lim_upper=0.8, T=298.0):
    """Calculate a best fit model for the Shockley Diode equation

    Args:
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
    """
    v_ca_cropped, i_c_cropped = crop_data_range_to_x(v_ca, i_c, 
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


def diode_capacitance_model(v_ca, i_c, c_ca, vca_lim_lower=0.65,     
                            vca_lim_upper=0.8):
    """Calculate a best fit model for the diffusion capacitance

    Args:
        v_ca (float array): Cathode-Anode voltage.
        i_c (float array): Diode current.
        c_ca (float array): Diode capacitance.
        vca_lim_lower (float, optional): 
            Lower limit in Volt of range the model is based on. 
            Defaults to 0.65.
        vca_lim_upper (float, optional): [description]. 
            Upper limit in Volt of range the model is based on. 
            Defaults to 0.8.

    Returns:
        tt (float): Transit time.
    """
    v_ca_cropped, c_ca_cropped = crop_data_range_to_x(v_ca, c_ca, 
                                            vca_lim_lower, vca_lim_upper)
    v_ca_cropped, i_c_cropped = crop_data_range_to_x(v_ca, i_c, 
                                            vca_lim_lower, vca_lim_upper)
    # Map to log to reduce numerical error
    # log_vector = np.vectorize(np.log)
    # p_opt, pcov = curve_fit(diode_capacitance_TT_eq, log_vector(i_c_cropped), 
    #                         log_vector(c_ca_cropped))
    p_opt, pcov = curve_fit(diode_capacitance_TT_eq, i_c_cropped, 
                            c_ca_cropped)
    tt = p_opt[0]   # Transit time

    return tt


def plot_vca_cca(v_ca, c_ca, i_c, model_params, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures'):
    """Plot diode capacitance over diode voltage
    
    Args:
        v_ca (float array): Cathode-Anode voltage.
        i_c (float array): Diode current.
        v_ca (float array): Diode capacitance.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """        
    T = model_params['T']
    i_s = model_params['I_S']
    m = model_params['m']
    r_s = model_params['R_S']
    vca_lim_lower_cca = model_params['vca_lim_lower_cca']
    vca_lim_upper_cca = model_params['vca_lim_upper_cca']
    tt = model_params['TT']
    
    # Calculate I_C_R model based on a series of ideal diode ohmic resistance
    i_c_r = np.zeros(len(v_ca))
    for i in range(len(i_c_r)):
        i_c_r[i] = i_c_eq_d_r(v_ca[i], i_s, m, T, r_s)
        
    c_ca_model = np.zeros(len(v_ca))
    for i in range(len(c_ca_model)):
        c_ca_model[i] = diode_capacitance_TT_eq(i_c_r[i], tt) 
        
    label_cca_model = ('C_CA_model = TT * I_C_model\n TT =' + 
                       "{:.4g}".format(tt) + 's\n based on ' + 
                       str(vca_lim_lower_cca) + 'V <= V_CA <= ' + 
                       str(vca_lim_upper_cca) + 'V')
          
    # Plot IC over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.plot(v_ca, i_c, '.', label='I_C')
    ax1.set_ylabel('I_C [A]')

    # Prepare Labels of ax1
    labelnames =  label_ic 
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')  # TODO Get rid of 'Line2D()'

    # Plot C_CA over V_CA
    ax2 = ax1.twinx()
    label_cca = ax2.plot(v_ca, c_ca, 'rx', label='C_CA')
    lab_cca_model = ax2.plot(v_ca, c_ca_model, 'g-', label=label_cca_model)
    ax2.set_ylabel('C_CA [F]')

    # Prepare Labels of ax2
    labelnames =  label_cca + lab_cca_model
    labels = [l.get_label() for l in labelnames]
    ax2.legend(labelnames, labels, loc='lower right')  # TODO Get rid of 'Line2D()'

    fname_pdf = plot_dir + '/VCA_CCA_T' + "{:.1f}".format(T) + '.png'
    plt.savefig(fname_pdf)
    plt.clf()


def plot_vca_ic(v_ca, i_c, model_params, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures'):
    """Plot diode current over diode voltage
    
    Args:
        v_ca (float array): Cathode-Anode voltage.
        i_c (float array): Diode current.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    T = model_params['T']
    i_s = model_params['I_S']
    m = model_params['m']
    r_s = model_params['R_S']
    vca_lim_lower_ic = model_params['vca_lim_lower_ic']
    vca_lim_upper_ic = model_params['vca_lim_upper_ic']

    diode_eq_T = get_ideal_diode_eq_log(T)
    i_c_ideal_diode_model = np.zeros(len(v_ca))
    for i in range(len(i_c_ideal_diode_model)):
        i_c_ideal_diode_model[i] = np.exp(diode_eq_T(v_ca[i], np.log(i_s), m)) 
    
    label_ideal_diode_model=(
        'I_C_ideal = I_S * (exp (V_CA/(V_T*m)) -1):\n I_S = ' +
        "{:.4g}".format(i_s) + ' A, m = ' + "{:.4g}".format(m) +
        '\n based on ' + str(vca_lim_lower_ic) + 'V <= V_CA <= ' + 
        str(vca_lim_upper_ic) + 'V')

    # # Calculate differential resistance r_D
    r_D = np.zeros(len(v_ca))
    # Backward derivative d(v_ca)/d(i_c)
    for i in range(1, len(r_D)):
        r_D[i] = (v_ca[i] - v_ca[i-1])/(i_c[i] - i_c[i-1])
    r_D[0] = r_D[1]     # No backward derivative possible for first value

    # Calculate I_C_R model based on a series of ideal diode ohmic resistance
    i_c_r = np.zeros(len(v_ca))
    for i in range(len(i_c_r)):
        i_c_r[i] = i_c_eq_d_r(v_ca[i], i_s, m, T, r_s)
    
    label_diode_ohmic = 'I_C_model (R_S = ' + "{:.4g}".format(r_s) + ' Ohm)'
    label_r = 'r_D = d(I_C)/d(V_CA)'
    
    # Plot I_C and models over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.semilogy(v_ca, i_c, '.', label='I_C')
    label_ic_model = ax1.semilogy(v_ca, i_c_ideal_diode_model, '-b', 
                                  label=label_ideal_diode_model)
    label_ic_r_model = ax1.semilogy(v_ca, i_c_r, '--r', label=label_diode_ohmic)
    
    ax1.set_xlabel('V_CA [V]')
    ax1.set_ylabel('I_C [A]')
    
    # Plot r_D over V_CA
    axr = ax1.twinx()
    axr.set_ylabel('Resistance [Ohm]')
    axr.set_ylim([0, 10])
    label_r = axr.plot(v_ca, r_D, 'g-.', label=label_r)
    axr.legend(label_r, loc='lower right')  # TODO: avoid 'Line2D()
    
    # Prepare Labels of ax1
    labelnames =  label_ic + label_ic_model + label_ic_r_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    fname_pdf = plot_dir + '/VCA_IC_T' + "{:.1f}".format(T) + '.png'
    plt.savefig(fname_pdf)
    plt.clf()


def main():
  process_diode_measurement('/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/PythonExtraction/data.json')


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
