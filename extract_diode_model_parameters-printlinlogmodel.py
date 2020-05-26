#! /usr/bin/python3

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const


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

    
    # Try curve fitting
    def func_ideal_diode(v_ca,  i_s, m):
        """Shockley Diode equation
        
        TODO: handle Temperature! (automatically ?)
        Args:
            v_ca (float): Cathode - Anode voltage
            i_s (float): Saturation current
            m (float): Ideality factor

        Returns:
            [type]: [description]
        """
        i_c = i_s * (np.exp(v_ca * const.e)/(m * const.Boltzmann * 298.0) - 1)
        return i_c
    
    def func_ideal_diode_log(v_ca,  i_s_log, m):
        """Log of Shockley Diode equation
        
        Valid for v_ca >> v_t = (const.Boltzmann * T)/const.e = 0.026V *T/300K
        TODO: handle Temperature! (automatically ?)
        Args:
            v_ca (float): Cathode - Anode voltage
            i_s_log (float): Log of saturation current
            m (float): Ideality factor

        Returns:
            [type]: [description]
        """
        i_c_log = i_s_log + v_ca * (const.e/(m * const.Boltzmann * 298.0))
        return i_c_log
    
    v_ca_cropped, i_c_cropped = crop_data_range_to_x(v_ca, i_c, 0.65, 0.8)
    
    log_vector = np.vectorize(np.log)
    p_opt, pcov = curve_fit(func_ideal_diode_log, v_ca_cropped, log_vector(i_c_cropped))
    i_s = np.exp(p_opt[0])      # Result of func_ideal_diode_log is log(i_s)
    m = p_opt[1]
    
    i_c_model_cropped = np.zeros(len(v_ca_cropped))
    for i in range(len(v_ca_cropped)):
        i_c_model_cropped[i] = np.exp(func_ideal_diode_log(v_ca_cropped[i], np.log(i_s), m)) 
        
    i_c_model = np.zeros(len(v_ca))
    for i in range(len(v_ca)):
        i_c_model[i] = np.exp(func_ideal_diode_log(v_ca[i], np.log(i_s), m)) 
        
    print('Model parameters for ', meas_run[0], ': I_S =', i_s, 'm =', m)
    
    # Plot I_C over V_CA
    fig, ax1 = plt.subplots()
    
    label_ic = ax1.semilogy(v_ca, i_c, '.', label='I_C')
    label_ic_model = ax1.semilogy(v_ca, i_c_model, '-', label=(
                     'I_C_model = I_S * (exp (V_CA/(V_T*m)) -1):\n I_S = '
                     + "{:.4g}".format(i_s) + ' A\nm = ' + 
                     "{:.4g}".format(m)))
    # ax2 = ax1.twinx()
    # label_cca = ax2.plot(v_ca, c_ca, 'rx', label='C_CA')
    # ax2.set_ylabel('C_CA [F]')
    
    ax1.set_xlabel('V_CA [V]')
    ax1.set_ylabel('I_C [A]')
    
    # Prepare Labels of different axes
    # labelnames =  label_ic + label_cca + label_ic_model
    labelnames =  label_ic + label_ic_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    fname_pdf = plot_dir + '/' + meas_run[0] + '_IC_VCA.png'
    plt.savefig(fname_pdf)
    plt.clf()

    # # Plot C_CA over V_CA
    # plt.plot(v_ca, v_ca, '.')
    # plt.xlabel('V_CA [V]')
    # plt.ylabel('C_CA [F]')
    # fname_pdf = plot_dir + '/' + meas_run[0] + '_V_CA.png'
    # plt.savefig(fname_pdf)
    # plt.clf()

def main():
  extract_diode_model_parameters('/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/PythonExtraction/data.json')


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
