#! /usr/bin/python3

# pylint: disable=unused-wildcard-import
import json
import numpy as np

# Ensure custom functions can be found.
# import sys
# sys.path.append('.')

# Use diode functions of Python files in working directory
from diode_plots import  *
from diode_modelling import *


def process_diode_measurement(measurements_fname='data.json', 
    results_fname='model.json', plot_dir='figures' ):
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
    with open(measurements_fname, 'r') as f:
        measurements = json.load(f)

    plot_measurements_overview(measurements)
    
    models = []
    # Prepare variables for temperature dependence of I_S
    i_s_temp_a  = np.zeros(len(measurements.items()))   # Saturation currents at
    T_i_s_a  = np.zeros(len(measurements.items()))      # different temperatures
    i = 0
    #TODO Guarantee correct order of dict entries for Python < 3.7
    for measurement in measurements.items():
        v_ca_a = measurement[1]['V_CA'][:]
        i_c_a  = measurement[1]['I_C'][:]
        c_ca_a  =  measurement[1]['C_CA'][:]
        T = float(measurement[0][1:5])     # e.g. measurement[0] = "T298.0K"
        model = DiodeModelIsotherm(v_ca_a, i_c_a , c_ca_a , T)
        plot_vca_ic(v_ca_a, i_c_a , model, plot_dir=plot_dir)
        plot_vca_cca(v_ca_a, i_c_a , c_ca_a , model, plot_dir=plot_dir)
        models.append(model)
        i_s_0 = diode_saturation_current_0(model.i_s, T)
        print('I_S = ', model.i_s, 'I_S0_model =', i_s_0)  
        i_s_model = diode_saturation_current(i_s_0, model.T)
        i_s_temp_a[i] = model.i_s
        T_i_s_a[i] = model.T 
        print(' For T = ' + str(model.T) + 'K: I_S = ' + str(model.i_s) +
              ' A, I_S_model(T) = ' + str(i_s_model) + ' A.')
        i += 1
        if (290. < model.T < 310.15):
            print('Measurement series at T =', model.T, 
                  'K has less than 10 K difference to T_nom = 300.15 K', 
                  'and will be used as reference.')
            # Set variables for base measurements (T ~ T_0 = 300.15K)
            v_ca_a_0 = v_ca_a
            i_c_a_0 = i_c_a
            c_ca_a_0 = c_ca_a
            # Plots for presentation, optional
            plot_vca_cca_for_presentation(v_ca_a, i_c_a , c_ca_a , model, 
                                          plot_dir=plot_dir)
            plot_vca_ic_ideal(v_ca_a, i_c_a , model, plot_dir=plot_dir)
            plot_vca_ic_r(v_ca_a, i_c_a , model, plot_dir=plot_dir)

    base_model = DiodeModel(v_ca_a_0, i_c_a_0 , c_ca_a_0 , T, T_i_s_a, 
                            i_s_temp_a)
    # Plot for presentation, optional
    plot_T_is(T_i_s_a, i_s_temp_a, base_model, plot_dir=plot_dir)
    
    with open(results_fname, 'w') as f:
        json.dump(base_model.params, f, ensure_ascii=True)


def main():
    with open('file_names.json', 'r') as f:
        fnames = json.load(f)
    
    process_diode_measurement(fnames["data"], results_fname=fnames["results"],
                               plot_dir=fnames["plot_dir"])


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
